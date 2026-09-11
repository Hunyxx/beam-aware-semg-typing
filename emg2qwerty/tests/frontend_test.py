from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from scripts.evaluate_frontend_sessions import (
    _json_config,
    _plain_metric_values,
    forward_session_in_chunks,
)
from torch import nn

from emg2qwerty.transcription.data import (
    SidePositionTargetBuilder,
    SidePositionWindowedEMGDataset,
)
from emg2qwerty.transcription.frontend import (
    ConformerCTC,
    conv_stack_out_length,
    FrontendOutput,
    MultiScaleRawEMGFrontEnd,
    SplitShareConformerCTC,
)
from emg2qwerty.transcription.lightning import side_position_losses
from emg2qwerty.transcription.position_labels import (
    key_to_side_position_names,
    LEFT_POSITION_CHARSET,
    RIGHT_POSITION_CHARSET,
)


def test_multiscale_frontend_preserves_same_padded_stride_length() -> None:
    frontend = MultiScaleRawEMGFrontEnd(
        input_channels=16,
        branch_out_channels=4,
        branch_kernels=(7, 11, 15),
        stride=5,
        output_channels=8,
    )
    assert [type(layer) for layer in frontend.fusion] == [
        nn.Conv1d,
        nn.InstanceNorm1d,
    ]
    for branch in frontend.branches:
        assert [type(layer) for layer in branch] == [
            nn.Conv1d,
            nn.InstanceNorm1d,
            nn.ReLU,
        ]
    inputs = torch.randn(2, 16, 101)
    outputs = frontend(inputs)
    assert outputs.shape == (2, 8, 21)


def test_final_model_has_only_release_heads_and_runs_on_cpu() -> None:
    model = SplitShareConformerCTC(
        hidden_dim=32,
        conv_channels=(16, 16, 16),
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        multiscale_branch_channels=4,
        use_side_position_aux=True,
    ).eval()
    assert set(dict(model.named_children())) == {
        "cnn_featurizer",
        "conformer_encoder",
        "output_projection",
        "left_position_head",
        "right_position_head",
        "position_activity_head",
    }

    inputs = torch.randn(2, 2, 16, 200)
    lengths = torch.tensor([200, 181])
    head_calls = {"left": 0, "right": 0, "activity": 0}

    def count_call(name: str):
        def hook(*_args) -> None:
            head_calls[name] += 1

        return hook

    handles = [
        model.left_position_head.register_forward_hook(count_call("left")),
        model.right_position_head.register_forward_hook(count_call("right")),
        model.position_activity_head.register_forward_hook(count_call("activity")),
    ]
    with torch.inference_mode():
        plain_log_probs, plain_lengths = model(inputs, lengths)
        flat_log_probs, flat_lengths = model(inputs.flatten(1, 2), lengths)
    assert head_calls == {"left": 0, "right": 0, "activity": 0}
    with torch.inference_mode():
        outputs = model.forward_with_aux(inputs, lengths)
    for handle in handles:
        handle.remove()

    expected_time = conv_stack_out_length(200)
    assert outputs.log_probs.shape[:2] == (expected_time, 2)
    assert outputs.output_lengths.tolist() == [10, 10]
    torch.testing.assert_close(plain_log_probs, outputs.log_probs)
    torch.testing.assert_close(flat_log_probs, outputs.log_probs)
    torch.testing.assert_close(plain_lengths, outputs.output_lengths)
    torch.testing.assert_close(flat_lengths, outputs.output_lengths)
    assert head_calls == {"left": 1, "right": 1, "activity": 1}
    assert outputs.left_position_logits.shape[-1] == 16
    assert outputs.right_position_logits.shape[-1] == 17
    assert outputs.activity_logits.shape[-1] == 1
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_position_mapping_and_time_targets() -> None:
    assert key_to_side_position_names("Key.space") == ("SPACE", "SPACE")
    assert key_to_side_position_names("q") == ("LP_UP", None)
    assert key_to_side_position_names("p") == (None, "RP_UP")

    builder = SidePositionTargetBuilder.from_keystrokes(
        [{"key": "q", "start": 0.5}, {"key": "p", "start": 0.75}],
        time_shift=0.0,
        window_radius=1,
        gaussian_sigma=1.0,
    )
    left, right, activity = builder.build(
        window_start=0.0,
        window_end=1.0,
        output_length=11,
        valid_start=0.0,
        valid_end=1.0,
    )
    assert left[5].item() == LEFT_POSITION_CHARSET.vocab["LP_UP"]
    assert right[7].item() == RIGHT_POSITION_CHARSET.vocab["RP_UP"]
    assert activity.max().item() == 1.0


def test_side_position_loss_ignores_minus_one_and_padding() -> None:
    outputs = FrontendOutput(
        log_probs=torch.empty(0),
        output_lengths=torch.tensor([2]),
        left_position_logits=torch.zeros(3, 1, 16),
        right_position_logits=torch.zeros(3, 1, 17),
        activity_logits=torch.zeros(3, 1, 1),
    )
    left = torch.tensor([[0], [-1], [4]])
    right = torch.tensor([[-1], [1], [5]])
    activity = torch.tensor([[1.0], [0.0], [1.0]])
    position_loss, activity_loss = side_position_losses(outputs, left, right, activity)
    expected_position = 0.5 * (
        torch.log(torch.tensor(16.0)) + torch.log(torch.tensor(17.0))
    )
    torch.testing.assert_close(position_loss, expected_position)
    torch.testing.assert_close(activity_loss, torch.log(torch.tensor(2.0)))


def test_side_position_builder_rejects_ambiguous_event_arrays() -> None:
    with pytest.raises(ValueError, match="sorted"):
        SidePositionTargetBuilder(
            event_times=np.asarray([0.2, 0.1]),
            left_event_classes=np.asarray([0, 1]),
            right_event_classes=np.asarray([-1, -1]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        SidePositionTargetBuilder.from_keystrokes([], window_radius=-1)


def test_side_position_collate_contains_only_final_auxiliary_targets() -> None:
    samples = [
        (
            torch.randn(5, 2, 16),
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8]),
            torch.tensor([0.1, 1.0, 0.2]),
        ),
        (
            torch.randn(4, 2, 16),
            torch.tensor([3]),
            torch.tensor([9, 10]),
            torch.tensor([11, 12]),
            torch.tensor([1.0, 0.3]),
        ),
    ]
    batch = SidePositionWindowedEMGDataset.collate(samples)
    assert set(batch) == {
        "inputs",
        "targets",
        "input_lengths",
        "target_lengths",
        "left_position_labels",
        "right_position_labels",
        "position_activity_labels",
    }
    assert batch["left_position_labels"][-1, 1].item() == -1
    assert batch["right_position_labels"][-1, 1].item() == -1
    assert batch["position_activity_labels"][-1, 1].item() == 0.0


def test_full_session_chunk_reassembly_is_cpu_only_and_contiguous() -> None:
    class DownsampleByTwo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            featurizer = SimpleNamespace(get_total_downsampling=lambda: 2)
            self.model = SimpleNamespace(cnn_featurizer=featurizer)

        def forward(self, inputs, input_lengths):
            return inputs[::2], (input_lengths + 1) // 2

    module = DownsampleByTwo()
    inputs = torch.arange(18, dtype=torch.float32).reshape(18, 1, 1)
    emissions, lengths = forward_session_in_chunks(
        module,
        inputs,
        torch.tensor([18]),
        chunk_size=8,
        overlap_size=4,
    )
    torch.testing.assert_close(emissions, inputs[::2])
    assert lengths.tolist() == [9]
    assert emissions.device.type == "cpu"


@pytest.mark.parametrize(
    ("model_name", "module_name", "multiscale", "side_aux", "data_name"),
    (
        (
            "conformer_base",
            "ConformerCTCModule",
            False,
            False,
            "RawEMGDataModule",
        ),
        (
            "conformer_multiscale",
            "ConformerCTCModule",
            True,
            False,
            "RawEMGDataModule",
        ),
        (
            "split_share",
            "SplitShareConformerCTCModule",
            True,
            False,
            "RawEMGDataModule",
        ),
        (
            "split_share_position_aux",
            "SplitShareConformerCTCModule",
            True,
            True,
            "SidePositionDataModule",
        ),
    ),
)
def test_frontend_ablation_configs_compose(
    model_name: str,
    module_name: str,
    multiscale: bool,
    side_aux: bool,
    data_name: str,
) -> None:
    config_dir = Path(__file__).parents[2] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="frontend",
            overrides=[f"model={model_name}"],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(config)
    assert config.module._target_.endswith(module_name)
    assert bool(config.module.use_multiscale_frontend) is multiscale
    assert bool(config.module.get("use_side_position_aux", False)) is side_aux
    assert config.datamodule._target_.endswith(data_name)
    json.dumps(_json_config(config))


def test_both_frontend_families_expose_the_same_downsampling_contract() -> None:
    common: dict[str, Any] = {
        "hidden_dim": 16,
        "conv_channels": (8, 8, 8),
        "num_layers": 1,
        "num_heads": 4,
        "ffn_dim": 32,
        "multiscale_branch_channels": 2,
    }
    models = (
        ConformerCTC(input_channels=32, **common),
        SplitShareConformerCTC(input_channels_per_hand=16, **common),
    )
    assert {model.cnn_featurizer.get_total_downsampling() for model in models} == {20}
    assert all(
        parameter.device.type == "cpu"
        for model in models
        for parameter in model.parameters()
    )


def test_metric_values_are_json_scalars() -> None:
    values = _plain_metric_values(
        {
            "CER": torch.tensor(1.0),
            "IER": 2.0,
            "DER": torch.tensor(3),
            "SER": 4,
        }
    )
    assert values == {"CER": 1.0, "IER": 2.0, "DER": 3.0, "SER": 4.0}
    json.dumps(values)
