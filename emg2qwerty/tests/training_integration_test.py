"""Exercise the real frontend entry point, checkpoint initialization and resume."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from emg2qwerty.train import main
from emg2qwerty.transcription.checkpoint import load_lightning_weights


def test_frontend_training_initialization_and_resume(tmp_path: Path) -> None:
    data = np.zeros(
        1600,
        dtype=[
            ("emg_left", "f4", (16,)),
            ("emg_right", "f4", (16,)),
            ("time", "f8"),
        ],
    )
    rng = np.random.default_rng(42)
    for side in ("emg_left", "emg_right"):
        data[side] = rng.normal(size=(len(data), 16))
    data["time"] = np.arange(len(data)) / 2000
    for split in ("train", "val", "test"):
        with h5py.File(tmp_path / f"{split}.hdf5", "w") as handle:
            group = handle.create_group("emg2qwerty")
            group.create_dataset("timeseries", data=data)
            group.attrs.update(
                {
                    "session_name": split,
                    "user": "synthetic",
                    "condition": "on_keyboard",
                    "duration_mins": len(data) / 2000 / 60,
                    "prompts": "[]",
                    "keystrokes": json.dumps(
                        [
                            {"key": "a", "start": 0.1, "end": 0.15},
                            {"key": "p", "start": 0.5, "end": 0.55},
                        ]
                    ),
                }
            )
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="frontend",
            return_hydra_config=True,
            overrides=[
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.precision=32-true",
                "trainer.max_epochs=1",
                "batch_size=2",
                "num_workers=0",
                "module.hidden_dim=32",
                "module.conv_channels=[16,16,16]",
                "module.num_layers=1",
                "module.num_heads=4",
                "module.ffn_dim=64",
                "module.multiscale_branch_channels=4",
                "datamodule.window_length=800",
                "datamodule.padding=[100,100]",
            ],
        )
    with open_dict(config):
        config.dataset.root = str(tmp_path)
        for split in ("train", "val", "test"):
            config.dataset[split] = [{"session": split}]
        config.trainer.limit_train_batches = 1
        config.trainer.limit_val_batches = 1
        config.trainer.limit_test_batches = 1
        config.trainer.enable_progress_bar = False
        config.trainer.num_sanity_val_steps = 0
        config.trainer.enable_model_summary = False
        config.lr_scheduler.scheduler.max_epochs = 200

    def run(name: str, *, checkpoint=None, resume=None, epochs=1):
        OmegaConf.set_readonly(config.hydra, False)
        with open_dict(config):
            config.hydra.runtime.output_dir = str(tmp_path / name)
            config.checkpoint = checkpoint
            config.resume_from_checkpoint = resume
            config.trainer.max_epochs = epochs
        HydraConfig.instance().set_config(config)
        main.__wrapped__(config)
        return max(
            (tmp_path / name / "checkpoints").glob("last*.ckpt"),
            key=lambda path: path.stat().st_mtime_ns,
        )

    previous_config = HydraConfig.instance().cfg
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        original = run("original")
        saved = torch.load(original, map_location="cpu", weights_only=True)
        assert saved["global_step"] == 1
        assert isinstance(saved["hyper_parameters"]["conv_channels"], list)
        resumed = run("resumed", resume=str(original), epochs=2)
        restored_state = torch.load(resumed, map_location="cpu", weights_only=True)
        optimizer_states = restored_state["optimizer_states"][0]["state"].values()
        assert all(int(state["step"]) == 2 for state in optimizer_states)
        assert (
            torch.load(resumed, map_location="cpu", weights_only=True)["global_step"]
            == 2
        )
        automatically_resumed = run("resumed", epochs=3)
        assert (
            torch.load(automatically_resumed, map_location="cpu", weights_only=True)[
                "global_step"
            ]
            == 3
        )
        # Explicit initialization must not auto-resume an existing longer run.
        initialized = run("resumed", checkpoint=str(original))
        assert (
            torch.load(initialized, map_location="cpu", weights_only=True)[
                "global_step"
            ]
            == 1
        )
    finally:
        HydraConfig.instance().cfg = previous_config
        torch.set_num_threads(previous_threads)


def test_legacy_omegaconf_checkpoint_loads_strict_weights(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "hyper_parameters": OmegaConf.create({"channels": [1, 2]}),
        },
        checkpoint,
    )
    restored = torch.nn.Linear(3, 2)
    load_lightning_weights(restored, checkpoint)
    torch.testing.assert_close(restored.weight, source.weight)


def test_tail_windows_match_development_switch(tmp_path: Path) -> None:
    from emg2qwerty.data import LabelData, WindowedEMGDataset
    from emg2qwerty.transcription.data import SidePositionWindowedEMGDataset

    data = np.zeros(
        10000,
        dtype=[
            ("emg_left", "f4", (16,)),
            ("emg_right", "f4", (16,)),
            ("time", "f8"),
        ],
    )
    data["time"] = np.arange(len(data)) / 2000
    for size in (4000, 10000):
        path = tmp_path / f"session-{size}.hdf5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("emg2qwerty")
            group.create_dataset("timeseries", data=data[:size])
            group.attrs.update(
                {
                    "session_name": "synthetic",
                    "user": "synthetic",
                    "condition": "on_keyboard",
                    "duration_mins": size / 2000 / 60,
                    "prompts": "[]",
                    "keystrokes": json.dumps(
                        [
                            {"key": "a", "start": 1.0, "end": 1.05},
                            {"key": "p", "start": 4.5, "end": 4.55},
                        ]
                    ),
                }
            )
        for dataset_type in (WindowedEMGDataset, SidePositionWindowedEMGDataset):
            default = dataset_type(path, window_length=8000)
            inclusive = dataset_type(path, window_length=8000, include_tail=True)
            assert len(default) == 1
            assert len(inclusive) == (1 if size == 4000 else 2)
            if size == 4000:
                import pytest

                with pytest.raises(IndexError):
                    default[0]
            sample = inclusive[len(inclusive) - 1]
            assert len(sample[0]) == (4000 if size == 4000 else 2000)
            assert LabelData.from_labels(sample[1].tolist()).text == (
                "a" if size == 4000 else "p"
            )
