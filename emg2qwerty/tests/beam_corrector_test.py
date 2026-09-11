from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import random
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

from emg2qwerty.transcription.beam_corrector import (
    _cosine_lr_scale,
    _resolve_dtype,
    BeamAwareQwenModule,
)
from emg2qwerty.transcription.beam_dataset import (
    _QualityMixDataset,
    BeamCorrectionDataModule,
    BeamCorrectionDataset,
    build_beam_payload,
    resolve_topk_files,
)
from emg2qwerty.transcription.beam_protocol import (
    build_structural_bad_words_ids,
    compute_cer_wer,
    parse_model_output,
    reconstruct_sessions,
    shared_output_filter,
    STRUCTURAL_PROMPT_PHRASES,
)
from emg2qwerty.transcription.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
    load_user_template,
    prompt_sha256,
    render_user_prompt,
)
from emg2qwerty.transcription.text import (
    collate_text_batch,
    normalize_text,
    resolve_backspaces,
)


REPO = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_beams(
    path: Path,
    *,
    windows: list[int],
    split: str = "train",
    prefix: str = "standard",
    top_k: int = 2,
) -> Path:
    rows = []
    for window in windows:
        for rank in range(1, top_k + 1):
            rows.append(
                {
                    "split": split,
                    "session": "session-a",
                    "window_index": window,
                    "rank": rank,
                    "prediction": f"{prefix}-{window}-r{rank}",
                    "target": f"target-{window}",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_text_normalization_backspaces_and_collation() -> None:
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    # Match the development repository: only its two textual backspace tokens.
    assert resolve_backspaces("ab<backspace>c⌫d\be") == "ad\be"
    assert resolve_backspaces("a<NO_CONTEXT><backspace>b") == "ab"
    assert collate_text_batch([{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == {
        "a": [1, 3],
        "b": [2, 4],
    }
    with pytest.raises(ValueError, match="different keys"):
        collate_text_batch([{"a": 1}, {"b": 2}])


def test_prompt_is_frozen_and_brace_safe(tmp_path: Path) -> None:
    template_path = REPO / "config/templates/beam_correction.txt"
    assert load_user_template(template_path) == DEFAULT_USER_TEMPLATE
    payload = "literal {brace} and <NO_CONTEXT>"
    rendered = render_user_prompt(payload)
    assert "literal {{brace}} and <NO_CONTEXT>" in rendered
    assert payload in render_user_prompt(payload, development_format=False)
    assert len(prompt_sha256(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE)) == 64
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("no boundary", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_user_template(invalid)


def test_beam_payload_keeps_primary_separate() -> None:
    payload = build_beam_payload(
        [["rank1", "rank2"], ["", "line\nbreak"]],
        primary="rank1second",
    )
    assert "=== Window 1 ===" in payload
    assert "- <EMPTY>" in payload
    assert "line\\nbreak" in payload
    assert payload.endswith("rank1second")


def test_dataset_training_drops_tail_and_evaluation_keeps_it(tmp_path: Path) -> None:
    source = _write_beams(tmp_path / "sample_topk.csv", windows=list(range(5)))
    assert resolve_topk_files(source) == [source.resolve()]
    training = BeamCorrectionDataset(
        source, split="train", chunk_size=3, top_k=2, drop_incomplete=True
    )
    evaluation = BeamCorrectionDataset(
        source, split="train", chunk_size=3, top_k=2, drop_incomplete=False
    )
    assert [sample.window_indices for sample in training.samples] == [(0, 1, 2)]
    assert [sample.window_indices for sample in evaluation.samples] == [
        (0, 1, 2),
        (3, 4),
    ]
    assert evaluation[0]["current_prediction"] == "".join(
        f"standard-{window}-r1" for window in range(3)
    )


def test_dataset_sorts_window_indices_numerically(tmp_path: Path) -> None:
    source = _write_beams(tmp_path / "unordered_topk.csv", windows=[10, 2])
    dataset = BeamCorrectionDataset(
        source,
        split="train",
        chunk_size=2,
        top_k=2,
        require_consecutive_windows=False,
        drop_incomplete=False,
    )
    assert [sample.window_indices for sample in dataset.samples] == [(2, 10)]


@pytest.mark.parametrize("invalid_index", ["not-a-number", "0.5", "inf"])
def test_dataset_rejects_invalid_window_indices(
    tmp_path: Path, invalid_index: str
) -> None:
    source = _write_beams(tmp_path / "invalid_topk.csv", windows=[0])
    frame = pd.read_csv(source)
    frame["window_index"] = frame["window_index"].astype(object)
    frame.loc[0, "window_index"] = invalid_index
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="window_index"):
        BeamCorrectionDataset(source, split="train", chunk_size=1, top_k=2)


def test_topk_discovery_is_recursive_stable_and_deduplicated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "beams"
    nested = root / "nested"
    nested.mkdir(parents=True)
    top = _write_beams(root / "top_topk.csv", windows=[0])
    inner = _write_beams(nested / "nested_topk.csv", windows=[1])

    discovered = resolve_topk_files([root, nested, top])
    assert discovered == sorted(
        {top.resolve(), inner.resolve()},
        key=lambda path: path.as_posix(),
    )


def test_dataset_preserves_na_like_strings_and_true_empty_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "literal_topk.csv"
    source.write_text(
        "split,session,window_index,rank,prediction,target\n"
        "train,NA,0,1,NA,nan\n"
        "train,NA,0,2,nan,nan\n"
        "train,empty,0,1,,\n"
        "train,empty,0,2,,\n",
        encoding="utf-8",
    )
    dataset = BeamCorrectionDataset(
        source,
        split="train",
        chunk_size=1,
        top_k=2,
        filter_empty_targets=False,
    )
    samples = {sample.session: sample for sample in dataset.samples}
    literal = samples["NA"]
    assert literal.target == "nan"
    assert literal.candidates_per_window == (("NA", "nan"),)
    empty = samples["empty"]
    assert empty.target == ""
    assert empty.candidates_per_window == (("", ""),)


def test_dataset_splits_gaps_and_shuffles_without_changing_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_beams(tmp_path / "gap_topk.csv", windows=[0, 1, 4])
    dataset = BeamCorrectionDataset(
        source,
        split="train",
        chunk_size=3,
        top_k=2,
        drop_incomplete=False,
        require_consecutive_windows=True,
        shuffle_candidates=True,
    )
    assert [sample.window_indices for sample in dataset.samples] == [(0, 1), (4,)]
    primary = dataset.samples[0].primary
    monkeypatch.setattr(random, "shuffle", lambda values: values.reverse())
    record = dataset[0]
    assert record["current_prediction"] == primary
    candidates = cast(list[list[str]], record["candidates_per_window"])
    assert candidates[0][0].endswith("r2")
    assert str(record["payload"]).endswith(primary)


def test_dataset_rejects_rank_order_and_invalid_sizes(tmp_path: Path) -> None:
    source = _write_beams(tmp_path / "rank_topk.csv", windows=[0])
    frame = pd.read_csv(source).iloc[::-1]
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="rank invariant"):
        BeamCorrectionDataset(source, split="train", top_k=2)
    with pytest.raises(ValueError, match="chunk_size"):
        BeamCorrectionDataset(source, split="train", chunk_size=0)


def test_quality_mix_validates_ratios_and_partitions(tmp_path: Path) -> None:
    standard = BeamCorrectionDataset(
        _write_beams(tmp_path / "standard_topk.csv", windows=[0, 1, 2]),
        split="train",
        top_k=2,
    )
    with pytest.raises(ValueError, match="must not exceed"):
        _QualityMixDataset(standard, None, None, 0.6, 0.5)
    mismatched = BeamCorrectionDataset(
        _write_beams(tmp_path / "degraded_topk.csv", windows=[1, 2, 3]),
        split="train",
        top_k=2,
    )
    with pytest.raises(ValueError, match="does not match"):
        _QualityMixDataset(standard, mismatched, None, 0.3, 0.0)


def test_quality_mix_rejects_empty_and_target_mismatched_pools(
    tmp_path: Path,
) -> None:
    standard = BeamCorrectionDataset(
        _write_beams(tmp_path / "standard_topk.csv", windows=[0, 1, 2]),
        split="train",
        top_k=2,
    )
    empty = BeamCorrectionDataset(
        _write_beams(
            tmp_path / "empty_degraded_topk.csv",
            windows=[0, 1, 2],
            split="val",
        ),
        split="train",
        top_k=2,
    )
    with pytest.raises(ValueError, match="non-zero.*empty"):
        _QualityMixDataset(standard, empty, None, 0.3, 0.0)

    target_path = _write_beams(tmp_path / "target_mismatch_topk.csv", windows=[0, 1, 2])
    frame = pd.read_csv(target_path)
    frame["target"] = "different-target"
    frame.to_csv(target_path, index=False)
    target_mismatch = BeamCorrectionDataset(target_path, split="train", top_k=2)
    with pytest.raises(ValueError, match="targets do not match"):
        _QualityMixDataset(standard, target_mismatch, None, 0.3, 0.0)


def test_datamodule_rejects_hidden_or_invalid_options(tmp_path: Path) -> None:
    source = _write_beams(tmp_path / "standard_topk.csv", windows=[0, 1, 2])
    with pytest.raises(TypeError):
        BeamCorrectionDataModule(source, unknown_branch=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="sum to one"):
        BeamCorrectionDataModule(
            source,
            standard_ratio=1,
            degraded_ratio=0.2,
            severely_degraded_ratio=0,
        )
    with pytest.raises(ValueError, match="val_source"):
        BeamCorrectionDataModule(
            source,
            standard_ratio=1,
            degraded_ratio=0,
            severely_degraded_ratio=0,
            val_source="low_quality",
        )


class _EncodingTokenizer:
    additional_special_tokens = ["<NO_CONTEXT>"]
    all_special_tokens = ["<NO_CONTEXT>", "<|im_end|>"]
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    bos_token = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


def test_qwen_structural_generation_guards_have_all_contexts() -> None:
    tokenizer = _EncodingTokenizer()
    values = build_structural_bad_words_ids(tokenizer)
    sequences = {tuple(value) for value in values}
    for phrase in STRUCTURAL_PROMPT_PHRASES:
        for prefix in ("", " ", "\n"):
            assert tuple((prefix + phrase).encode("utf-8")) in sequences
    assert (
        "model_family"
        not in inspect.signature(build_structural_bad_words_ids).parameters
    )


def test_output_parser_strips_only_qwen_control_tokens() -> None:
    tokenizer = _EncodingTokenizer()
    raw = "<|im_start|>assistant\n<NO_CONTEXT>answer<|im_end|>"
    assert parse_model_output(raw, tokenizer=tokenizer) == "<NO_CONTEXT>answer"
    assert parse_model_output("assistant: literal", tokenizer=tokenizer) == (
        "assistant: literal"
    )
    assert parse_model_output("<think>unfinished", tokenizer=tokenizer) == ""


def test_output_filter_uses_lexical_drift_guard_and_rejects_leakage() -> None:
    primary = "hello worl"
    corrected, diagnostics = shared_output_filter(
        primary,
        [[primary, "hello world", "hello world"]],
        "hello world",
    )
    assert corrected == "hello world"
    assert diagnostics["accepted"] == "True"
    assert diagnostics["filter"] == "lexical_drift_guard_v1"
    assert diagnostics["lexical_check"] == "applied"
    fallback, diagnostics = shared_output_filter(
        primary, [[primary]], "unrelated hallucination"
    )
    assert fallback == primary
    assert diagnostics["reason"] == "lexical_drift"
    leaked, diagnostics = shared_output_filter(
        primary, [[primary]], "=== Window 1 ===\nhello"
    )
    assert leaked == primary
    assert diagnostics["reason"] == "prompt_leakage"


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("12345", "12345"),
        ("!?—", "!?—"),
        ("⏎", "⏎"),
        ("<NO_CONTEXT>", "<NO_CONTEXT>"),
        ("1\r2", "1\n2"),
    ],
)
def test_output_filter_marks_nonlexical_content_not_applicable(
    candidate: str, expected: str
) -> None:
    corrected, diagnostics = shared_output_filter("source text", [], candidate)
    assert corrected == expected
    assert diagnostics["accepted"] == "True"
    assert diagnostics["reason"] == "lexical_check_not_applicable"
    assert diagnostics["lexical_check"] == "not_applicable"
    assert diagnostics["lexical_overlap_rate"] == "nan"


def test_session_metrics_preserve_provenance() -> None:
    chunks = pd.DataFrame(
        [
            {
                "session": "s1",
                "split": "test",
                "window_index": "10",
                "prediction": "world",
                "target": "world",
            },
            {
                "session": "s1",
                "split": "test",
                "window_index": "2.0",
                "prediction": "helo ",
                "target": "hello ",
            },
        ]
    )
    sessions = reconstruct_sessions(
        chunks, prediction_col="prediction", target_col="target"
    )
    assert sessions.loc[0, "prediction"] == "helo world"
    assert sessions.loc[0, "chunk_start_indices"] == "2,10"
    metrics, summary = compute_cer_wer(sessions, "prediction", "target")
    assert metrics.loc[0, "session"] == "s1"
    assert metrics.loc[0, "split"] == "test"
    assert summary["samples"] == 1
    duplicated = pd.concat([chunks.iloc[[0]], chunks.iloc[[0]]], ignore_index=True)
    duplicated.loc[1, "window_index"] = 10.0
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_sessions(
            duplicated, prediction_col="prediction", target_col="target"
        )


@pytest.mark.parametrize(
    "window_index",
    [
        "not-a-number",
        1.5,
        float("nan"),
        float("inf"),
        True,
        2**63,
    ],
)
def test_session_reconstruction_rejects_invalid_window_indices(
    window_index: object,
) -> None:
    table = pd.DataFrame(
        [
            {
                "session": "s",
                "window_index": window_index,
                "prediction": "x",
                "target": "x",
            }
        ]
    )
    with pytest.raises(ValueError, match="window_index"):
        reconstruct_sessions(
            table,
            prediction_col="prediction",
            target_col="target",
        )


class _TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    eos_token = "<|im_end|>"
    pad_token = "<|endoftext|>"
    bos_token = None
    padding_side = "right"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [10 + index for index, _ in enumerate(text)]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs["tokenize"] is False
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        assert kwargs["return_tensors"] == "pt"
        width = max(len(value) for value in prompts)
        rows = []
        masks = []
        for prompt in prompts:
            ids = [1] * len(prompt)
            rows.append([0] * (width - len(ids)) + ids)
            masks.append([0] * (width - len(ids)) + [1] * len(ids))
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }

    def decode(self, row: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "decoded" if len(row) else ""


class _TinyModel(torch.nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.options: dict[str, Any] = {}

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.options = kwargs
        if self.fail:
            raise RuntimeError("generation failed")
        continuation = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long)
        return torch.cat([input_ids, continuation], dim=1)


def _bare_module() -> BeamAwareQwenModule:
    module = BeamAwareQwenModule.__new__(BeamAwareQwenModule)
    pl.LightningModule.__init__(module)
    module.tokenizer = _TinyTokenizer()
    module.model = _TinyModel()
    module.max_seq_length = 4096
    module.max_new_tokens = 8
    module.generation_kwargs = {}
    module.chat_template_kwargs = {"enable_thinking": False}
    module.system_prompt = DEFAULT_SYSTEM_PROMPT
    module.user_template = DEFAULT_USER_TEMPLATE
    module.prompt_digest = prompt_sha256(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE)
    module.model_id = "Qwen/Qwen3.5-9B"
    module.base_model_revision = "a" * 40
    module.resolved_model_revision = "a" * 40
    module.base_model_class = "Qwen3_5ForCausalLM"
    module.dtype_name = "bf16"
    module.lora_r = 16
    module.lora_alpha = 32
    module.lora_dropout = 0.05
    module.lora_target_modules = ("q_proj",)
    return module


def test_response_only_labels_mask_prompt() -> None:
    module = _bare_module()
    batch = module.prepare_response_only_batch(["x"], ["ok"])
    labels = batch["labels"][0]
    assert labels[-1].item() == module.tokenizer.eos_token_id
    first_response = int((labels != -100).nonzero()[0])
    assert torch.all(labels[:first_response] == -100)
    assert torch.all(batch["attention_mask"] == 1)


def test_generation_is_deterministic_and_restores_train_mode() -> None:
    module = _bare_module()
    module.model.train()
    assert module.generate_text(["payload"]) == ["decoded"]
    assert module.model.training is True
    assert module.model.options["do_sample"] is False
    assert module.model.options["max_new_tokens"] == 8
    assert module.model.options["bad_words_ids"]
    module.model = _TinyModel(fail=True)
    module.model.train()
    with pytest.raises(RuntimeError, match="generation failed"):
        module.generate_text(["payload"])
    assert module.model.training is True


def test_dtype_and_cosine_schedule_are_pure_cpu_helpers() -> None:
    assert _resolve_dtype("fp32") is torch.float32
    with pytest.raises(ValueError, match="unsupported"):
        _resolve_dtype("fp8")
    assert _cosine_lr_scale(0, 10, 2) == pytest.approx(0.5)
    assert _cosine_lr_scale(1, 10, 2) == pytest.approx(1.0)
    assert _cosine_lr_scale(10, 10, 2) == pytest.approx(0.0)


def test_evaluator_topk_manifest_uses_logical_names(tmp_path: Path) -> None:
    evaluate = _load_script("beam_evaluate_paths", "scripts/evaluate_beam_corrector.py")
    root = tmp_path / "topk"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = _write_beams(nested / "user_topk.csv", windows=[0])
    entries = evaluate._topk_input_manifest(root, [source.resolve()])
    assert entries == [
        {
            "name": "nested/user_topk.csv",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    assert str(tmp_path) not in json.dumps(entries)
    assert (
        evaluate._topk_input_manifest(source, [source.resolve()])[0]["name"]
        == source.name
    )
    outside = _write_beams(tmp_path / "outside_topk.csv", windows=[0])
    with pytest.raises(ValueError, match="outside"):
        evaluate._topk_input_manifest(root, [outside.resolve()])


def test_evaluator_pure_batch_and_output_path(tmp_path: Path) -> None:
    evaluate = _load_script(
        "beam_evaluate_tables", "scripts/evaluate_beam_corrector.py"
    )

    class Corrector:
        def correct(
            self, payloads: Any, primaries: Any, candidates: Any
        ) -> list[dict[str, Any]]:
            return [
                {
                    "raw": "hello!",
                    "parsed": "hello!",
                    "corrected": "hello",
                    "diagnostics": {"accepted": "True"},
                }
                for _ in payloads
            ]

    batch = {
        "payload": ["p"],
        "current_prediction": ["helo"],
        "candidates_per_window": [[[["helo", "hello"]][0]]],
        "session": ["s1"],
        "split": ["test"],
        "window_index": [0],
        "window_indices": [[0]],
        "target": ["hello"],
    }
    chunks = evaluate._prediction_table(Corrector(), [batch], user="user-1")
    tables, summaries = evaluate._evaluation_tables(chunks)
    assert tables["window_llm_session_metrics.csv"].loc[0, "session"] == "s1"
    assert summaries["corrected"]["CER"] == 0.0
    assert summaries["unfiltered"]["CER"] == 20.0
    output = tmp_path / "results"
    hashes = evaluate._write_outputs(
        output,
        tables=tables,
        summaries=summaries,
        user="user-1",
    )
    assert hashes
    for relative, expected in hashes.items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        evaluate._write_outputs(
            output,
            tables=tables,
            summaries=summaries,
            user="user-1",
        )


def test_release_dependencies_and_qwen_loader_are_current() -> None:
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==2.13.0+cu132" in requirements
    assert "torchaudio==2.11.0.dev20260818+cu132" in requirements
    assert "pytorch-lightning==2.6.1" in requirements
    assert "transformers==5.5.4" in requirements
    assert "peft==0.19.1" in requirements
    assert "pytest==7.4.4" in requirements
    assert "pre-commit==3.7.1" in requirements
    source = (REPO / "emg2qwerty/transcription/beam_corrector.py").read_text(
        encoding="utf-8"
    )
    assert "Qwen3_5ForCausalLM.from_pretrained" in source
    protocol_source = (
        (REPO / "emg2qwerty/transcription/beam_protocol.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "flan" not in protocol_source
    assert "t5" not in protocol_source
