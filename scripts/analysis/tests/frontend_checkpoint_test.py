"""Direct checkpoint loading and frontend evaluation checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from emg2qwerty.transcription.checkpoint import (
    FRONTEND_VARIANTS,
    load_frontend_checkpoint,
)
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scripts.analysis.benchmark_frontend_latency import load_config, parse_args
from scripts.evaluate_frontend_sessions import (
    decoder_lm_artifact as evaluation_decoder_lm_artifact,
)
from scripts.export_beam_hypotheses import (
    decoder_lm_artifact as export_decoder_lm_artifact,
    output_csv_hashes,
)


ANALYSIS_DIR = Path(__file__).resolve().parents[1]


def test_latency_uses_configured_language_model_unless_explicitly_disabled() -> None:
    previous = HydraConfig.instance().cfg
    try:
        args = parse_args(["--weights", "frontend.ckpt"])
        assert (
            Path(load_config(args).decoder.lm_path).name
            == "wikitext-103-6gram-charlm.bin"
        )
        args = parse_args(["--weights", "frontend.ckpt", "--no-lm"])
        assert load_config(args).decoder.lm_path is None
    finally:
        HydraConfig.instance().cfg = previous


@pytest.mark.parametrize("variant", FRONTEND_VARIANTS)
def test_all_paper_frontends_load_ckpt(variant: str, tmp_path: Path) -> None:
    config_dir = ANALYSIS_DIR.parents[1] / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="frontend", overrides=[f"model={variant}"])
    module = instantiate(config.module)
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    source = tmp_path / "frontend.ckpt"
    torch.save(checkpoint, source)
    restored = instantiate(config.module)
    load_frontend_checkpoint(restored, source)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    changed = dict(checkpoint)
    changed["state_dict"] = dict(checkpoint["state_dict"])
    changed["state_dict"].pop(next(iter(changed["state_dict"])))
    torch.save(changed, source)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_frontend_checkpoint(restored, source)
    changed = dict(checkpoint)
    changed["hyper_parameters"] = {
        **checkpoint["hyper_parameters"],
        "dropout_rate": 0.9,
    }
    torch.save(changed, source)
    with pytest.raises(ValueError, match="dropout_rate"):
        load_frontend_checkpoint(restored, source)


def test_decoder_lm_artifacts_and_export_hashes_are_portable(tmp_path: Path) -> None:
    language_model = tmp_path / "language-model.bin"
    language_model.write_bytes(b"language-model")
    config = OmegaConf.create({"decoder": {"lm_path": str(language_model)}})
    expected_lm = {
        "file": language_model.name,
        "sha256": hashlib.sha256(language_model.read_bytes()).hexdigest(),
    }
    assert evaluation_decoder_lm_artifact(config) == expected_lm
    assert export_decoder_lm_artifact(config) == expected_lm
    assert str(tmp_path) not in json.dumps(expected_lm)

    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    first = output_dir / "first_topk.csv"
    nested = output_dir / "nested" / "second_topk.csv"
    nested.parent.mkdir()
    first.write_text("value\nfirst\n", encoding="utf-8")
    nested.write_text("value\nsecond\n", encoding="utf-8")
    hashes = output_csv_hashes(output_dir, [nested, first])
    assert hashes == {
        "first_topk.csv": hashlib.sha256(first.read_bytes()).hexdigest(),
        "nested/second_topk.csv": hashlib.sha256(nested.read_bytes()).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(hashes)


def test_decoder_lm_artifact_rejects_missing_file_and_accepts_null(
    tmp_path: Path,
) -> None:
    missing = OmegaConf.create({"decoder": {"lm_path": str(tmp_path / "missing")}})
    for helper in (evaluation_decoder_lm_artifact, export_decoder_lm_artifact):
        with pytest.raises(FileNotFoundError, match="decoder language model"):
            helper(missing)
        assert helper(OmegaConf.create({"decoder": {"lm_path": None}})) is None
