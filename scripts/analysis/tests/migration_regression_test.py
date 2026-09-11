"""Regression cases uncovered while checking the migrated paper code."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.transcription.beam_corrector import validate_tokenizer_embeddings
from emg2qwerty.transcription.beam_dataset import BeamCorrectionDataset
from emg2qwerty.transcription.beam_protocol import compute_cer_wer
from emg2qwerty.transcription.text import read_transcription_csv
from scripts.analysis.analyze_rare_unseen_words import training_counts
from scripts.analysis.compute_topk_oracle_cer import (
    compute_session_oracle,
    load_candidates,
    select_window_oracle,
)


def write_candidates(path: Path, targets: list[str], split: str = "train") -> Path:
    pd.DataFrame(
        [
            {
                "split": split,
                "session": "001",
                "window_index": index,
                "rank": rank,
                "prediction": target,
                "target": target,
            }
            for index, target in enumerate(targets)
            for rank in (1, 2)
        ]
    ).to_csv(path, index=False)
    return path


def test_padded_vocabulary_is_valid_but_out_of_bounds_token_is_not() -> None:
    model = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(8, 4))
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "b": 1, "eos": 4},
        eos_token_id=4,
        pad_token_id=4,
    )
    validate_tokenizer_embeddings(tokenizer, model)
    tokenizer.get_vocab = lambda: {"a": 0, "extra": 8}
    with pytest.raises(ValueError, match="exceed"):
        validate_tokenizer_embeddings(tokenizer, model)


def test_csv_literals_survive_dataset_and_oracle_loading(tmp_path: Path) -> None:
    path = write_candidates(tmp_path / "literal_topk.csv", ["NA", "N/A", "NULL", "001"])
    assert read_transcription_csv(path)["target"].tolist()[::2] == [
        "NA",
        "N/A",
        "NULL",
        "001",
    ]
    oracle = load_candidates([path], top_k=2)
    dataset = BeamCorrectionDataset(path, split="train", top_k=2, chunk_size=1)
    assert [sample.target for sample in dataset.samples] == oracle["target"].tolist()[
        ::2
    ]
    assert {sample.session for sample in dataset.samples} == {"001"}


def test_oracle_selects_character_and_word_paths_independently(tmp_path: Path) -> None:
    path = write_candidates(tmp_path / "oracle_topk.csv", ["a b cd"], "test")
    frame = read_transcription_csv(path)
    frame["prediction"] = ["ab cd", "a b xx"]
    frame.to_csv(path, index=False)
    windows = select_window_oracle(load_candidates([path], top_k=2))
    assert windows["oracle_rank"].tolist() == [1]
    assert windows["word_oracle_rank"].tolist() == [2]
    _, _, summary = compute_session_oracle(windows)
    assert summary["oracle_topk_CER"] == pytest.approx(100 / 6)
    assert summary["oracle_topk_WER"] == pytest.approx(100 / 3)


def test_oracle_matches_empty_filter_and_rejects_incomplete_candidates(
    tmp_path: Path,
) -> None:
    path = write_candidates(tmp_path / "filter_topk.csv", ["", "NA"])
    assert set(load_candidates([path], top_k=2)["window_index"]) == {1}
    assert len(load_candidates([path], top_k=2, filter_empty_targets=False)) == 4
    with pytest.raises(ValueError, match="incomplete"):
        load_candidates([path], top_k=3)


def test_training_frequency_uses_complete_consecutive_training_chunks(
    tmp_path: Path,
) -> None:
    write_candidates(
        tmp_path / "train_topk.csv",
        ["hel", "lo ", "world", "discard ", "", "also ", "discard"],
    )
    write_candidates(tmp_path / "val_topk.csv", ["validation"] * 3, "val")
    assert training_counts(tmp_path, chunk_size=3, top_k=2) == {"hello": 1, "world": 1}


@pytest.mark.parametrize(
    ("prediction", "target", "insertion", "deletion"),
    [
        ("a x", "a", 2, 0),
        ("a", "a x", 0, 2),
    ],
)
def test_error_components_describe_recognition_errors(
    prediction: str, target: str, insertion: int, deletion: int
) -> None:
    table = pd.DataFrame([{"prediction": prediction, "target": target}])
    metrics, _ = compute_cer_wer(table, "prediction", "target")
    assert metrics.loc[0, "insertions"] == insertion
    assert metrics.loc[0, "deletions"] == deletion
    assert metrics.loc[0, "word_insertions"] == bool(insertion)
    assert metrics.loc[0, "word_deletions"] == bool(deletion)

    frontend_metric = CharacterErrorRates()
    frontend_metric.update(LabelData.from_str(prediction), LabelData.from_str(target))
    assert int(frontend_metric.insertions) == insertion
    assert int(frontend_metric.deletions) == deletion
