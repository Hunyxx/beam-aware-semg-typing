"""Top-k beam dataset used by the hypothesis-aware corrector."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from emg2qwerty.transcription.text import (
    collate_text_batch,
    normalize_text,
    read_transcription_csv,
    resolve_backspaces,
)


@dataclass(frozen=True)
class WindowCandidates:
    session: str
    window_index: int
    target: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class BeamChunk:
    session: str
    window_indices: tuple[int, ...]
    candidates_per_window: tuple[tuple[str, ...], ...]
    target: str
    split: str
    primary: str


def resolve_topk_files(
    paths: str | Path | Sequence[str | Path],
) -> list[Path]:
    sources = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: list[Path] = []
    for raw_path in sources:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            resolved.extend(path.rglob("*_topk.csv"))
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(path)
    unique = sorted(
        {path.resolve() for path in resolved},
        key=lambda path: path.as_posix(),
    )
    if not unique:
        raise FileNotFoundError("no *_topk.csv files found")
    return unique


def build_beam_payload(
    candidates_per_window: Sequence[Sequence[str]],
    *,
    primary: str,
) -> str:
    lines: list[str] = []
    for window_number, candidates in enumerate(candidates_per_window, start=1):
        lines.append(f"=== Window {window_number} ===")
        for candidate in candidates:
            cleaned = candidate.replace("\n", "\\n") if candidate else "<EMPTY>"
            lines.append(f"- {cleaned}")
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    primary_text = primary.replace("\n", "\\n") if primary else "<EMPTY>"
    lines.extend(["", "PRIMARY (rank-1 baseline, your starting point):", primary_text])
    return "\n".join(lines)


class BeamCorrectionDataset(Dataset):
    """Group top-k rows into fixed-size, optionally consecutive chunks."""

    def __init__(
        self,
        topk_paths: str | Path | Sequence[str | Path],
        *,
        split: str,
        chunk_size: int = 3,
        top_k: int = 10,
        filter_empty_targets: bool = True,
        require_consecutive_windows: bool = True,
        drop_incomplete: bool = True,
        shuffle_candidates: bool = False,
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        if not str(split):
            raise ValueError("split must be non-empty")
        self.split = str(split)
        self.chunk_size = int(chunk_size)
        self.top_k = int(top_k)
        self.filter_empty_targets = bool(filter_empty_targets)
        self.require_consecutive_windows = bool(require_consecutive_windows)
        self.drop_incomplete = bool(drop_incomplete)
        self.shuffle_candidates = bool(shuffle_candidates)
        windows = self._load_windows(resolve_topk_files(topk_paths))
        self.samples = self._build_chunks(windows)

    def _load_windows(self, paths: Sequence[Path]) -> list[WindowCandidates]:
        required = {"split", "session", "window_index", "rank", "prediction", "target"}
        frames: list[pd.DataFrame] = []
        for path in paths:
            frame = read_transcription_csv(path)
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            frame = frame[frame["split"].astype(str) == self.split]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return []
        frame = pd.concat(frames, ignore_index=True)
        frame["_session"] = frame["session"].fillna("").astype(str)
        numeric_windows = pd.to_numeric(frame["window_index"], errors="coerce")
        numeric_ranks = pd.to_numeric(frame["rank"], errors="coerce")
        if numeric_windows.isna().any() or numeric_ranks.isna().any():
            raise ValueError("window_index and rank must be numeric")
        if (numeric_windows % 1 != 0).any() or (numeric_ranks % 1 != 0).any():
            raise ValueError("window_index and rank must be integer-valued")
        frame["_window"] = numeric_windows.astype(int)
        frame["_rank"] = numeric_ranks.astype(int)

        windows: list[WindowCandidates] = []
        for (session, window_index), group in frame.groupby(
            ["_session", "_window"], sort=False
        ):
            relevant = group[group["_rank"].between(1, self.top_k)]
            observed_ranks = relevant["_rank"].tolist()
            expected_ranks = list(range(1, self.top_k + 1))
            if observed_ranks != expected_ranks:
                raise ValueError(
                    "top-k rank invariant failed for "
                    f"session={session!r}, window_index={window_index}: expected "
                    f"{expected_ranks}, observed {observed_ranks}"
                )
            targets = {
                resolve_backspaces(normalize_text(value))
                for value in relevant["target"]
            }
            if len(targets) != 1:
                raise ValueError(
                    "top-k candidates have inconsistent targets for "
                    f"session={session!r}, window_index={window_index}"
                )
            target = targets.pop()
            if self.filter_empty_targets and not target.strip():
                continue
            candidates = tuple(
                resolve_backspaces(normalize_text(value))
                for value in relevant["prediction"].tolist()
            )
            if len(candidates) != self.top_k:
                raise AssertionError(
                    "rank validation did not yield exactly top_k candidates"
                )
            windows.append(
                WindowCandidates(
                    session=str(session),
                    window_index=int(window_index),
                    target=target,
                    candidates=candidates,
                )
            )
        return sorted(windows, key=lambda item: (item.session, item.window_index))

    def _build_chunks(self, windows: Sequence[WindowCandidates]) -> list[BeamChunk]:
        by_session: dict[str, list[WindowCandidates]] = {}
        for window in windows:
            by_session.setdefault(window.session, []).append(window)

        chunks: list[BeamChunk] = []
        for session, session_windows in by_session.items():
            runs: list[list[WindowCandidates]] = [[]]
            for window in session_windows:
                previous = runs[-1][-1] if runs[-1] else None
                if (
                    self.require_consecutive_windows
                    and previous is not None
                    and window.window_index != previous.window_index + 1
                ):
                    runs.append([])
                runs[-1].append(window)
            for run in runs:
                for start in range(0, len(run), self.chunk_size):
                    selected = run[start : start + self.chunk_size]
                    if self.drop_incomplete and len(selected) < self.chunk_size:
                        continue
                    target = "".join(window.target for window in selected)
                    if self.filter_empty_targets and not target.strip():
                        continue
                    chunks.append(
                        BeamChunk(
                            session=session,
                            window_indices=tuple(w.window_index for w in selected),
                            candidates_per_window=tuple(w.candidates for w in selected),
                            target=target,
                            split=self.split,
                            primary="".join(w.candidates[0] for w in selected),
                        )
                    )
        self._validate_chunk_partition(windows, chunks)
        return chunks

    def _validate_chunk_partition(
        self,
        windows: Sequence[WindowCandidates],
        chunks: Sequence[BeamChunk],
    ) -> None:
        by_session: dict[str, list[WindowCandidates]] = {}
        for window in windows:
            by_session.setdefault(window.session, []).append(window)

        for session, session_windows in by_session.items():
            ordered = sorted(session_windows, key=lambda item: item.window_index)
            expected: list[int] = []
            if self.drop_incomplete:
                runs: list[list[int]] = [[]]
                for window in ordered:
                    previous = runs[-1][-1] if runs[-1] else None
                    if (
                        self.require_consecutive_windows
                        and previous is not None
                        and window.window_index != previous + 1
                    ):
                        runs.append([])
                    runs[-1].append(window.window_index)
                for run in runs:
                    keep = len(run) // self.chunk_size * self.chunk_size
                    expected.extend(run[:keep])
            else:
                expected = [window.window_index for window in ordered]

            actual = [
                window_index
                for chunk in chunks
                if chunk.session == session
                for window_index in chunk.window_indices
            ]
            if actual != expected or len(actual) != len(set(actual)):
                raise ValueError(
                    "beam chunk partition invariant failed for "
                    f"session={session!r}: expected={expected}, actual={actual}"
                )

        rank1 = {
            (window.session, window.window_index): window.candidates[0]
            for window in windows
        }
        for chunk in chunks:
            if len(chunk.candidates_per_window) != len(chunk.window_indices):
                raise ValueError("candidate/window count mismatch in beam chunk")
            expected_primary = "".join(
                rank1[(chunk.session, index)] for index in chunk.window_indices
            )
            if chunk.primary != expected_primary:
                raise ValueError(
                    "PRIMARY is not the ordered, pre-shuffle rank-1 concatenation"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        candidates = [list(window) for window in sample.candidates_per_window]
        if self.shuffle_candidates:
            for window in candidates:
                random.shuffle(window)
        return {
            "payload": build_beam_payload(candidates, primary=sample.primary),
            "target": sample.target,
            "session": sample.session,
            "window_index": sample.window_indices[0],
            "window_indices": list(sample.window_indices),
            "split": sample.split,
            "current_prediction": sample.primary,
            "candidates_per_window": candidates,
        }


class _QualityMixDataset(Dataset):
    def __init__(
        self,
        standard: BeamCorrectionDataset,
        degraded: BeamCorrectionDataset | None,
        severely_degraded: BeamCorrectionDataset | None,
        degraded_ratio: float,
        severely_degraded_ratio: float,
    ) -> None:
        degraded_ratio = float(degraded_ratio)
        severely_degraded_ratio = float(severely_degraded_ratio)
        if (
            not 0.0 <= degraded_ratio <= 1.0
            or not 0.0 <= severely_degraded_ratio <= 1.0
        ):
            raise ValueError("quality-mix ratios must be between zero and one")
        if degraded_ratio + severely_degraded_ratio > 1.0 + 1e-9:
            raise ValueError(
                "degraded_ratio + severely_degraded_ratio must not exceed one"
            )
        self.standard = standard
        self.degraded = degraded
        self.severely_degraded = severely_degraded
        self.degraded_ratio = degraded_ratio
        self.severely_degraded_ratio = severely_degraded_ratio
        self._degraded = self._index(degraded)
        self._severely_degraded = self._index(severely_degraded)
        standard_keys = {
            (sample.session, sample.window_indices) for sample in standard.samples
        }
        standard_targets = {
            (sample.session, sample.window_indices): sample.target
            for sample in standard.samples
        }
        for name, source, source_index, ratio in (
            ("degraded", degraded, self._degraded, degraded_ratio),
            (
                "severely_degraded",
                severely_degraded,
                self._severely_degraded,
                severely_degraded_ratio,
            ),
        ):
            if ratio > 0.0 and not source_index:
                raise ValueError(f"{name} ratio is non-zero but its beam pool is empty")
            if source_index and set(source_index) != standard_keys:
                missing = sorted(standard_keys - set(source_index))[:5]
                extra = sorted(set(source_index) - standard_keys)[:5]
                raise ValueError(
                    f"{name} beam pool does not match the standard partition; "
                    f"missing={missing}, extra={extra}"
                )
            if source_index:
                assert source is not None
                source_targets = {
                    (sample.session, sample.window_indices): sample.target
                    for sample in source.samples
                }
                mismatched_targets = sorted(
                    key
                    for key in standard_keys
                    if source_targets[key] != standard_targets[key]
                )
                if mismatched_targets:
                    raise ValueError(
                        f"{name} beam pool targets do not match the standard "
                        f"partition for keys={mismatched_targets[:5]}"
                    )

    @staticmethod
    def _index(
        dataset: BeamCorrectionDataset | None,
    ) -> dict[tuple[str, tuple[int, ...]], int]:
        if dataset is None:
            return {}
        return {
            (sample.session, sample.window_indices): index
            for index, sample in enumerate(dataset.samples)
        }

    def __len__(self) -> int:
        return len(self.standard)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.standard.samples[index]
        key = (sample.session, sample.window_indices)
        draw = random.random()
        if self.severely_degraded is not None and draw < self.severely_degraded_ratio:
            alternative = self._severely_degraded.get(key)
            if alternative is None:
                raise AssertionError("validated severely_degraded partition changed")
            return self.severely_degraded[alternative]
        if (
            self.degraded is not None
            and draw < self.severely_degraded_ratio + self.degraded_ratio
        ):
            alternative = self._degraded.get(key)
            if alternative is None:
                raise AssertionError("validated degraded partition changed")
            return self.degraded[alternative]
        return self.standard[index]


class BeamCorrectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        topk_root: str | Path,
        *,
        topk_root_degraded: str | Path | None = None,
        topk_root_severely_degraded: str | Path | None = None,
        standard_ratio: float = 0.2,
        degraded_ratio: float = 0.3,
        severely_degraded_ratio: float = 0.5,
        val_source: str = "severely_degraded",
        chunk_size: int = 3,
        top_k: int = 10,
        filter_empty_targets: bool = True,
        require_consecutive_windows: bool = True,
        shuffle_candidates_train: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        ratios = {
            "standard": float(standard_ratio),
            "degraded": float(degraded_ratio),
            "severely_degraded": float(severely_degraded_ratio),
        }
        if any(value < 0 or value > 1 for value in ratios.values()):
            raise ValueError("quality ratios must each be between zero and one")
        if abs(sum(ratios.values()) - 1.0) > 1e-9:
            raise ValueError(
                f"quality ratios must sum to one, got {sum(ratios.values())}"
            )
        if val_source not in {"standard", "degraded", "severely_degraded"}:
            raise ValueError(
                "val_source must be standard, degraded, or severely_degraded"
            )
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(num_workers) < 0:
            raise ValueError("num_workers must be non-negative")
        self.roots = {
            "standard": Path(topk_root),
            "degraded": Path(topk_root_degraded) if topk_root_degraded else None,
            "severely_degraded": (
                Path(topk_root_severely_degraded)
                if topk_root_severely_degraded
                else None
            ),
        }
        for name in ("degraded", "severely_degraded"):
            if ratios[name] > 0 and self.roots[name] is None:
                raise ValueError(
                    f"{name} ratio is non-zero but its top-k root is missing"
                )
        if self.roots[val_source] is None:
            raise ValueError(f"validation source {val_source!r} has no top-k root")
        self.standard_ratio = ratios["standard"]
        self.degraded_ratio = ratios["degraded"]
        self.severely_degraded_ratio = ratios["severely_degraded"]
        self.val_source = val_source
        self.dataset_options: dict[str, Any] = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "filter_empty_targets": filter_empty_targets,
            "require_consecutive_windows": require_consecutive_windows,
        }
        self.shuffle_candidates_train = bool(shuffle_candidates_train)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def _dataset(self, quality: str, split: str, *, training: bool = False):
        root = self.roots[quality]
        if root is None:
            return None
        return BeamCorrectionDataset(
            [root],
            split=split,
            shuffle_candidates=training and self.shuffle_candidates_train,
            drop_incomplete=training,
            **self.dataset_options,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            standard = self._dataset("standard", "train", training=True)
            assert standard is not None
            degraded = self._dataset("degraded", "train", training=True)
            severely_degraded = self._dataset(
                "severely_degraded", "train", training=True
            )
            self.train_dataset = _QualityMixDataset(
                standard,
                degraded,
                severely_degraded,
                self.degraded_ratio,
                self.severely_degraded_ratio,
            )
        if stage in (None, "fit", "validate"):
            self.val_dataset = self._dataset(self.val_source, "val")
        if stage in (None, "test"):
            self.test_dataset = self._dataset("standard", "test")

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_text_batch,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)


__all__ = [
    "BeamChunk",
    "BeamCorrectionDataModule",
    "BeamCorrectionDataset",
    "WindowCandidates",
    "build_beam_payload",
    "resolve_topk_files",
]
