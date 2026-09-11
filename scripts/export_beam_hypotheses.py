#!/usr/bin/env python3
"""Export ordered CTC top-k hypotheses from a transcription frontend checkpoint.

The default device is CPU. Pass --device explicitly when running a production
export on another device.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from emg2qwerty import transforms
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.transcription.artifacts import portable_value, sha256_file
from emg2qwerty.transcription.checkpoint import load_frontend_checkpoint
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


FIELDS = (
    "split",
    "session",
    "window_index",
    "rank",
    "prediction",
    "target",
    "score",
)


class _IndexedWindows(Dataset):
    def __init__(self, dataset: WindowedEMGDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        return self.dataset[index], index

    @staticmethod
    def collate(
        samples: Sequence[tuple[tuple[torch.Tensor, torch.Tensor], int]],
    ) -> dict[str, Any]:
        batch = WindowedEMGDataset.collate([sample for sample, _ in samples])
        batch["window_indices"] = [index for _, index in samples]
        return batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--config-name", default="frontend")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override; may be repeated.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-tail",
        action="store_true",
        help="Include a final shorter window, as in the development exporter.",
    )
    return parser.parse_args()


def _session_paths(config: DictConfig, split: str) -> list[Path]:
    root = Path(config.dataset.root)
    return [root / f"{entry['session']}.hdf5" for entry in config.dataset[split]]


def _build_transform(config: DictConfig, split: str) -> Any:
    # Match the development exporter: never augment inference inputs, even train.
    del split
    return transforms.Compose([instantiate(item) for item in config.transforms.val])


def _json_config(config: DictConfig) -> object:
    if "hydra" in config:
        keys = [key for key in config if key != "hydra"]
        config = OmegaConf.masked_copy(config, keys)
    value = OmegaConf.to_container(config, resolve=True)
    json.dumps(value)
    return value


def decoder_lm_artifact(config: DictConfig) -> dict[str, str] | None:
    """Return a portable identity for the configured decoder language model."""

    raw_path = config.decoder.get("lm_path")
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"decoder language model not found: {path}")
    return {"file": path.name, "sha256": sha256_file(path)}


def output_csv_hashes(output_dir: Path, paths: Sequence[Path]) -> dict[str, str]:
    """Hash generated CSVs under ``output_dir`` using only relative names."""

    root = output_dir.expanduser().resolve()
    entries: dict[str, str] = {}
    for path in sorted((candidate.resolve() for candidate in paths), key=str):
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"generated CSV is outside output directory: {path}"
            ) from exc
        if name in entries:
            raise ValueError(f"duplicate generated CSV path: {name}")
        if path.suffix.lower() != ".csv" or not path.is_file():
            raise FileNotFoundError(f"generated CSV not found: {path}")
        entries[name] = sha256_file(path)
    return entries


def _decoder_candidates(decoder: Any, top_k: int) -> list[dict[str, object]]:
    beam = getattr(decoder, "beam", None)
    if beam is None or len(beam) < top_k:
        raise RuntimeError(
            f"decoder produced {0 if beam is None else len(beam)} states; "
            f"{top_k} are required"
        )
    candidates: list[dict[str, object]] = []
    decoder_charset = getattr(decoder, "_charset", None)
    for rank, state in enumerate(beam[:top_k], start=1):
        labels = list(state.decoding)
        timestamps = list(state.timestamps)
        prediction = LabelData.from_labels(
            labels,
            timestamps=timestamps,
            _charset=decoder_charset,
        ).text
        candidates.append(
            {
                "rank": rank,
                "prediction": prediction,
                "score": float(state.p_total),
            }
        )
    return candidates


def _load_module(config: DictConfig, weights: Path, device: torch.device) -> Any:
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    load_frontend_checkpoint(module, weights)
    return module.eval().to(device)


def _export_session(
    *,
    module: Any,
    session_path: Path,
    split: str,
    transform: Any,
    config: DictConfig,
    output_dir: Path,
    top_k: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    include_tail: bool = False,
) -> tuple[int, Path]:
    if not session_path.is_file():
        raise FileNotFoundError(f"session file not found: {session_path}")
    dataset = _IndexedWindows(
        WindowedEMGDataset(
            session_path,
            window_length=int(config.datamodule.window_length),
            stride=int(config.datamodule.window_length),
            padding=tuple(config.datamodule.padding),
            jitter=False,
            transform=transform,
            include_tail=include_tail,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_IndexedWindows.collate,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    output_path = output_dir / f"{session_path.stem}_topk.csv"
    rows_written = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["inputs"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                emissions, emission_lengths = module.forward(inputs, input_lengths)
                if emission_lengths is None:
                    raise RuntimeError(
                        "transcription frontend must return output lengths"
                    )
                targets = batch["targets"].cpu().numpy()
                target_lengths = batch["target_lengths"].cpu().numpy()
                emissions = emissions.detach().float().cpu().numpy()
                emission_lengths = emission_lengths.cpu().numpy()

                for batch_index, window_index in enumerate(batch["window_indices"]):
                    valid = int(emission_lengths[batch_index])
                    module.decoder.reset()
                    module.decoder.decode(
                        emissions=emissions[:valid, batch_index],
                        timestamps=np.arange(valid),
                        finish=True,
                    )
                    candidates = _decoder_candidates(module.decoder, top_k)
                    target_length = int(target_lengths[batch_index])
                    target = LabelData.from_labels(
                        targets[:target_length, batch_index]
                    ).text
                    for candidate in candidates:
                        score = candidate["score"]
                        if not isinstance(score, (int, float)):
                            raise TypeError("decoder candidate score must be numeric")
                        writer.writerow(
                            {
                                "split": split,
                                "session": session_path.stem,
                                "window_index": int(window_index),
                                "rank": candidate["rank"],
                                "prediction": candidate["prediction"],
                                "target": target,
                                "score": f"{score:.8f}",
                            }
                        )
                        rows_written += 1
    return rows_written, output_path


def main() -> None:
    args = _parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        raise ValueError("top-k and batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    config_dir = args.config_dir.expanduser().resolve()
    overrides = ["decoder=ctc_beam", *args.override]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=args.config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(config)
    if args.top_k > int(config.decoder.beam_size):
        raise ValueError("top-k cannot exceed decoder beam_size")
    decoder_lm = decoder_lm_artifact(config)

    weights = args.weights.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = portable_value(
        _json_config(config),
        roots={
            "${EMG2QWERTY_DATA_ROOT}": Path(str(config.dataset.root)),
            "${REPOSITORY_ROOT}": config_dir.parent,
        },
    )
    if not isinstance(resolved_config, dict):
        raise TypeError("resolved frontend configuration must be a mapping")
    device = torch.device(args.device)
    module = _load_module(config, weights, device)

    row_counts: dict[str, int] = {}
    session_files: dict[str, list[str]] = {}
    generated_csvs: list[Path] = []
    for split in args.splits:
        transform = _build_transform(config, split)
        sessions = _session_paths(config, split)
        session_files[split] = [path.name for path in sessions]
        row_counts[split] = 0
        for session_path in sessions:
            rows_written, output_path = _export_session(
                module=module,
                session_path=session_path,
                split=split,
                transform=transform,
                config=config,
                output_dir=output_dir,
                top_k=args.top_k,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                include_tail=args.include_tail,
            )
            row_counts[split] += rows_written
            generated_csvs.append(output_path)

    manifest = {
        "format_version": 1,
        "frontend_checkpoint": {"file": weights.name, "sha256": sha256_file(weights)},
        "decoder_language_model": decoder_lm,
        "top_k": args.top_k,
        "export_transform": "validation",
        "augmentation": False,
        "include_tail": args.include_tail,
        "device": str(device),
        "splits": list(args.splits),
        "row_counts": row_counts,
        "session_files": session_files,
        "outputs_sha256": output_csv_hashes(output_dir, generated_csvs),
        "resolved_config": resolved_config,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row_counts, indent=2))


if __name__ == "__main__":
    main()
