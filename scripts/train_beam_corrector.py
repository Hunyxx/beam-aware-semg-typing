#!/usr/bin/env python3
"""Train the beam-aware Qwen corrector and save Lightning checkpoints."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _device_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("trainer.devices must not be a boolean")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("trainer.devices must be positive")
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("trainer.devices must not be empty")
        return len(value)
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(
        "batch-size validation requires trainer.devices to be an "
        "integer or explicit device list"
    )


def _effective_global_batch_size(config: DictConfig) -> int:
    devices = _device_count(config.trainer.devices)
    nodes = int(config.trainer.get("num_nodes", 1))
    accumulation = int(config.trainer.get("accumulate_grad_batches", 1))
    microbatch = int(config.datamodule.batch_size)
    if nodes <= 0 or accumulation <= 0 or microbatch <= 0:
        raise ValueError("batch-size factors must all be positive")
    return microbatch * devices * nodes * accumulation


@hydra.main(version_base=None, config_path="../config", config_name="beam_corrector")
def main(config: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(config, resolve=True))
    actual = _effective_global_batch_size(config)
    expected = int(config.expected_global_batch_size)
    if actual != expected:
        raise ValueError(
            f"effective global batch size is {actual}, expected {expected}"
        )
    pl.seed_everything(int(config.seed), workers=True)
    module = instantiate(config.module, _recursive_=False)
    datamodule = instantiate(config.datamodule, _recursive_=False)
    callbacks = [instantiate(callback) for callback in config.get("callbacks", [])]
    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    resume = config.get("resume_from_checkpoint")
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=None if resume is None else str(Path(resume).expanduser()),
    )
    callback = trainer.checkpoint_callback
    if callback is not None:
        log.info("Best checkpoint: %s", callback.best_model_path)
        log.info("Last checkpoint: %s", callback.last_model_path)


if __name__ == "__main__":
    main()
