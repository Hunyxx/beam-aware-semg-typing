import pytest
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from emg2qwerty import utils


def _scheduler_config():
    return OmegaConf.create(
        {
            "scheduler": {
                "_target_": "emg2qwerty.utils.LinearWarmupCosineAnnealingLR",
                "warmup_epochs": 5,
                "max_epochs": 200,
                "warmup_start_lr": 1e-8,
                "eta_min": 1e-6,
            },
            "interval": "epoch",
        }
    )


def test_native_warmup_cosine_scheduler_is_instantiated_from_config() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    result = utils.instantiate_optimizer_and_scheduler(
        [parameter],
        optimizer_config=OmegaConf.create(
            {
                "_target_": "torch.optim.AdamW",
                "lr": 3e-4,
                "weight_decay": 0.1,
            }
        ),
        lr_scheduler_config=_scheduler_config(),
    )

    scheduler = result["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, utils.LinearWarmupCosineAnnealingLR)
    assert isinstance(scheduler._schedulers[0], LinearLR)
    assert scheduler._schedulers[0].total_iters == 5
    assert isinstance(scheduler._schedulers[1], CosineAnnealingLR)
    assert scheduler._schedulers[1].T_max == 195
    assert result["lr_scheduler"]["interval"] == "epoch"


@pytest.mark.parametrize(
    ("warmup_epochs", "max_epochs"),
    [(0, 200), (5, 5), (6, 5)],
)
def test_native_scheduler_rejects_invalid_epoch_budgets(
    warmup_epochs: int,
    max_epochs: int,
) -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=3e-4)
    with pytest.raises(ValueError):
        utils.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
        )
