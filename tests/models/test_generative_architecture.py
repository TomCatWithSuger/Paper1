"""非语音模型的分层配置和 Lightning 生命周期回归。"""

from pathlib import Path
from typing import cast

import pytest
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
CASES = ("ae", "vae", "gan", "ddpm", "flow_matching", "score_based", "dit", "unet", "mnist")
MODEL_CONFIGS = sorted((ROOT / "configs" / "model").glob("*.yaml"))


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda path: path.stem)
def test_all_model_configs_use_method_network_ownership(path):
    cfg = OmegaConf.load(path)
    assert "net" not in cfg
    assert "beta" not in cfg
    assert "method" in cfg
    assert "network" in cfg.method


def small_config(name):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml")
    network = cfg.method.network
    if name in ("ae", "vae"):
        network.update(hidden_dims=[16, 8], latent_dim=4)
    elif name == "gan":
        network.update(
            latent_dim=4, generator_hidden_dims=[8, 16], discriminator_hidden_dims=[16, 8]
        )
    elif name in ("ddpm", "flow_matching", "score_based"):
        network.hidden_dims = [16, 8]
        network["noise_embedding_dim" if name == "score_based" else "time_embedding_dim"] = 8
    elif name == "dit":
        network.update(hidden_dim=16, context_dim=8, num_layers=1, num_heads=2)
    elif name == "unet":
        network.update(channels=[4, 8], kernel_size=3)
    else:
        network.update(lin1_size=8, lin2_size=16, lin3_size=8)
    if name == "ddpm":
        cfg.method.timesteps = 4
    elif name in ("flow_matching", "dit"):
        cfg.method.integration_steps = 3
    elif name == "score_based":
        cfg.method.update(num_noise_levels=3, sampling_steps_per_level=1)
    return cfg


def batch(name):
    if name in ("dit", "unet"):
        waveforms = torch.randn(3, 17)
        mask = torch.ones(3, 17, dtype=torch.bool)
        mask[1, 14:] = False
        return {"waveforms": waveforms, "attention_mask": mask, "lengths": mask.sum(1)}
    return torch.randn(3, 1, 28, 28), torch.tensor([1, 3, 5])


@pytest.mark.parametrize("name", CASES)
def test_all_configs_lightning_lifecycle(name):
    module = instantiate(small_config(name))
    values = batch(name)
    loader = DataLoader(cast(Dataset, [values]), batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    key = "val/generator_loss" if name == "gan" else "val/loss"
    assert torch.isfinite(trainer.callback_metrics[key])
    trainer.test(module, dataloaders=loader)
    predict_loader = (
        DataLoader(
            cast(Dataset, [cast(tuple[torch.Tensor, torch.Tensor], values)[0]]), batch_size=None
        )
        if name == "mnist"
        else loader
    )
    predictions = cast(list[torch.Tensor], trainer.predict(module, dataloaders=predict_loader))
    assert torch.isfinite(predictions[0]).all()


def test_old_production_files_are_removed():
    models = ROOT / "src" / "models"
    assert not (models / "legacy").exists()
    assert sorted(path.name for path in models.glob("*.py")) == ["__init__.py"]
    assert sorted(path.name for path in (models / "components").glob("*.py")) == ["__init__.py"]
