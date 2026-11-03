"""模块说明：非语音模型的分层配置和 Lightning 生命周期回归。"""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
from hydra.utils import instantiate
from lightning import Callback, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "ae",
    "vae",
    "gan",
    "ddpm",
    "flow_matching",
    "score_based",
    "dit",
    "unet",
    "mnist",
)
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
            latent_dim=4,
            generator_hidden_dims=[8, 16],
            discriminator_hidden_dims=[16, 8],
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
            cast(Dataset, [cast(tuple[torch.Tensor, torch.Tensor], values)[0]]),
            batch_size=None,
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


@pytest.mark.parametrize("name", ["gan", "flow_matching"])
def test_checkpoint_restores_training_state_before_next_update(name, tmp_path):
    cfg = small_config(name)
    if name == "flow_matching":
        cfg.scheduler = {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "_partial_": True,
            "step_size": 1,
            "gamma": 0.5,
        }
    loader = DataLoader(cast(Dataset, [batch(name)]), batch_size=None)

    def make_trainer(max_epochs, callbacks=None):
        return Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path,
        )

    original = instantiate(cfg)
    trainer = make_trainer(max_epochs=1)
    trainer.fit(original, train_dataloaders=loader, val_dataloaders=loader)
    checkpoint = tmp_path / f"{name}.ckpt"
    trainer.save_checkpoint(checkpoint)
    expected_model = deepcopy(original.method.state_dict())
    expected_optimizers = [deepcopy(optimizer.state_dict()) for optimizer in trainer.optimizers]
    expected_schedulers = [
        deepcopy(config.scheduler.state_dict()) for config in trainer.lr_scheduler_configs
    ]
    previous_step = trainer.global_step

    class VerifyRestoration(Callback):
        checked = False

        def on_train_start(self, trainer, pl_module):
            torch.testing.assert_close(
                pl_module.method.state_dict(), expected_model, rtol=0, atol=0
            )
            torch.testing.assert_close(
                [optimizer.state_dict() for optimizer in trainer.optimizers],
                expected_optimizers,
                rtol=0,
                atol=0,
            )
            assert [
                config.scheduler.state_dict() for config in trainer.lr_scheduler_configs
            ] == expected_schedulers
            assert trainer.global_step == previous_step
            self.checked = True

    verification = VerifyRestoration()
    restored = instantiate(cfg)
    resumed_trainer = make_trainer(max_epochs=2, callbacks=[verification])
    resumed_trainer.fit(
        restored, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=checkpoint
    )
    assert verification.checked
    assert resumed_trainer.global_step > previous_step
    assert any(
        not torch.equal(value, expected_model[key])
        for key, value in restored.method.state_dict().items()
    )
