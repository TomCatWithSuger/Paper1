"""非语音四层迁移的已提交算法、优化器和生命周期回归。"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

from hydra.utils import instantiate
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.modules.generative import _map_legacy_state_dict
from tests.models.committed_baseline import committed_module


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "ae": ("dense_autoencoder", "DenseAutoencoder", "AELitModule"),
    "vae": ("dense_vae", "DenseVAE", "VAELitModule"),
    "gan": ("dense_gan", "DenseGAN", "GANLitModule"),
    "ddpm": ("dense_ddpm", "DenseDDPM", "DDPMLitModule"),
    "flow_matching": ("dense_flow_matching", "DenseFlowMatching", "FlowMatchingLitModule"),
    "score_based": ("dense_score_model", "DenseScoreModel", "ScoreBasedLitModule"),
    "dit": ("dit", "DiT1D", "DiTLitModule"),
    "unet": ("unet_1d", "UNet1D", "UNetLitModule"),
    "mnist": ("simple_dense_net", "SimpleDenseNet", "MNISTLitModule"),
}
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
        network.update(latent_dim=4, generator_hidden_dims=[8, 16], discriminator_hidden_dims=[16, 8])
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


def old_module(name, cfg):
    filename, network_name, module_name = CASES[name]
    network_type = getattr(committed_module("legacy." + filename), network_name)
    module_type = getattr(committed_module(name + "_module"), module_name)
    kwargs = cast(dict[str, Any], OmegaConf.to_container(cfg.method.network))
    kwargs.pop("_target_")
    method_kwargs = cast(dict[str, Any], OmegaConf.to_container(cfg.method))
    method_kwargs.pop("_target_")
    method_kwargs.pop("network")
    if name in ("ddpm", "flow_matching", "score_based", "dit"):
        kwargs.update(method_kwargs)
    net = network_type(**kwargs)
    if name == "gan":
        return module_type(net, instantiate(cfg.generator_optimizer), instantiate(cfg.discriminator_optimizer))
    module_kwargs = {"compile": False}
    if name == "vae":
        module_kwargs["beta"] = method_kwargs["beta"]
    return module_type(net, instantiate(cfg.optimizer), instantiate(cfg.scheduler), **module_kwargs)


def optimizers(module) -> list[torch.optim.Optimizer]:
    configured = module.configure_optimizers()
    return cast(list[torch.optim.Optimizer], configured if isinstance(configured, list) else [configured["optimizer"]])


def assert_outputs_equal(left, right):
    for expected, actual in zip(left, right):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("name", CASES)
def test_committed_weights_loss_gradients_optimizer_and_sampling(name):
    cfg = small_config(name)
    torch.manual_seed(21)
    old = old_module(name, cfg)
    torch.manual_seed(21)
    module = instantiate(cfg)
    assert isinstance(module, LightningModule)
    assert old.hparams["compile"] == module.hparams["compile"]
    if name == "vae":
        assert old.hparams["beta"] == module.method.beta
    assert isinstance(module.method, nn.Module)
    assert not isinstance(module.method, LightningModule)
    assert not hasattr(module, "net")
    assert isinstance(module.method.network, nn.Module)
    assert tuple(module._modules)[:1] == ("method",)
    for operation in ("sample", "generate", "interpolate", "perturb", "q_sample", "reparameterize"):
        assert not hasattr(module.method.network, operation)
    old_params = list(old.named_parameters())
    new_params = list(module.named_parameters())
    assert ["method.network." + key[len("net."):] for key, _ in old_params] == [
        key for key, _ in new_params
    ]
    assert len(list(module.named_parameters(remove_duplicate=False))) == len(new_params)
    for (_, expected), (_, actual) in zip(old_params, new_params):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    old_state = old.state_dict()
    mapped = _map_legacy_state_dict(module.method, old_state)
    assert list(mapped) == list(module.state_dict())
    assert dict(mapped._metadata) == dict(module.state_dict()._metadata)
    result = module.load_state_dict(old_state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    values = batch(name)
    for seed in (12, 123):
        torch.manual_seed(seed)
        expected = old.model_step(values)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(seed)
        actual = module.model_step(values)
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert_outputs_equal(expected, actual)
    old.zero_grad()
    module.zero_grad()
    (expected[0] + expected[1] if name == "gan" else expected[0]).backward()
    (actual[0] + actual[1] if name == "gan" else actual[0]).backward()
    for (_, expected_parameter), (_, actual_parameter) in zip(old_params, new_params):
        torch.testing.assert_close(expected_parameter.grad, actual_parameter.grad, rtol=0, atol=0)
    old_optimizers, new_optimizers = optimizers(old), optimizers(module)
    for expected_optimizer, actual_optimizer in zip(old_optimizers, new_optimizers):
        old_group_names = [[
            "method.network." + next(
                key for key, parameter in old_params if parameter is value
            )[len("net."):]
            for value in group["params"]
        ] for group in expected_optimizer.param_groups]
        new_group_names = [[next(key for key, p in new_params if p is value) for value in group["params"]]
                           for group in actual_optimizer.param_groups]
        assert old_group_names == new_group_names
        expected_optimizer.step()
        actual_optimizer.step()
        actual_optimizer.load_state_dict(expected_optimizer.state_dict())
    for (_, expected_parameter), (_, actual_parameter) in zip(old_params, new_params):
        torch.testing.assert_close(expected_parameter, actual_parameter, rtol=0, atol=0)
    module.eval()
    old.eval()
    if name in ("vae", "ddpm", "flow_matching", "score_based", "dit", "gan"):
        torch.manual_seed(51)
        expected_sample = old.net.generate(2, "cpu") if name == "gan" else old.net.sample(2 if name != "dit" else 17, "cpu")
        torch.manual_seed(51)
        if name == "gan":
            actual_sample = module.method.generate(2, "cpu")
        elif name == "dit":
            actual_sample = module.method.sample_waveform(17, "cpu")
        else:
            actual_sample = module.method.sample(2, "cpu")
        torch.testing.assert_close(expected_sample, actual_sample, rtol=0, atol=0)
    missing = OrderedDict(old.state_dict())
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict(missing, strict=True)
    extra = OrderedDict(old.state_dict())
    extra["unknown.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        module.load_state_dict(extra, strict=True)
    if name in ("ddpm", "score_based"):
        buffers = dict(module.method.named_buffers())
        key = next(iter(buffers))
        extra = OrderedDict(old.state_dict())
        extra["method." + key] = buffers[key]
        with pytest.raises(ValueError, match="冲突"):
            module.load_state_dict(extra, strict=True)


@pytest.mark.parametrize("name", CASES)
def test_all_configs_lightning_lifecycle(name):
    module = instantiate(small_config(name))
    values = batch(name)
    loader = DataLoader(cast(Dataset, [values]), batch_size=None)
    trainer = Trainer(
        accelerator="cpu", devices=1, fast_dev_run=True, logger=False,
        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    key = "val/generator_loss" if name == "gan" else "val/loss"
    assert torch.isfinite(trainer.callback_metrics[key])
    trainer.test(module, dataloaders=loader)
    predict_loader = DataLoader(cast(Dataset, [cast(tuple[torch.Tensor, torch.Tensor], values)[0]]), batch_size=None) if name == "mnist" else loader
    predictions = cast(list[torch.Tensor], trainer.predict(module, dataloaders=predict_loader))
    assert torch.isfinite(predictions[0]).all()


def test_old_production_files_are_removed():
    models = ROOT / "src" / "models"
    assert not (models / "legacy").exists()
    assert sorted(path.name for path in models.glob("*.py")) == ["__init__.py"]
    assert sorted(path.name for path in (models / "components").glob("*.py")) == ["__init__.py"]