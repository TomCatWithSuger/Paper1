"""语音四层架构、配置及 Lightning 生命周期回归。"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.methods.generative.flow.cfm.conditional_flow_matching import (
    ConditionalFlowMatching,
)
from src.models.methods.generative.flow.mean_flow.conditional_mean_flow import (
    ConditionalMeanFlow,
)
from src.models.methods.generative.flow.mean_flow.meanvoiceflow import MeanVoiceFlow
from src.models.modules.voice_flow_module import VoiceFlowModule
from src.models.networks.unet.conditional import UNet

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    (
        "conditional_flow_matching",
        ConditionalFlowMatching,
        ("time",),
    ),
    (
        "conditional_flow_matching_transformer",
        ConditionalFlowMatching,
        ("time",),
    ),
    (
        "conditional_mean_flow",
        ConditionalMeanFlow,
        ("time", "interval"),
    ),
    (
        "meanvoiceflow",
        MeanVoiceFlow,
        ("time", "interval", "source_time"),
    ),
]


def small_config(name):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml")
    cfg.method.condition_encoder.update(content_dim=8, speaker_dim=8, output_dim=8)
    cfg.method.network.update(n_mels=8, condition_dim=8)
    if name.endswith("transformer"):
        cfg.method.network.update(hidden_dim=16, num_layers=1, num_heads=2)
    else:
        cfg.method.network.update(hidden_channels=16, time_embedding_dim=8)
    if name.startswith("conditional_flow_matching"):
        cfg.method.integration_steps = 2
    return cfg


def batch():
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, 10:] = False
    return {
        "mel_spectrograms": torch.randn(2, 8, 13),
        "mel_attention_mask": mask,
        "content_features": torch.randn(2, 8, 11),
        "content_feature_lengths": torch.tensor([11, 9]),
        "speaker_features": torch.randn(2, 8),
        "reference_speaker_features": torch.randn(2, 8),
    }


@pytest.mark.parametrize("name,method_type,fields", CASES)
def test_config_ownership_and_training(name, method_type, fields):
    module = instantiate(small_config(name))
    assert type(module) is VoiceFlowModule
    assert isinstance(module.method, method_type)
    assert isinstance(module.method, nn.Module)
    assert not isinstance(module.method, LightningModule)
    assert not hasattr(module, "net")
    assert not hasattr(module, "condition_encoder")
    assert tuple(module.method._modules) == ("network", "condition_encoder")
    assert not hasattr(module.method.network, "sample")
    assert not hasattr(module.method.network, "interpolate")
    assert tuple(module.method.network.time_fields) == fields
    names = [name for name, _ in module.named_parameters(remove_duplicate=False)]
    assert all(name.startswith("method.") for name in names)
    assert len(names) == len({id(p) for p in module.parameters()})
    values = batch()
    loss, prediction, target, _ = module.model_step(values)
    assert torch.isfinite(loss)
    assert not target.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in module.method.condition_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    generated = module.predict_step(values, 0)
    assert generated.shape == prediction.shape
    assert torch.count_nonzero(generated[1, :, 10:]) == 0


@pytest.mark.parametrize(
    "fields", [("time",), ("time", "interval"), ("time", "interval", "source_time")]
)
def test_independent_time_embeddings_sum(fields):
    network = UNet(8, 8, 16, 8, time_fields=fields)
    times = {field: torch.rand(2) for field in fields}
    states, condition = torch.randn(2, 8, 13), torch.randn(2, 8, 13)
    embedding = network.time_embedding(times["time"])
    if "interval" in fields:
        embedding = embedding + network.interval_embedding(times["interval"])
    if "source_time" in fields:
        embedding = embedding + network.source_diffusion_embedding(times["source_time"])
    expected = network._forward_with_time_embedding(states, condition, embedding, None)
    torch.testing.assert_close(network(states, times, condition), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="时间字段"):
        network(states, {"wrong": torch.rand(2)}, condition)


@pytest.mark.parametrize("fields", [(), ("interval",), ("time", "time"), ("time", "unknown")])
def test_invalid_time_fields(fields):
    with pytest.raises(ValueError):
        UNet(8, 8, 16, 8, time_fields=fields)


@pytest.mark.parametrize(
    "experiment",
    ["conditional_flow_matching_vctk", "conditional_mean_flow_vctk", "meanvoiceflow_vctk"],
)
def test_experiment_nested_overrides(experiment):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "model.method.network.hidden_channels=16",
                "model.method.condition_encoder.output_dim=8",
                "model.method.network.condition_dim=8",
            ],
        )
    assert "net" not in cfg.model and "condition_encoder" not in cfg.model
    assert cfg.model.method.network.hidden_channels == 16
    if experiment == "conditional_mean_flow_vctk":
        assert cfg.model.validation_sampling_count == 4
        assert cfg.model.test_sampling_count == 8
        assert cfg.model.evaluation_seed == 12345
    if experiment == "meanvoiceflow_vctk":
        assert cfg.model.warmup_steps == 10000
        assert cfg.model.method.conditional_input_probability == 0.5


def test_optimizer_resume_requires_verified_names():
    module = instantiate(small_config("meanvoiceflow"))
    module._trainer = SimpleNamespace(estimated_stepping_batches=20)
    with pytest.raises(RuntimeError, match="optimizer"):
        module.on_load_checkpoint({"optimizer_states": [{}]})
    checkpoint = {"optimizer_states": [{}]}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("name", [case[0] for case in CASES])
def test_lightning_fit_validate_test_predict(name):
    module = instantiate(small_config(name))
    loader = DataLoader(cast(Dataset, [batch()]), batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        inference_mode=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
    trainer.test(module, dataloaders=loader)
    predictions = cast(list[torch.Tensor], trainer.predict(module, dataloaders=loader))
    assert predictions[0].shape == (2, 8, 13)
