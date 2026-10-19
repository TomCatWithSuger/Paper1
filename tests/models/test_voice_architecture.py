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
        "speaker_ids": ["speaker_a", "speaker_b"],
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


def test_meanvoiceflow_fixed_evaluation_and_random_training():
    module = instantiate(small_config("meanvoiceflow")).eval()
    values = batch()
    with torch.no_grad():
        first = module.model_step(values, generator=torch.Generator().manual_seed(12345))
        second = module.model_step(values, generator=torch.Generator().manual_seed(12345))
        random_first = module.model_step(values)
        random_second = module.model_step(values)
        state = torch.random.get_rng_state().clone()
        actual = module._evaluation_loss(values, 0, 4, 0)
        repeated = module._evaluation_loss(values, 0, 4, 0)
        expected = torch.stack([
            module.model_step(values, generator=torch.Generator().manual_seed(12345 + index))[0]
            for index in range(4)
        ]).mean()
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert not torch.equal(random_first[3], random_second[3])
    assert torch.equal(actual, repeated)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("batch_size", [1, 2, 5, 32])
def test_meanvoiceflow_exact_conditional_prior(batch_size):
    method = instantiate(small_config("meanvoiceflow")).method
    noise = torch.randn(batch_size, 8, 13, requires_grad=True)
    speaker = torch.randn(batch_size, 8)
    speaker_ids = [str(index % 2) for index in range(batch_size)]
    mask = torch.ones(batch_size, 13, dtype=torch.bool)
    prior, times, selected = method._build_training_prior(
        noise, torch.randn(batch_size, 8, 11), torch.full((batch_size,), 11),
        speaker, mask, torch.Generator().manual_seed(12), speaker_ids,
    )
    assert selected.sum() == batch_size // 2
    assert torch.equal(prior[~selected], noise[~selected])
    assert torch.all(times[~selected] == 1)
    assert torch.all((times[selected] > 0) & (times[selected] < 1))
    if batch_size > 1:
        assert not prior.requires_grad
        shuffled, valid = method._shuffle_speaker_features(
            torch.arange(batch_size).float().unsqueeze(1), speaker_ids,
            torch.Generator().manual_seed(12),
        )
        assert torch.sort(shuffled[:, 0]).values.tolist() == list(range(batch_size))
        for index in torch.nonzero(valid).flatten().tolist():
            assert speaker_ids[index] != speaker_ids[int(shuffled[index, 0])]


def test_meanvoiceflow_unavailable_speaker_pairs_warn_and_fall_back():
    method = instantiate(small_config("meanvoiceflow")).method
    values = batch()
    values["speaker_ids"] = ["same", "same"]
    module = instantiate(small_config("meanvoiceflow"))
    with pytest.warns(RuntimeWarning, match="不同说话人配对不足"):
        loss = module.model_step(values)[0]
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="speaker_ids"):
        method._shuffle_speaker_features(torch.randn(2, 8), None)


def test_meanvoiceflow_ssim_margin_padding_and_empty_masks():
    method = instantiate(small_config("meanvoiceflow")).method
    mask = torch.ones(2, 13, dtype=torch.bool)
    for target in (torch.zeros(2, 8, 13), torch.randn(2, 8, 13)):
        prediction = target.clone().requires_grad_()
        loss = method._masked_ssim_loss(prediction, target, mask)
        torch.testing.assert_close(loss, torch.tensor(0.3))
        loss.backward()
        assert torch.count_nonzero(prediction.grad) == 0

    target = torch.randn(2, 8, 13)
    prediction = torch.randn_like(target).requires_grad_()
    actual = method._masked_ssim_loss(prediction, target, mask)
    padded = method._masked_ssim_loss(
        torch.nn.functional.pad(prediction, (0, 5), value=10000),
        torch.nn.functional.pad(target, (0, 5), value=-10000),
        torch.nn.functional.pad(mask, (0, 5), value=False),
    )
    torch.testing.assert_close(actual, padded)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0
    empty_loss = method._masked_ssim_loss(prediction, target, torch.zeros_like(mask))
    assert empty_loss == 0
    empty_loss.backward()


def test_meanvoiceflow_logit_normal_and_equal_time_sampling():
    method = instantiate(small_config("meanvoiceflow")).method
    generator = torch.Generator().manual_seed(43)
    expected = torch.sigmoid(torch.randn(10000, generator=generator))
    actual = method._sample_logit_normal(
        10000, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(43)
    )
    assert torch.equal(actual, expected)
    starts, ends = method._sample_time_intervals(
        10000, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(44)
    )
    assert torch.all((starts > 0) & (starts <= ends) & (ends < 1))
    assert abs((starts == ends).float().mean().item() - 0.75) < 0.02


def test_meanvoiceflow_adaptive_loss_formula_and_gradient():
    method = instantiate(small_config("meanvoiceflow")).method
    prediction = torch.randn(2, 8, 13, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[1, 9:] = False
    valid = mask.unsqueeze(1).to(prediction.dtype)
    energy = ((prediction - target).square() * valid).sum(dim=(1, 2))
    expected = (energy / (energy + 1e-3).detach()).mean()
    actual = method._adaptive_mean_flow_loss(prediction, target, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected_gradient = (
        2 * (prediction.detach() - target) * valid
        / (energy.detach() + 1e-3).view(-1, 1, 1) / prediction.size(0)
    )
    torch.testing.assert_close(prediction.grad, expected_gradient)
    assert torch.count_nonzero(prediction.grad[1, :, 9:]) == 0


def test_meanvoiceflow_pseudo_source_formula(monkeypatch):
    method = instantiate(small_config("meanvoiceflow")).method

    def constant_velocity(states, starts, ends, source_times, condition, mask):
        assert not torch.is_grad_enabled()
        return torch.full_like(states, 2.0)

    monkeypatch.setattr(method, "forward", constant_velocity)
    values = batch()
    noise = torch.randn(2, 8, 13)
    prior, times, selected = method._build_training_prior(
        noise, values["content_features"], values["content_feature_lengths"],
        values["speaker_features"], values["mel_attention_mask"],
        torch.Generator().manual_seed(45), values["speaker_ids"],
    )
    expected = noise - 2 * (1 - times).view(-1, 1, 1)
    torch.testing.assert_close(prior, expected)
    assert selected.sum() == 1
    assert not prior.requires_grad


def test_meanvoiceflow_analytic_jvp_target_and_stop_gradient(monkeypatch):
    method = instantiate(small_config("meanvoiceflow")).method
    scale = nn.Parameter(torch.tensor(2.0))

    def analytic_velocity(states, starts, ends, source_times, condition, mask):
        return (scale * states + 3 * ends.view(-1, 1, 1)
                + 5 * (ends - starts).view(-1, 1, 1)
                + 7 * source_times.view(-1, 1, 1))

    def pure_noise_prior(noise, *args):
        return noise, torch.ones(noise.size(0)), torch.zeros(noise.size(0), dtype=torch.bool)

    starts = torch.tensor([0.1, 0.6])
    ends = torch.tensor([0.8, 0.6])
    monkeypatch.setattr(method, "forward", analytic_velocity)
    monkeypatch.setattr(method, "_build_training_prior", pure_noise_prior)
    monkeypatch.setattr(method, "_sample_time_intervals", lambda *args: (starts, ends))
    values = batch()
    generator = torch.Generator().manual_seed(46)
    noise = torch.randn(values["mel_spectrograms"].shape, generator=generator)
    instantaneous = noise - values["mel_spectrograms"]
    expected = instantaneous - (ends - starts).view(-1, 1, 1) * (2 * instantaneous + 8)
    loss, prediction, target, _ = method.compute_loss(
        values["mel_spectrograms"], values["mel_attention_mask"],
        values["content_features"], values["content_feature_lengths"],
        values["speaker_features"], generator=torch.Generator().manual_seed(46),
    )
    torch.testing.assert_close(target, expected)
    torch.testing.assert_close(target[1], instantaneous[1])
    assert not target.requires_grad
    assert prediction.requires_grad
    loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)


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
