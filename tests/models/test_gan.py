"""模块说明：验证对抗网络形状、参数角色与分阶段优化和状态隔离。"""

from copy import deepcopy
from typing import cast

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from src.models.methods.generative.adversarial.gan import Adversarial
from src.models.modules.adversarial import GANLitModule
from src.models.networks.dense.adversarial import DenseAdversarialPair


def test_dense_gan_shapes() -> None:
    net = DenseAdversarialPair(
        input_shape=(1, 28, 28),
        latent_dim=16,
        generator_hidden_dims=(32, 64),
        discriminator_hidden_dims=(64, 32),
    )
    latent = torch.randn(4, 16)

    generated_images = net(latent)
    logits = net.discriminate(generated_images)

    assert generated_images.shape == (4, 1, 28, 28)
    assert logits.shape == (4, 1)
    assert Adversarial(net).generate(4, device=latent.device).shape == (4, 1, 28, 28)


def test_gan_model_step() -> None:
    net = DenseAdversarialPair(
        input_shape=(1, 28, 28),
        latent_dim=16,
        generator_hidden_dims=(32, 64),
        discriminator_hidden_dims=(64, 32),
    )
    module = GANLitModule(
        method=Adversarial(net),
        generator_optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        discriminator_optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    generator_loss, discriminator_loss, generated_images = module.model_step(batch)
    optimizers = module.configure_optimizers()

    assert generator_loss.ndim == 0
    assert discriminator_loss.ndim == 0
    assert torch.isfinite(generator_loss)
    assert torch.isfinite(discriminator_loss)
    assert generator_loss >= 0
    assert discriminator_loss >= 0
    assert generated_images.shape == batch[0].shape
    assert len(optimizers) == 2


def small_method():
    return Adversarial(
        DenseAdversarialPair(
            input_shape=(1, 4, 4),
            latent_dim=4,
            generator_hidden_dims=(8,),
            discriminator_hidden_dims=(8,),
        )
    )


def test_optimization_roles_cover_parameters_without_duplicates():
    method = small_method()
    roles = method.optimization_roles()
    identities = [id(parameter) for parameters in roles.values() for parameter in parameters]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(parameter) for parameter in method.parameters()}
    assert method.optimization_phases == ("discriminator", "generator")
    method.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="未分配"):
        method.optimization_roles()


def test_optimization_roles_reject_shared_parameters():
    method = small_method()
    method.network.discriminator.shared = next(method.network.generator.parameters())
    with pytest.raises(ValueError, match="共享"):
        method.optimization_roles()


@pytest.mark.parametrize("role", ["discriminator", "generator"])
def test_phase_loss_preserves_gradients_buffers_and_rng(role):
    method = small_method()
    reference = deepcopy(method)
    real = torch.randn(4, 1, 4, 4)
    for candidate in (method, reference):
        for name, parameters in candidate.optimization_roles().items():
            for parameter in parameters:
                parameter.requires_grad_(name == role)
    state = torch.random.get_rng_state()
    generated = reference.generate(real.size(0), real.device)
    expected = (
        reference.discriminator_loss(real, generated.detach())
        if role == "discriminator"
        else reference.generator_loss(generated)
    )
    expected.backward()
    expected_state = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    actual = method.compute_phase_loss(role, real)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_state)
    for actual_parameter, expected_parameter in zip(method.parameters(), reference.parameters()):
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            torch.testing.assert_close(
                actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0
            )
    for actual_buffer, expected_buffer in zip(method.buffers(), reference.buffers()):
        torch.testing.assert_close(actual_buffer, expected_buffer, rtol=0, atol=0)
    before = {name: parameter.detach().clone() for name, parameter in method.named_parameters()}
    optimizer = torch.optim.SGD(method.optimization_roles()[role], lr=0.01)
    optimizer.step()
    updated = []
    for name, parameter in method.named_parameters():
        if not torch.equal(parameter, before[name]):
            updated.append(name)
            assert name.startswith(f"network.{role}.")
    assert updated


def test_phase_loss_rejects_unknown_role_without_consuming_rng():
    method = small_method()
    state = torch.random.get_rng_state()
    with pytest.raises(ValueError, match="未知"):
        method.compute_phase_loss("unknown", torch.zeros(4, 1, 4, 4))
    assert torch.equal(torch.random.get_rng_state(), state)


def test_lightning_updates_only_active_role_in_declared_order(monkeypatch):
    method = small_method()
    roles = method.optimization_roles()
    identities = {
        role: {id(parameter) for parameter in parameters} for role, parameters in roles.items()
    }
    updates = []
    original_step = torch.optim.SGD.step

    def checked_step(optimizer, closure=None):
        active = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        role = next(name for name, parameters in identities.items() if parameters == active)
        before = {id(parameter): parameter.detach().clone() for parameter in method.parameters()}
        for parameter in method.parameters():
            assert parameter.requires_grad == (id(parameter) in active)
        result = original_step(optimizer, closure=closure)
        changed = {
            id(parameter)
            for parameter in method.parameters()
            if not torch.equal(parameter, before[id(parameter)])
        }
        assert changed and changed <= active
        updates.append(role)
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", checked_step)
    module = GANLitModule(
        method,
        generator_optimizer=lambda params: torch.optim.SGD(params, lr=0.01),
        discriminator_optimizer=lambda params: torch.optim.SGD(params, lr=0.01),
    )
    loader = DataLoader(
        cast(Dataset, [(torch.randn(4, 1, 4, 4), torch.zeros(4))]), batch_size=None
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=loader)
    assert tuple(updates) == method.optimization_phases
    assert all(parameter.requires_grad for parameter in method.parameters())
