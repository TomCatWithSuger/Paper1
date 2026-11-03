"""模块说明：验证流匹配目标、积分数值、梯度、随机状态及输入校验。"""

from pathlib import Path

import hydra
import pytest
import torch

from src.data.synthetic_datamodule import SyntheticDataModule
from src.models.algorithms.flow import euler_integrate, linear_path
from src.models.methods.generative.flow.flow_matching import FlowMatching
from src.models.modules.generative import FlowMatchingLitModule
from src.models.networks.dense.field import DenseTimeField


def test_synthetic_experiment_lifecycle_and_checkpoint_resume(tmp_path: Path) -> None:
    """演练配置组装、独立反传、训练评估预测及优化器续跑，不读取外部数据。"""
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = hydra.compose(
            config_name="train.yaml",
            overrides=["experiment=flow_matching_synthetic", f"paths.output_dir={tmp_path}"],
        )
    data = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer, logger=False)
    assert isinstance(data, SyntheticDataModule)
    assert isinstance(model, FlowMatchingLitModule)
    assert len(list(model.parameters())) == len(list(model.method.network.parameters()))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = model.model_step(next(iter(data.train_dataloader())))[0]
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )
    assert trainer.validate(model, datamodule=data)[0]["val/loss"] >= 0
    assert trainer.test(model, datamodule=data)[0]["test/loss"] >= 0
    predictions = trainer.predict(model, datamodule=data)
    assert predictions is not None and len(predictions) == 1
    assert isinstance(predictions[0], torch.Tensor)
    assert predictions[0].shape == (8, 1, 8, 8)
    assert torch.isfinite(predictions[0]).all()
    generated = model.method.sample(2, device="cpu")
    assert generated.shape == (2, 1, 8, 8) and torch.isfinite(generated).all()
    checkpoint = tmp_path / "synthetic.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = hydra.utils.instantiate(config.model)
    resumed_trainer = hydra.utils.instantiate(config.trainer, max_epochs=2, logger=False)
    resumed_trainer.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed_trainer.global_step == 4
    optimizer = resumed_trainer.optimizers[0]
    assert optimizer.state and all(int(state["step"]) == 4 for state in optimizer.state.values())


def test_dense_flow_matching_shapes() -> None:
    net = DenseTimeField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
    )
    method = FlowMatching(net, integration_steps=8)
    data = torch.randn(4, 1, 28, 28)
    noise = torch.randn_like(data)
    times = torch.tensor([0.0, 0.25, 0.5, 1.0])

    path_samples = method.interpolate(data, noise, times)
    predicted_velocity = net(path_samples, times)

    assert path_samples.shape == data.shape
    assert predicted_velocity.shape == data.shape
    assert torch.equal(path_samples[0], noise[0])
    assert torch.equal(path_samples[-1], data[-1])
    assert method.sample(2, device=data.device, integration_steps=4).shape == (
        2,
        1,
        28,
        28,
    )


def test_flow_matching_model_step() -> None:
    net = DenseTimeField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
    )
    module = FlowMatchingLitModule(
        method=FlowMatching(net, integration_steps=8),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, predicted_velocity, target_velocity, path_samples = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert predicted_velocity.shape == batch[0].shape
    assert target_velocity.shape == batch[0].shape
    assert path_samples.shape == batch[0].shape


def test_linear_path_preserves_values_gradients_and_rng():
    data = torch.randn(3, 2, 5, requires_grad=True)
    noise = torch.randn_like(data, requires_grad=True)
    times = torch.rand(3, requires_grad=True)
    state = torch.random.get_rng_state()
    expanded = times.view(3, 1, 1)
    expected = (1.0 - expanded) * noise + expanded * data
    actual = linear_path(data, noise, times)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(actual.sum(), (data, noise, times))
    expected_gradients = torch.autograd.grad(expected.sum(), (data, noise, times))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())


def test_euler_preserves_values_gradients_and_rng():
    initial = torch.randn(3, 2, 5, requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor(0.3))

    def field(samples, times):
        return weight * samples + times.view(-1, 1, 1)

    state = torch.random.get_rng_state()
    expected = initial
    for step in range(4):
        times = torch.full((3,), step / 4, dtype=initial.dtype, device=initial.device)
        expected = expected + 0.25 * field(expected, times)
    actual = euler_integrate(field, initial, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual.sum(), (initial, weight)),
        torch.autograd.grad(expected.sum(), (initial, weight)),
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_euler_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="正整数"):
        euler_integrate(lambda samples, times: samples, torch.ones(2, 3), steps)


def test_flow_primitives_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="形状"):
        linear_path(torch.ones(2, 3), torch.ones(2, 1), torch.zeros(2))
    with pytest.raises(ValueError, match="times"):
        linear_path(torch.ones(2, 3), torch.ones(2, 3), torch.zeros(3))
    with pytest.raises(ValueError, match="向量场"):
        euler_integrate(lambda samples, times: samples[:, :1], torch.ones(2, 3), 2)


def test_flow_loss_rejects_broadcastable_network_output():
    class IncompatibleField(torch.nn.Module):
        def forward(self, samples, times):
            return samples[:, :1]

    with pytest.raises(ValueError, match="速度场"):
        FlowMatching(IncompatibleField()).compute_loss(torch.ones(2, 3))


def test_flow_sampling_matches_original_loop_and_rng():
    method = FlowMatching(
        DenseTimeField(input_shape=(1, 4, 4), hidden_dims=(8,), time_embedding_dim=8),
        integration_steps=3,
    ).eval()
    state = torch.random.get_rng_state()
    with torch.no_grad():
        expected = torch.randn(2, 1, 4, 4)
        for step in range(3):
            times = torch.full((2,), step / 3)
            expected = expected + (1.0 / 3) * method(expected, times)
    expected_state = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    actual = method.sample(2, device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(expected_state, torch.random.get_rng_state())
