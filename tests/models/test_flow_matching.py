import torch

from src.models.flow_matching_module import FlowMatchingLitModule
from src.models.legacy.dense_flow_matching import DenseFlowMatching


def test_dense_flow_matching_shapes() -> None:
    net = DenseFlowMatching(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
        integration_steps=8,
    )
    data = torch.randn(4, 1, 28, 28)
    noise = torch.randn_like(data)
    times = torch.tensor([0.0, 0.25, 0.5, 1.0])

    path_samples = net.interpolate(data, noise, times)
    predicted_velocity = net(path_samples, times)

    assert path_samples.shape == data.shape
    assert predicted_velocity.shape == data.shape
    assert torch.equal(path_samples[0], noise[0])
    assert torch.equal(path_samples[-1], data[-1])
    assert net.sample(2, device=data.device, integration_steps=4).shape == (2, 1, 28, 28)


def test_flow_matching_model_step() -> None:
    net = DenseFlowMatching(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
        integration_steps=8,
    )
    module = FlowMatchingLitModule(
        net=net,
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
