import torch

from src.models.dit_module import DiTLitModule
from src.models.legacy.dit import DiT1D


def test_dit_shapes() -> None:
    net = DiT1D(
        in_channels=1,
        hidden_dim=32,
        context_dim=16,
        num_layers=1,
        num_heads=2,
        integration_steps=4,
    )
    x = torch.randn(2, 320, 1)
    times = torch.rand(2)
    velocity = net(x, times)

    assert velocity.shape == x.shape
    assert net.sample(320, device="cpu", integration_steps=2).shape == (320, 1)


def test_dit_model_step() -> None:
    net = DiT1D(
        in_channels=1,
        hidden_dim=32,
        context_dim=16,
        num_layers=1,
        num_heads=2,
        integration_steps=4,
    )
    module = DiTLitModule(
        net=net,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = {
        "waveforms": torch.randn(3, 400),
        "lengths": torch.tensor([400, 350, 300]),
        "attention_mask": torch.ones(3, 400, dtype=torch.bool),
    }
    loss, predicted_velocity, target_velocity, path_samples = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert predicted_velocity.shape == (3, 400, 1)
    assert target_velocity.shape == (3, 400, 1)
    assert path_samples.shape == (3, 400, 1)
