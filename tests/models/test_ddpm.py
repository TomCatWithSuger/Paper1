import torch

from src.models.methods.generative.diffusion.ddpm import DDPM
from src.models.modules.generative import DDPMLitModule
from src.models.networks.dense.field import DenseNoiseField


def test_dense_ddpm_shapes() -> None:
    net = DenseNoiseField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
    )
    method = DDPM(net, timesteps=8)
    x = torch.randn(4, 1, 28, 28)
    timesteps = torch.tensor([0, 1, 4, 7], dtype=torch.long)
    noise = torch.randn_like(x)

    noisy_images = method.q_sample(x, timesteps, noise)
    predicted_noise = net(x=noisy_images, timesteps=timesteps)

    assert noisy_images.shape == x.shape
    assert predicted_noise.shape == noise.shape
    assert method.sample(2, device=x.device).shape == (2, 1, 28, 28)


def test_ddpm_model_step() -> None:
    net = DenseNoiseField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        time_embedding_dim=16,
    )
    module = DDPMLitModule(
        method=DDPM(net, timesteps=8),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, predicted_noise, noise, noisy_images = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert predicted_noise.shape == batch[0].shape
    assert noise.shape == batch[0].shape
    assert noisy_images.shape == batch[0].shape
