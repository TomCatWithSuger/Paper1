import torch

from src.models.legacy.dense_vae import DenseVAE
from src.models.vae_module import VAELitModule


def test_dense_vae_shapes() -> None:
    net = DenseVAE(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    x = torch.randn(4, 1, 28, 28)

    reconstruction, mu, logvar = net(x)

    assert reconstruction.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert net.sample(3, device=x.device).shape == (3, 1, 28, 28)


def test_vae_model_step() -> None:
    net = DenseVAE(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    module = VAELitModule(
        net=net,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
        beta=0.5,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, reconstruction_loss, kl_loss, reconstruction = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert reconstruction_loss >= 0
    assert kl_loss >= 0
    assert reconstruction.shape == batch[0].shape
