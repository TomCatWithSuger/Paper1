import torch

from src.models.methods.generative.latent_variable.vae import VariationalAutoencoder
from src.models.modules.reconstruction import VAELitModule
from src.models.networks.dense.autoencoder import DenseGaussianAutoencoder


def test_dense_vae_shapes() -> None:
    net = DenseGaussianAutoencoder(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    method = VariationalAutoencoder(net)
    x = torch.randn(4, 1, 28, 28)

    reconstruction, mu, logvar = method(x)

    assert reconstruction.shape == x.shape
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert method.sample(3, device=x.device).shape == (3, 1, 28, 28)


def test_vae_model_step() -> None:
    net = DenseGaussianAutoencoder(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    module = VAELitModule(
        method=VariationalAutoencoder(net, beta=0.5),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, reconstruction_loss, kl_loss, reconstruction = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert reconstruction_loss >= 0
    assert kl_loss >= 0
    assert reconstruction.shape == batch[0].shape
