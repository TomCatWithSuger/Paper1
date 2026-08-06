import torch

from src.models.ae_module import AELitModule
from src.models.components.dense_autoencoder import DenseAutoencoder


def test_dense_autoencoder_shapes() -> None:
    net = DenseAutoencoder(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    x = torch.randn(4, 1, 28, 28)

    latent = net.encode(x)
    reconstruction = net(x)

    assert latent.shape == (4, 8)
    assert reconstruction.shape == x.shape


def test_ae_model_step() -> None:
    net = DenseAutoencoder(input_shape=(1, 28, 28), hidden_dims=(64, 32), latent_dim=8)
    module = AELitModule(
        net=net,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, reconstruction = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert reconstruction.shape == batch[0].shape
