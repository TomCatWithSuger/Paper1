import torch

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
