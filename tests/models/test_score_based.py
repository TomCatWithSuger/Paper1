import torch

from src.models.methods.generative.diffusion.score_based import ScoreMatching
from src.models.modules.generative import ScoreBasedLitModule
from src.models.networks.dense.field import DenseScoreField


def test_dense_score_model_shapes() -> None:
    net = DenseScoreField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        noise_embedding_dim=16,
    )
    method = ScoreMatching(
        net,
        sigma_min=0.1,
        sigma_max=1.0,
        num_noise_levels=4,
        sampling_steps_per_level=1,
        sampling_step_size=1e-4,
    )
    data = torch.randn(4, 1, 28, 28)
    noise = torch.randn_like(data)
    noise_levels = torch.tensor([0.1, 0.2, 0.5, 1.0])

    noisy_samples = method.perturb(data, noise_levels, noise)
    predicted_score = net(noisy_samples, noise_levels)
    samples = method.sample(2, device=data.device, steps_per_level=1)

    assert noisy_samples.shape == data.shape
    assert predicted_score.shape == data.shape
    assert samples.shape == (2, 1, 28, 28)
    assert torch.isfinite(samples).all()


def test_score_based_model_step() -> None:
    net = DenseScoreField(
        input_shape=(1, 28, 28),
        hidden_dims=(64, 32),
        noise_embedding_dim=16,
    )
    method = ScoreMatching(
        net,
        sigma_min=0.1,
        sigma_max=1.0,
        num_noise_levels=4,
        sampling_steps_per_level=1,
        sampling_step_size=1e-4,
    )
    module = ScoreBasedLitModule(
        method=method,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    batch = (torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    loss, predicted_score, target_score, noisy_samples = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert predicted_score.shape == batch[0].shape
    assert target_score.shape == batch[0].shape
    assert noisy_samples.shape == batch[0].shape
