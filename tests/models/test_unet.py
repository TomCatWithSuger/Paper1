import torch

from src.models.components.unet_1d import UNet1D
from src.models.unet_module import UNetLitModule


def test_unet_1d_shapes() -> None:
    net = UNet1D(channels=(4, 8), kernel_size=3)
    waveforms = torch.randn(3, 257)

    reconstruction = net(waveforms)

    assert reconstruction.shape == waveforms.shape
    assert torch.isfinite(reconstruction).all()


def test_unet_model_step() -> None:
    net = UNet1D(channels=(4, 8), kernel_size=3)
    module = UNetLitModule(
        net=net,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )
    waveforms = torch.randn(3, 256)
    attention_mask = torch.ones_like(waveforms, dtype=torch.bool)
    attention_mask[1, 224:] = False
    attention_mask[2, 192:] = False
    batch = {
        "waveforms": waveforms,
        "attention_mask": attention_mask,
        "lengths": attention_mask.sum(dim=1),
    }

    loss, reconstruction = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert reconstruction.shape == waveforms.shape
