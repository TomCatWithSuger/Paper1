"""验证条件归一化的有效帧不受批内 padding 影响，并保留旧权重行为。"""

# ====================
# 1. Imports
# ====================

import torch
import torch.nn.functional as F

from src.models.components.conditioning.meanvc import CachedConditionEncoder

# ====================
# 3. Utilities
# 通用工具函数
# ====================


def _case(masked):
    torch.manual_seed(17)
    encoder = CachedConditionEncoder(
        content_dim=3, speaker_dim=2, output_dim=4, mask_padding_in_normalization=masked
    )
    content = torch.randn(1, 3, 8)
    lengths = torch.tensor([8])
    speaker = torch.randn(1, 2)
    mask = torch.ones(1, 12, dtype=torch.bool)
    return encoder, content, lengths, speaker, mask


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def test_masked_condition_values_and_gradients_ignore_extra_padding():
    encoder, content, lengths, speaker, mask = _case(True)
    plain = encoder(content, lengths, speaker, mask)
    plain.square().sum().backward()
    first_gradients = [p.grad.clone() for p in encoder.parameters()]
    encoder.zero_grad(set_to_none=True)
    padded = encoder(content, lengths, speaker, F.pad(mask, (0, 36), value=False))
    torch.testing.assert_close(plain, padded[:, :, :12], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(padded[:, :, 12:]) == 0
    padded.square().sum().backward()
    for before, parameter in zip(first_gradients, encoder.parameters()):
        torch.testing.assert_close(before, parameter.grad, atol=1e-5, rtol=1e-5)


def test_legacy_checkpoint_keys_and_unpadded_outputs_are_preserved():
    legacy, content, lengths, speaker, mask = _case(False)
    fixed, *_ = _case(True)
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.equal(
        legacy(content, lengths, speaker, mask), fixed(content, lengths, speaker, mask)
    )
    padded_mask = F.pad(mask, (0, 36), value=False)
    before = legacy(content, lengths, speaker, padded_mask)
    after = fixed(content, lengths, speaker, padded_mask)
    assert not torch.allclose(before[:, :, :12], after[:, :, :12])
