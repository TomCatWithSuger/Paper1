"""带流时间调制的 Transformer 基础块。"""

from torch import nn


class ConditionalTransformerBlock(nn.Module):
    """使用全局条件调制注意力和前馈子层。"""

    def __init__(self, hidden_dim: int, num_heads: int, condition_dim: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.modulation = nn.Linear(condition_dim, hidden_dim * 4)

    @staticmethod
    def _modulate(hidden, scale, shift):
        return hidden * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, hidden, condition, padding_mask):
        attention_scale, attention_shift, ffn_scale, ffn_shift = self.modulation(condition).chunk(4, dim=1)
        normalized = self._modulate(self.norm1(hidden), attention_scale, attention_shift)
        attended, _ = self.attention(
            query=normalized, key=normalized, value=normalized,
            key_padding_mask=padding_mask, need_weights=False,
        )
        hidden = hidden + attended
        normalized = self._modulate(self.norm2(hidden), ffn_scale, ffn_shift)
        return hidden + self.feed_forward(normalized)
