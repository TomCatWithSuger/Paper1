"""波形网络的卷积与上下文注意力块。"""

from torch import nn


class ConvBlock1D(nn.Module):
    """保持时间长度的一维卷积特征块。"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels), nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels), nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class TransformerBlock(nn.Module):
    """使用外部上下文调制波形序列表示。"""

    def __init__(self, embed_dim, num_heads, context_dim):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), nn.Linear(embed_dim * 4, embed_dim),
        )
        self.condition_proj = nn.Linear(context_dim, embed_dim)

    def forward(self, x, context):
        context = self.condition_proj(context).unsqueeze(1)
        hidden = self.norm(x)
        hidden, _ = self.self_attention(query=hidden, key=context, value=context)
        x = x + hidden
        return x + self.feed_forward(x)