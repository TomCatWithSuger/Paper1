"""时间条件的一维波形 Transformer。"""

from torch import nn

from src.models.components.blocks.waveform import TransformerBlock
from src.models.components.embeddings.dense_time import ContinuousTimeEmbedding


class DiT1D(nn.Module):
    """使用连续时间调制的一维波形 Transformer。"""

    def __init__(self, in_channels=1, hidden_dim=64, context_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        for value in (in_channels, hidden_dim, context_dim, num_layers, num_heads):
            if value <= 0:
                raise ValueError("network dimensions must be positive")
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, context_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, waveforms, times):
        if waveforms.ndim != 3:
            raise ValueError("waveforms must have shape [batch, time, channels]")
        if waveforms.size(-1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {waveforms.size(-1)}"
            )
        hidden = self.input_proj(waveforms)
        context = self.time_embedding(times)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output_proj(hidden)
