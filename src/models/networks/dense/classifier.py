"""带批归一化的全连接分类网络。"""

from torch import nn


class SimpleDenseNet(nn.Module):
    """将固定尺寸输入映射为多分类 logits。"""

    def __init__(
        self, input_size=784, lin1_size=256, lin2_size=256, lin3_size=256, output_size=10
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model(x.view(batch_size, -1))
