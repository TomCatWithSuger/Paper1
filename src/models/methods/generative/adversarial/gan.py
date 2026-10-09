"""生成对抗网络方法。"""

import torch
from torch import nn


class Adversarial(nn.Module):
    """组织生成器、判别器及对应的对抗损失。"""

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, latent):
        return self.network(latent)

    def generator_loss(self, generated_images):
        """计算生成样本被判定为真实的生成器损失。"""
        logits = self.network.discriminate(generated_images)
        return self.criterion(logits, torch.ones_like(logits))

    def discriminator_loss(self, real_images, generated_images):
        """计算真实与生成样本的平均判别器损失。"""
        real = self.network.discriminate(real_images)
        generated = self.network.discriminate(generated_images)
        return 0.5 * (
            self.criterion(real, torch.ones_like(real))
            + self.criterion(generated, torch.zeros_like(generated))
        )

    def generate(self, num_samples, device):
        """从标准高斯潜变量生成指定数量的样本。"""
        return self(torch.randn(num_samples, self.network.latent_dim, device=device))

    def compute_loss(self, real):
        """返回生成器损失、判别器损失和生成样本。"""
        generated = self.generate(real.size(0), real.device)
        generator_loss = self.generator_loss(generated)
        discriminator_loss = self.discriminator_loss(real, generated.detach())
        return generator_loss, discriminator_loss, generated
