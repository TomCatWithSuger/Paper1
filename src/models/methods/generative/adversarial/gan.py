"""模块说明：生成对抗网络方法。"""

import torch
from torch import nn


class Adversarial(nn.Module):
    """组织生成器、判别器及对应的对抗损失。"""

    optimization_phases = ("discriminator", "generator")

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, latent):
        return self.network(latent)

    def optimization_roles(self):
        """返回无重复、无交集且覆盖可训练参数的优化角色。"""
        roles = {
            "generator": tuple(self.network.generator.parameters()),
            "discriminator": tuple(self.network.discriminator.parameters()),
        }
        owned = {id(parameter) for parameter in self.parameters()}
        assigned = set()
        for role, parameters in roles.items():
            identities = [id(parameter) for parameter in parameters]
            if not identities:
                raise ValueError(f"优化角色 {role} 不能为空")
            if len(set(identities)) != len(identities):
                raise ValueError(f"优化角色 {role} 存在重复参数")
            if not set(identities) <= owned:
                raise ValueError(f"优化角色 {role} 包含 Method 以外的参数")
            if assigned.intersection(identities):
                raise ValueError("GAN 优化角色不允许共享参数")
            assigned.update(identities)
        if any(
            parameter.requires_grad and id(parameter) not in assigned
            for parameter in self.parameters()
        ):
            raise ValueError("存在未分配优化角色的可训练参数")
        return roles

    def compute_phase_loss(self, role, real_images):
        """构造阶段目标；两个阶段分别采样，保持原有训练语义。"""
        if role not in self.optimization_phases:
            raise ValueError(f"未知 GAN 更新阶段: {role}")
        generated = self.generate(real_images.size(0), real_images.device)
        if role == "discriminator":
            return self.discriminator_loss(real_images, generated.detach())
        return self.generator_loss(generated)

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
