"""模块说明：提供可解析求导的微型模型，供离线评估流程测试使用。"""

from types import SimpleNamespace

import torch
from torch import nn


class AnalyticField(nn.Module):
    """线性状态项与时间项使 JVP 和有限差分有可独立计算的真值。"""

    def forward(self, states, times):
        """返回线性场，状态导数为 0.2，结束时间导数为 0.1。"""
        return states * 0.2 + times["end_time"][:, None, None] * 0.1


class AnalyticMethod(nn.Module):
    """实现诊断流程所需的方法接口，不读取外部编码器或权重。"""

    requires_source_mels = True
    inference_source_diffusion_time = 0.95

    def __init__(self):
        super().__init__()
        self.network = AnalyticField()

    def encode_condition(self, content, lengths, speaker, mask):
        """保留说话人条件，供生成流程验证条件传递。"""
        return {"speaker": speaker}

    def interpolate(self, source, noise, times):
        """按每个样本的时间在源与噪声间线性插值。"""
        times = times[:, None, None]
        return source * (1 - times) + noise * times

    def forward(self, states, starts, ends, condition, mask):
        """通过真实网络调用保留前向钩子和自动微分能力。"""
        return self.network(states, {"end_time": ends, "source_time": starts})

    def sample(self, condition, mask, integration_steps=1):
        """返回与批次及帧长匹配的有限 Mel，用于测试生成编排。"""
        return torch.ones(mask.shape[0], 80, mask.shape[-1], device=mask.device) * 0.2


class AnalyticModel(nn.Module):
    """封装微型方法，保留正常的 train/eval 与状态序列化行为。"""

    def __init__(self):
        super().__init__()
        self.method = AnalyticMethod()


class LocalAsr:
    """固定转写使 CER/WER 的预期结果不依赖本地 ASR 权重。"""

    available = True
    _normalizer = None

    def __init__(self, *args):
        self.model = nn.Linear(1, 1)

    def score(self, waveform, text):
        """返回固定转写和已知的零编辑率。"""
        return {"asr_hyp": "hello world", "cer": 0.0, "wer": 0.0}


class LocalSpeaker:
    """固定单位向量，同时提供可审计的说话人模型状态。"""

    def __init__(self, *args):
        self.encoder = SimpleNamespace(model=nn.Linear(1, 1))

    def embedding(self, waveform, cache_key=None):
        """返回固定单位向量，便于独立核验相似度计算。"""
        return torch.tensor([1.0, 0.0])


class LocalVocos(nn.Module):
    """把小型 Mel 输入解码为非静音波形，保留参数冻结与设备接口。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("fixture", torch.ones(1))

    def decode(self, mel):
        """返回幅度受控的正弦波，不加载声码器权重。"""
        return torch.sin(torch.arange(2205, device=mel.device) * 0.1)[None] * 0.1


def diagnostic_batch(count: int = 1, frames: int = 8) -> dict[str, torch.Tensor]:
    """建立保留 padding 的小批次，以验证掩码和逐样本记录。"""
    return {
        "mel_spectrograms": torch.ones(count, 80, frames) * 0.1,
        "mel_attention_mask": torch.tensor([[True] * (frames - 2) + [False] * 2] * count),
        "content_features": torch.ones(count, 256, 4),
        "content_feature_lengths": torch.full((count,), 4),
        "speaker_features": torch.ones(count, 256),
    }
