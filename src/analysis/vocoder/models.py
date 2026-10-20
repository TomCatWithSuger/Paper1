"""加载本地声码器、语音识别与说话人评分模型。"""


# ====================
# 1. Imports
# ====================


from pathlib import Path
from typing import Any, Callable, cast

import torch
import torchaudio

from src.models.components.conditioning.meanvc import SpeakerEncoder
from src.analysis.vocoder.definitions import ASR_SAMPLE_RATE, SAMPLE_RATE
from src.analysis.vocoder.metrics import (
    _char_error_rate,
    _normalize_text,
    _word_error_rate,
)


# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _load_vocos(config_path: Path, weights_path: Path) -> Any:
    """本地加载仅用于 decode 的 Vocos，不提供原始波形特征提取。"""
    from vocos.feature_extractors import FeatureExtractor
    from vocos.pretrained import Vocos, instantiate_class
    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    backbone = instantiate_class(args=(), init=config["backbone"])
    head = instantiate_class(args=(), init=config["head"])
    model = Vocos(feature_extractor=FeatureExtractor(), backbone=backbone, head=head)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================


class _AsrScorer:
    """使用 Whisper 提取转写文本并计算 WER/CER。"""

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir)
        )
        cast(torch.nn.Module, self.model).to(device)
        self.model.eval()
        self.device = device
        normalizer = getattr(self.processor.tokenizer, "normalizer", None)
        self._normalizer: Callable[[str], str] | None = normalizer

    @torch.inference_mode()
    def transcribe(self, waveform: torch.Tensor) -> str:
        """转写单条波形（内部重采样到 16kHz）。"""
        audio = torchaudio.functional.resample(waveform.cpu(), SAMPLE_RATE, ASR_SAMPLE_RATE)
        inputs = self.processor(
            audio.numpy(), sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        try:
            generated = self.model.generate(
                input_features, language="english", task="transcribe", max_new_tokens=256
            )
        except TypeError:
            decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )
            generated = self.model.generate(
                input_features, forced_decoder_ids=decoder_ids, max_new_tokens=256
            )
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def score(self, waveform: torch.Tensor, text: str) -> dict[str, Any]:
        """转写并计算 WER/CER。"""
        hypothesis = self.transcribe(waveform)
        reference = _normalize_text(text, self._normalizer)
        normalized_hyp = _normalize_text(hypothesis, self._normalizer)
        return {
            "asr_hyp": hypothesis,
            "wer": _word_error_rate(reference, normalized_hyp),
            "cer": _char_error_rate(reference, normalized_hyp),
        }


class _SpeakerScorer:
    """使用冻结 WavLM+ECAPA 计算说话人余弦相似度（含参考向量缓存）。"""

    def __init__(
        self,
        source_path: Path,
        checkpoint_path: Path,
        wavlm_checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        self.encoder = SpeakerEncoder(
            source_path=str(source_path),
            checkpoint_path=str(checkpoint_path),
            wavlm_checkpoint_path=str(wavlm_checkpoint_path),
        ).to(device)
        self.encoder.eval()
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def embedding(self, waveform: torch.Tensor, cache_key: str | None = None) -> torch.Tensor:
        """提取说话人向量，可按 key 缓存。"""
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        vector = self.encoder.extract_embedding(waveform.to(self.device), SAMPLE_RATE)
        vector = vector / vector.norm().clamp_min(1e-12)
        if cache_key is not None:
            self._cache[cache_key] = vector
        return vector

    def similarity(
        self, waveform: torch.Tensor, reference: torch.Tensor, cache_key: str | None = None
    ) -> float:
        """计算生成波形与参考向量的余弦相似度。"""
        vector = self.embedding(waveform, cache_key)
        return float((vector * reference).sum())
