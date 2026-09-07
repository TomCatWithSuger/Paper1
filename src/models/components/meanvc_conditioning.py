"""使用 MeanVC 预训练模型提取内容条件和说话人条件。"""

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn


def _resample_waveform(
    waveform: torch.Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """使用线性插值将单条波形重采样到目标采样率。"""
    if source_sample_rate == target_sample_rate:
        return waveform
    output_length = round(waveform.numel() * target_sample_rate / source_sample_rate)
    return F.interpolate(
        waveform.view(1, 1, -1),
        size=output_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


class ContentEncoder(nn.Module):
    """使用 MeanVC 的冻结 FastU2++ 模型提取 256 维内容特征。"""

    def __init__(
        self,
        checkpoint_path: str,
        feature_sample_rate: int = 16_000,
        output_dim: int = 256,
        decoding_chunk_size: int = 5,
        num_decoding_left_chunks: int = 2,
        subsampling: int = 4,
        context: int = 7,
        upsample_factor: int = 4,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if (
            min(
                feature_sample_rate,
                output_dim,
                decoding_chunk_size,
                subsampling,
                context,
                upsample_factor,
            )
            <= 0
        ):
            raise ValueError("content encoder settings must be positive")
        if num_decoding_left_chunks < 0:
            raise ValueError("num_decoding_left_chunks must not be negative")

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if model is None:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"FastU2++ checkpoint not found: {checkpoint}")
            model = torch.jit.load(str(checkpoint), map_location="cpu")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        object.__setattr__(self, "_model", model)
        self.feature_sample_rate = feature_sample_rate
        self.output_dim = output_dim
        self.decoding_chunk_size = decoding_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.subsampling = subsampling
        self.context = context
        self.upsample_factor = upsample_factor

    @property
    def model(self) -> nn.Module:
        """返回冻结的 FastU2++ 模型。"""
        return object.__getattribute__(self, "_model")

    def train(self, mode: bool = True) -> "ContentEncoder":
        """始终保持预训练模型处于推理模式。"""
        super().train(False)
        self.model.eval()
        return self

    def _extract_one(self, waveform: torch.Tensor) -> torch.Tensor:
        """按照 MeanVC 的流式参数提取一条语音的瓶颈特征。"""
        try:
            import torchaudio.compliance.kaldi as kaldi
        except ImportError as error:
            raise ImportError("MeanVC 内容编码器需要安装 torchaudio==2.5.1") from error

        fbanks = kaldi.fbank(
            (waveform * (1 << 15)).unsqueeze(0),
            frame_length=25,
            frame_shift=10,
            snip_edges=True,
            num_mel_bins=80,
            energy_floor=0.0,
            dither=0.0,
            sample_frequency=self.feature_sample_rate,
        ).unsqueeze(0)

        offset = 0
        stride = self.subsampling * self.decoding_chunk_size
        required_cache_size = self.decoding_chunk_size * self.num_decoding_left_chunks
        decoding_window = (self.decoding_chunk_size - 1) * self.subsampling + self.context
        att_cache = torch.zeros((0, 0, 0, 0), device=waveform.device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=waveform.device)
        chunks = []
        for start in range(0, fbanks.size(1), stride):
            fbank_chunk = fbanks[:, start : start + decoding_window]
            if fbank_chunk.size(1) < required_cache_size:
                fbank_chunk = F.pad(
                    fbank_chunk,
                    (0, 0, 0, required_cache_size - fbank_chunk.size(1)),
                )
            encoder_output, att_cache, cnn_cache = self.model.forward_encoder_chunk(
                fbank_chunk,
                offset,
                required_cache_size,
                att_cache,
                cnn_cache,
            )
            offset += encoder_output.size(1)
            chunks.append(encoder_output)
        if not chunks:
            raise ValueError("waveform is too short for FastU2++ feature extraction")
        features = torch.cat(chunks, dim=1)
        if features.size(2) != self.output_dim:
            raise ValueError(
                f"expected {self.output_dim} content channels, got {features.size(2)}"
            )
        return features.transpose(1, 2)

    @torch.no_grad()
    def extract_features(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """提取单条语音的四倍上采样内容特征。"""
        if waveform.ndim != 1:
            raise ValueError("waveform must have shape [samples]")
        self.model.to(waveform.device)
        self.model.eval()
        waveform = _resample_waveform(
            waveform,
            source_sample_rate=sample_rate,
            target_sample_rate=self.feature_sample_rate,
        )
        features = self._extract_one(waveform)
        return F.interpolate(
            features,
            size=features.size(2) * self.upsample_factor,
            mode="linear",
            align_corners=True,
        )[0]

    @torch.no_grad()
    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        target_frames: int,
        sample_rate: int,
    ) -> torch.Tensor:
        """返回与目标 Mel 有效帧逐条对齐的内容特征。"""
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        self.model.to(waveforms.device)
        self.model.eval()

        batch_features = waveforms.new_zeros(waveforms.size(0), self.output_dim, target_frames)
        for index in range(waveforms.size(0)):
            waveform = waveforms[index, : int(lengths[index].item())]
            features = self.extract_features(waveform, sample_rate).unsqueeze(0)
            output_length = min(int(target_lengths[index].item()), target_frames)
            aligned_length = min(features.size(2), output_length)
            batch_features[index, :, :aligned_length] = features[0, :, :aligned_length]
        return batch_features


class SpeakerEncoder(nn.Module):
    """使用 MeanVC 的冻结 WavLM Large ECAPA-TDNN 提取说话人向量。"""

    def __init__(
        self,
        source_path: str,
        checkpoint_path: str,
        wavlm_checkpoint_path: str,
        feature_sample_rate: int = 16_000,
        output_dim: int = 256,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if feature_sample_rate <= 0 or output_dim <= 0:
            raise ValueError("speaker encoder settings must be positive")
        source = Path(source_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        wavlm_checkpoint = Path(wavlm_checkpoint_path).expanduser().resolve()
        if model is None:
            if not source.is_file():
                raise FileNotFoundError(f"MeanVC speaker encoder source not found: {source}")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"WavLM speaker checkpoint not found: {checkpoint}")
            if not wavlm_checkpoint.is_file():
                raise FileNotFoundError(f"WavLM checkpoint not found: {wavlm_checkpoint}")
            module = self._load_source_module(source)
            original_hub_load = torch.hub.load

            def load_local_wavlm(
                repo_or_dir: str, model_name: str, *args: object, **kwargs: object
            ) -> nn.Module:
                if repo_or_dir == "s3prl/s3prl" and model_name == "wavlm_large":
                    from s3prl.upstream.wavlm.hubconf import wavlm_local

                    return wavlm_local(str(wavlm_checkpoint))
                return original_hub_load(repo_or_dir, model_name, *args, **kwargs)

            torch.hub.load = load_local_wavlm
            try:
                model = module.ECAPA_TDNN_SMALL(
                    feat_dim=1024,
                    emb_dim=output_dim,
                    feat_type="wavlm_large",
                    sr=feature_sample_rate,
                    update_extract=False,
                )
            finally:
                torch.hub.load = original_hub_load
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict["model"], strict=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        object.__setattr__(self, "_model", model)
        self.feature_sample_rate = feature_sample_rate
        self.output_dim = output_dim

    @staticmethod
    def _load_source_module(source_path: Path) -> ModuleType:
        """从 MeanVC 源文件加载 ECAPA-TDNN 定义。"""
        spec = importlib.util.spec_from_file_location("meanvc_ecapa_tdnn", source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import MeanVC speaker encoder from {source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @property
    def model(self) -> nn.Module:
        """返回冻结的说话人模型。"""
        return object.__getattribute__(self, "_model")

    def train(self, mode: bool = True) -> "SpeakerEncoder":
        """始终保持预训练模型处于推理模式。"""
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def extract_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """提取单条语音的说话人向量。"""
        if waveform.ndim != 1:
            raise ValueError("waveform must have shape [samples]")
        self.model.to(waveform.device)
        self.model.eval()
        waveform = _resample_waveform(
            waveform,
            source_sample_rate=sample_rate,
            target_sample_rate=self.feature_sample_rate,
        )
        embedding = self.model(waveform.unsqueeze(0))[0]
        if embedding.ndim != 1 or embedding.size(0) != self.output_dim:
            raise ValueError(
                f"expected speaker embedding [{self.output_dim}], got {tuple(embedding.shape)}"
            )
        return embedding

    @torch.no_grad()
    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """逐条去除 padding 后提取说话人向量。"""
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        self.model.to(waveforms.device)
        self.model.eval()

        embeddings = []
        for index in range(waveforms.size(0)):
            waveform = waveforms[index, : int(lengths[index].item())]
            embeddings.append(self.extract_embedding(waveform, sample_rate))
        return torch.stack(embeddings)


class CachedConditionEncoder(nn.Module):
    """将离线内容特征和说话人向量融合为条件 ``h``。"""

    def __init__(
        self,
        content_dim: int = 256,
        speaker_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if content_dim <= 0 or speaker_dim <= 0 or output_dim <= 0:
            raise ValueError("condition dimensions must be positive")
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(content_dim + speaker_dim, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将缓存特征逐条对齐到目标 Mel 长度。"""
        if content_features.ndim != 3 or content_features.size(1) != self.content_dim:
            raise ValueError("content_features must have shape [batch, content_dim, frames]")
        if speaker_features.ndim != 2 or speaker_features.size(1) != self.speaker_dim:
            raise ValueError("speaker_features must have shape [batch, speaker_dim]")
        target_lengths = target_mask.sum(dim=1)
        target_frames = target_mask.size(1)
        aligned_content = content_features.new_zeros(
            content_features.size(0), self.content_dim, target_frames
        )
        for index in range(content_features.size(0)):
            source_length = int(content_lengths[index].item())
            target_length = int(target_lengths[index].item())
            aligned = F.interpolate(
                content_features[index : index + 1, :, :source_length],
                size=target_length,
                mode="linear",
                align_corners=True,
            )
            aligned_content[index, :, :target_length] = aligned[0]
        expanded_speaker = speaker_features.unsqueeze(2).expand(-1, -1, target_frames)
        condition = self.fusion(torch.cat((aligned_content, expanded_speaker), dim=1))
        return condition * target_mask.unsqueeze(1).to(condition.dtype)


class ConditionEncoder(nn.Module):
    """融合冻结的 MeanVC 内容特征和说话人特征。"""

    def __init__(
        self,
        content_encoder: nn.Module,
        speaker_encoder: nn.Module,
        content_dim: int = 256,
        speaker_dim: int = 256,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        if content_dim <= 0 or speaker_dim <= 0 or output_dim <= 0:
            raise ValueError("condition dimensions must be positive")
        self.content_encoder = content_encoder
        self.speaker_encoder = speaker_encoder
        self.fusion = nn.Sequential(
            nn.Conv1d(content_dim + speaker_dim, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        content_waveforms: torch.Tensor,
        speaker_waveforms: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_lengths: torch.Tensor,
        target_mask: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """构造与目标 Mel 帧对齐的统一条件 ``h``。"""
        target_lengths = target_mask.sum(dim=1)
        target_frames = target_mask.size(1)
        content_features = self.content_encoder(
            content_waveforms,
            content_lengths,
            target_lengths,
            target_frames,
            sample_rate,
        )
        speaker_features = self.speaker_encoder(
            speaker_waveforms,
            speaker_lengths,
            sample_rate,
        )
        expanded_speaker = speaker_features.unsqueeze(2).expand(-1, -1, target_frames)
        condition = self.fusion(torch.cat((content_features, expanded_speaker), dim=1))
        return condition * target_mask.unsqueeze(1).to(condition.dtype)
