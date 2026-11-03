"""模块说明：

- 负责 VCTK 样本的参考配对、音频裁剪和已有特征缓存读取。
- 对外提供 VCTKDataset；不创建数据划分，不下载资源，不依赖 Lightning。
"""

# ====================
# 1. Imports
# ====================

from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.components.vctk_splits import VCTKSample

# ====================
# 6. Public API
# ====================


class VCTKDataset(Dataset[Dict[str, Any]]):
    """按说话人构造参考配对，并返回波形与可选的缓存特征。"""

    def __init__(
        self,
        samples: Sequence[VCTKSample],
        sample_rate: int,
        max_duration_seconds: float,
        randomize: bool,
        reference_mode: Literal["same_speaker", "different_speaker"] = "same_speaker",
        feature_cache_dir: Path | None = None,
    ) -> None:
        if reference_mode not in {"same_speaker", "different_speaker"}:
            raise ValueError(f"Unsupported reference mode: {reference_mode}")
        self.samples = list(samples)
        self.speaker_ids = sorted({sample.speaker_id for sample in samples})
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_duration_seconds)
        self.randomize = randomize
        self.reference_mode = reference_mode
        self.feature_cache_dir = feature_cache_dir
        self.speaker_sample_indices = self._group_sample_indices()

    def _group_sample_indices(self) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {speaker_id: [] for speaker_id in self.speaker_ids}
        for index, sample in enumerate(self.samples):
            grouped[sample.speaker_id].append(index)
        if self.reference_mode == "same_speaker":
            speakers_without_references = [
                speaker_id for speaker_id, indices in grouped.items() if len(indices) < 2
            ]
        else:
            speakers_without_references = [] if len(grouped) >= 2 else self.speaker_ids
        if speakers_without_references:
            raise ValueError(
                "VCTK reference pairing is unavailable for speakers: "
                + ", ".join(speakers_without_references)
            )
        return grouped

    def _reference_index(self, index: int, speaker_id: str) -> int:
        if self.reference_mode == "different_speaker":
            indices = [
                candidate_index
                for candidate_speaker, candidate_indices in self.speaker_sample_indices.items()
                if candidate_speaker != speaker_id
                for candidate_index in candidate_indices
            ]
            if self.randomize:
                return indices[int(torch.randint(len(indices), ()).item())]
            return indices[index % len(indices)]
        indices = self.speaker_sample_indices[speaker_id]
        position = indices.index(index)
        if self.randomize:
            offset = int(torch.randint(1, len(indices), ()).item())
            return indices[(position + offset) % len(indices)]
        return indices[(position + 1) % len(indices)]

    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int, int]:
        result: tuple[Any, int] = sf.read(audio_path, dtype="float32", always_2d=True)
        audio, original_sample_rate = result
        waveform = torch.from_numpy(audio).mean(dim=1)
        if original_sample_rate != self.sample_rate:
            output_length = round(waveform.numel() * self.sample_rate / original_sample_rate)
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=output_length,
                mode="linear",
                align_corners=False,
            )
            waveform = waveform.view(-1)
        total_samples = waveform.numel()
        start = 0
        if total_samples > self.max_samples:
            if self.randomize:
                max_start = total_samples - self.max_samples
                start = int(torch.randint(max_start + 1, ()).item())
            waveform = waveform[start : start + self.max_samples]
        return waveform.contiguous(), start, total_samples

    def _load_content_features(
        self, utterance_id: str, crop_start: int, crop_samples: int, total_samples: int
    ) -> torch.Tensor:
        if self.feature_cache_dir is None:
            raise RuntimeError("feature cache directory is not configured")
        path = self.feature_cache_dir / "content" / f"{utterance_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"cached content feature not found: {path}")
        features = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError(f"invalid cached content feature: {path}")
        start_frame = round(crop_start * features.size(1) / total_samples)
        end_frame = round((crop_start + crop_samples) * features.size(1) / total_samples)
        end_frame = max(start_frame + 1, min(end_frame, features.size(1)))
        return features[:, start_frame:end_frame].float().contiguous()

    def _load_speaker_features(self, utterance_id: str) -> torch.Tensor:
        if self.feature_cache_dir is None:
            raise RuntimeError("feature cache directory is not configured")
        path = self.feature_cache_dir / "speaker" / f"{utterance_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"cached speaker feature not found: {path}")
        features = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor) or features.ndim != 1:
            raise ValueError(f"invalid cached speaker feature: {path}")
        return features.float().contiguous()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        reference_index = self._reference_index(index, sample.speaker_id)
        reference = self.samples[reference_index]
        waveform, crop_start, total_samples = self._load_audio(sample.audio_path)
        item = {
            "id": sample.utterance_id,
            "sentence_id": sample.sentence_id,
            "speaker_id": sample.speaker_id,
            "text": sample.text,
            "waveform": waveform,
            "reference_id": reference.utterance_id,
            "reference_sentence_id": reference.sentence_id,
            "reference_speaker_id": reference.speaker_id,
            "reference_text": reference.text,
            "sample_rate": self.sample_rate,
            "audio_path": str(sample.audio_path),
        }
        if self.feature_cache_dir is not None:
            item.update(
                {
                    "content_features": self._load_content_features(
                        sample.utterance_id, crop_start, waveform.numel(), total_samples
                    ),
                    "speaker_features": self._load_speaker_features(sample.utterance_id),
                    "reference_speaker_features": self._load_speaker_features(
                        reference.utterance_id
                    ),
                }
            )
        return item
