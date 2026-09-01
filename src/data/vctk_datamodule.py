"""VCTK dataset and Lightning data module."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.components.vctk_splits import (
    RatioSpeakerSplit,
    VCTKSample,
    VCTKSplitStrategy,
    discover_vctk_samples,
)


class VCTKDataset(Dataset[Dict[str, Any]]):
    """Speaker-aware VCTK dataset with configurable reference pairing."""

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
        result: tuple[Any, int] = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
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
        self,
        utterance_id: str,
        crop_start: int,
        crop_samples: int,
        total_samples: int,
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
                        sample.utterance_id,
                        crop_start,
                        waveform.numel(),
                        total_samples,
                    ),
                    "speaker_features": self._load_speaker_features(sample.utterance_id),
                    "reference_speaker_features": self._load_speaker_features(
                        reference.utterance_id
                    ),
                }
            )
        return item


class VCTKCollator:
    """Create padded waveform and log-Mel batches for voice conversion."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        log_mel_floor: float,
    ) -> None:
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length)
        self.mel_filterbank = self._create_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.log_mel_floor = log_mel_floor

    @staticmethod
    def _create_mel_filterbank(
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> torch.Tensor:
        def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
            return 2595.0 * torch.log10(1.0 + frequency / 700.0)

        def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
            return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(torch.tensor(f_min))
        mel_max = hz_to_mel(torch.tensor(f_max))
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        frequency_points = mel_to_hz(mel_points)
        fft_frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)

        lower = frequency_points[:-2].unsqueeze(1)
        center = frequency_points[1:-1].unsqueeze(1)
        upper = frequency_points[2:].unsqueeze(1)
        ascending = (fft_frequencies.unsqueeze(0) - lower) / (center - lower)
        descending = (upper - fft_frequencies.unsqueeze(0)) / (upper - center)
        return torch.minimum(ascending, descending).clamp_min(0.0)

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        mel = self.mel_filterbank @ spectrum.abs().pow(2)
        return torch.log(mel.clamp_min(self.log_mel_floor))

    @staticmethod
    def _pad_waveforms(
        waveforms: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
        padded = pad_sequence(waveforms, batch_first=True)
        mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, lengths, mask

    @staticmethod
    def _pad_mels(
        mels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([mel.size(1) for mel in mels], dtype=torch.long)
        padded = pad_sequence([mel.transpose(0, 1) for mel in mels], batch_first=True)
        padded = padded.transpose(1, 2)
        mask = torch.arange(padded.size(2)).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, lengths, mask

    @staticmethod
    def _pad_content_features(
        features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([feature.size(1) for feature in features], dtype=torch.long)
        padded = pad_sequence([feature.transpose(0, 1) for feature in features], batch_first=True)
        return padded.transpose(1, 2), lengths

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")

        sample_rates = {int(sample["sample_rate"]) for sample in samples}
        if len(sample_rates) != 1:
            raise ValueError(f"Mixed sample rates in one batch: {sorted(sample_rates)}")

        waveforms = [sample["waveform"] for sample in samples]
        mels = [self._log_mel(waveform) for waveform in waveforms]

        padded_waveforms, lengths, attention_mask = self._pad_waveforms(waveforms)
        padded_mels, mel_lengths, mel_attention_mask = self._pad_mels(mels)

        batch = {
            "ids": [sample["id"] for sample in samples],
            "sentence_ids": [sample["sentence_id"] for sample in samples],
            "speaker_ids": [sample["speaker_id"] for sample in samples],
            "texts": [sample["text"] for sample in samples],
            "waveforms": padded_waveforms,
            "lengths": lengths,
            "attention_mask": attention_mask,
            "mel_spectrograms": padded_mels,
            "mel_lengths": mel_lengths,
            "mel_attention_mask": mel_attention_mask,
            "reference_ids": [sample["reference_id"] for sample in samples],
            "reference_sentence_ids": [sample["reference_sentence_id"] for sample in samples],
            "reference_speaker_ids": [sample["reference_speaker_id"] for sample in samples],
            "reference_texts": [sample["reference_text"] for sample in samples],
            "sample_rate": sample_rates.pop(),
            "audio_paths": [sample["audio_path"] for sample in samples],
        }
        cached_samples = ["content_features" in sample for sample in samples]
        if any(cached_samples) and not all(cached_samples):
            raise ValueError("mixed cached and uncached VCTK samples in one batch")
        if all(cached_samples):
            content_features, content_feature_lengths = self._pad_content_features(
                [sample["content_features"] for sample in samples]
            )
            batch.update(
                {
                    "content_features": content_features,
                    "content_feature_lengths": content_feature_lengths,
                    "speaker_features": torch.stack(
                        [sample["speaker_features"] for sample in samples]
                    ),
                    "reference_speaker_features": torch.stack(
                        [sample["reference_speaker_features"] for sample in samples]
                    ),
                }
            )
        return batch


class VCTKDataModule(LightningDataModule):
    """Lightning data module using a configurable VCTK split strategy."""

    def __init__(
        self,
        data_dir: str,
        microphone: str = "mic1",
        split_strategy: Optional[VCTKSplitStrategy] = None,
        test_reference_mode: Literal["same_speaker", "different_speaker"] = "same_speaker",
        sample_rate: int = 22_050,
        max_duration_seconds: float = 10.0,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8_000.0,
        log_mel_floor: float = 1e-5,
        feature_cache_dir: str | None = None,
        require_feature_cache: bool = False,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if microphone not in {"mic1", "mic2"}:
            raise ValueError(f"microphone must be 'mic1' or 'mic2', got {microphone}")
        if test_reference_mode not in {"same_speaker", "different_speaker"}:
            raise ValueError(f"Unsupported test reference mode: {test_reference_mode}")
        if sample_rate <= 0 or max_duration_seconds <= 0:
            raise ValueError("sample_rate and max_duration_seconds must be positive")
        if n_fft <= 0 or win_length <= 0 or hop_length <= 0 or n_mels <= 0:
            raise ValueError("Mel spectrogram dimensions must be positive")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if not 0 <= f_min < f_max <= sample_rate / 2:
            raise ValueError("Mel frequencies must satisfy 0 <= f_min < f_max <= Nyquist")
        if log_mel_floor <= 0:
            raise ValueError("log_mel_floor must be positive")

        self.save_hyperparameters(logger=False)
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.microphone = microphone
        self.split_strategy = split_strategy or RatioSpeakerSplit()
        self.test_reference_mode: Literal["same_speaker", "different_speaker"] = (
            test_reference_mode
        )
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.feature_cache_dir = (
            Path(feature_cache_dir).expanduser().resolve() if feature_cache_dir else None
        )
        self.require_feature_cache = require_feature_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collator = VCTKCollator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            log_mel_floor=log_mel_floor,
        )

        self.data_train: Optional[VCTKDataset] = None
        self.data_val: Optional[VCTKDataset] = None
        self.data_test: Optional[VCTKDataset] = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Validate the expected VCTK directory structure."""
        required_paths = (
            self.data_dir / "txt",
            self.data_dir / "wav48_silence_trimmed",
        )
        missing_paths = [str(path) for path in required_paths if not path.is_dir()]
        if self.require_feature_cache:
            if self.feature_cache_dir is None:
                missing_paths.append("feature_cache_dir is not configured")
            else:
                missing_paths.extend(
                    str(path)
                    for path in (
                        self.feature_cache_dir / "content",
                        self.feature_cache_dir / "speaker",
                    )
                    if not path.is_dir()
                )
        if missing_paths:
            raise FileNotFoundError("Missing VCTK paths:\n" + "\n".join(missing_paths))

    def _dataset(
        self,
        samples: Sequence[VCTKSample],
        randomize: bool,
        reference_mode: Literal["same_speaker", "different_speaker"] = "same_speaker",
    ) -> VCTKDataset:
        return VCTKDataset(
            samples=samples,
            sample_rate=self.sample_rate,
            max_duration_seconds=self.max_duration_seconds,
            randomize=randomize,
            reference_mode=reference_mode,
            feature_cache_dir=self.feature_cache_dir,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets using the configured split strategy."""
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        samples = discover_vctk_samples(self.data_dir, self.microphone)
        split = self.split_strategy.split(samples)
        if stage in (None, "fit", "validate"):
            if self.data_train is None:
                self.data_train = self._dataset(split.train, randomize=True)
            if self.data_val is None:
                self.data_val = self._dataset(split.val, randomize=False)
        if stage in (None, "test", "predict") and self.data_test is None:
            self.data_test = self._dataset(
                split.test,
                randomize=False,
                reference_mode=self.test_reference_mode,
            )

    def _dataloader(self, dataset: VCTKDataset, shuffle: bool) -> DataLoader[Dict[str, Any]]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return the training dataloader."""
        if self.data_train is None:
            raise RuntimeError("Training dataset is not initialized. Call setup('fit') first.")
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return the validation dataloader."""
        if self.data_val is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup('fit') first.")
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return the test dataloader."""
        if self.data_test is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test') first.")
        return self._dataloader(self.data_test, shuffle=False)

    def predict_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return the prediction dataloader."""
        if self.data_test is None:
            raise RuntimeError(
                "Prediction dataset is not initialized. Call setup('predict') first."
            )
        return self._dataloader(self.data_test, shuffle=False)
