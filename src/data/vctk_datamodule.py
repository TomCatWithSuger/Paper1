"""VCTK dataset and Lightning data module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import soundfile as sf
import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class VCTKDataset(Dataset[Dict[str, Any]]):
    """Speaker-aware VCTK speech dataset backed by transcripts and trimmed audio."""

    def __init__(self, data_dir: str | Path, speaker_ids: Sequence[str], microphone: str) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.speaker_ids = list(speaker_ids)
        self.microphone = microphone
        self.samples = self._find_samples()

    def _find_samples(self) -> List[Tuple[str, str, str, Path]]:
        samples: List[Tuple[str, str, str, Path]] = []
        for speaker_id in self.speaker_ids:
            transcript_dir = self.data_dir / "txt" / speaker_id
            audio_dir = self.data_dir / "wav48_silence_trimmed" / speaker_id
            for transcript_path in sorted(transcript_dir.glob("*.txt")):
                utterance_id = transcript_path.stem
                audio_path = audio_dir / f"{utterance_id}_{self.microphone}.flac"
                if not audio_path.is_file():
                    audio_path = audio_dir / f"{utterance_id}_{self.microphone}.wav"
                if not audio_path.is_file():
                    continue
                text = transcript_path.read_text(encoding="utf-8").strip()
                if text:
                    samples.append((utterance_id, speaker_id, text, audio_path))

        if not samples:
            raise ValueError(f"No VCTK samples found for speakers: {self.speaker_ids}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        utterance_id, speaker_id, text, audio_path = self.samples[index]
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
        return {
            "id": utterance_id,
            "speaker_id": speaker_id,
            "text": text,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "num_samples": waveform.numel(),
            "audio_path": str(audio_path),
        }


def vctk_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length VCTK waveforms and preserve speaker metadata."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    sample_rates = {int(sample["sample_rate"]) for sample in samples}
    if len(sample_rates) != 1:
        raise ValueError(f"Mixed sample rates in one batch: {sorted(sample_rates)}")

    waveforms = [sample["waveform"] for sample in samples]
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    attention_mask = torch.arange(padded_waveforms.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "ids": [sample["id"] for sample in samples],
        "speaker_ids": [sample["speaker_id"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
        "waveforms": padded_waveforms,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "sample_rate": sample_rates.pop(),
        "audio_paths": [sample["audio_path"] for sample in samples],
    }


class VCTKDataModule(LightningDataModule):
    """Lightning data module using deterministic speaker-disjoint VCTK splits."""

    def __init__(
        self,
        data_dir: str,
        microphone: str = "mic1",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if microphone not in {"mic1", "mic2"}:
            raise ValueError(f"microphone must be 'mic1' or 'mic2', got {microphone}")
        if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")

        self.save_hyperparameters(logger=False)
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.microphone = microphone
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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
        if missing_paths:
            raise FileNotFoundError("Missing VCTK paths:\n" + "\n".join(missing_paths))

    def _split_speakers(self) -> Tuple[List[str], List[str], List[str]]:
        transcript_root = self.data_dir / "txt"
        audio_root = self.data_dir / "wav48_silence_trimmed"
        transcript_speakers = {
            path.name for path in transcript_root.iterdir() if path.is_dir() and path.name.startswith("p")
        }
        audio_speakers = {
            path.name for path in audio_root.iterdir() if path.is_dir() and path.name.startswith("p")
        }
        speakers = sorted(transcript_speakers & audio_speakers)
        if len(speakers) < 3:
            raise ValueError("VCTK requires at least three speakers for train/val/test splits")

        val_count = max(1, round(len(speakers) * self.val_ratio))
        test_count = max(1, round(len(speakers) * self.test_ratio))
        if val_count + test_count >= len(speakers):
            raise ValueError("VCTK split ratios leave no training speakers")

        train_end = len(speakers) - val_count - test_count
        val_end = len(speakers) - test_count
        return speakers[:train_end], speakers[train_end:val_end], speakers[val_end:]

    def setup(self, stage: Optional[str] = None) -> None:
        """Create speaker-disjoint datasets for the requested stage."""
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        train_speakers, val_speakers, test_speakers = self._split_speakers()
        if stage in (None, "fit", "validate"):
            if self.data_train is None:
                self.data_train = VCTKDataset(self.data_dir, train_speakers, self.microphone)
            if self.data_val is None:
                self.data_val = VCTKDataset(self.data_dir, val_speakers, self.microphone)
        if stage in (None, "test", "predict") and self.data_test is None:
            self.data_test = VCTKDataset(self.data_dir, test_speakers, self.microphone)

    def _dataloader(self, dataset: VCTKDataset, shuffle: bool) -> DataLoader[Dict[str, Any]]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=vctk_collate_fn,
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
