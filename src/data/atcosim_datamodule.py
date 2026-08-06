from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import wave

import torch
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ATCOSIMDataset(Dataset[Dict[str, Any]]):
    """ATCOSIM speech recognition dataset backed by a transcription file and WAV directory."""

    def __init__(self, transcription_file: str | Path, audio_dir: str | Path) -> None:
        self.transcription_file = Path(transcription_file).expanduser().resolve()
        self.audio_dir = Path(audio_dir).expanduser().resolve()

        if not self.transcription_file.is_file():
            raise FileNotFoundError(f"Transcription file not found: {self.transcription_file}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.samples = self._read_transcriptions()

    def _read_transcriptions(self) -> List[Tuple[str, str, Path]]:
        samples: List[Tuple[str, str, Path]] = []
        with self.transcription_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid transcription at {self.transcription_file}:{line_number}"
                    )

                utterance_id, text = parts
                audio_path = self.audio_dir / f"{utterance_id}.wav"
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio for '{utterance_id}' not found: {audio_path}"
                    )
                samples.append((utterance_id, text, audio_path))

        if not samples:
            raise ValueError(f"No samples found in {self.transcription_file}")
        return samples

    @staticmethod
    def _load_wav(audio_path: Path) -> Tuple[torch.Tensor, int]:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(wav_file.getnframes())

        if compression != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {audio_path}")
        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio: {audio_path}"
            )

        pcm = array("h")
        pcm.frombytes(frames)
        if not pcm:
            raise ValueError(f"Empty WAV file: {audio_path}")

        waveform = torch.tensor(pcm, dtype=torch.float32).reshape(-1, channels)
        waveform = waveform.mean(dim=1) / 32768.0
        return waveform, sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        utterance_id, text, audio_path = self.samples[index]
        waveform, sample_rate = self._load_wav(audio_path)
        return {
            "id": utterance_id,
            "text": text,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "num_samples": waveform.numel(),
            "audio_path": str(audio_path),
        }


def atcosim_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length waveforms and preserve transcription metadata."""
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
        "texts": [sample["text"] for sample in samples],
        "waveforms": padded_waveforms,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "sample_rate": sample_rates.pop(),
        "audio_paths": [sample["audio_path"] for sample in samples],
    }


class ATCOSIMDataModule(LightningDataModule):
    """Lightning data module for the ATCOSIM train, validation fold, and test splits."""

    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        if fold not in range(5):
            raise ValueError(f"fold must be between 0 and 4, got {fold}")

        self.save_hyperparameters(logger=False)
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[ATCOSIMDataset] = None
        self.data_val: Optional[ATCOSIMDataset] = None
        self.data_test: Optional[ATCOSIMDataset] = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        required_paths = (
            self.data_dir / "train",
            self.data_dir / "test",
            self.data_dir / "transcriptions" / "train_trans.txt",
            self.data_dir / "transcriptions" / "test_trans.txt",
            self.data_dir / "transcriptions" / f"fold{self.fold}" / "train_trans.txt",
            self.data_dir / "transcriptions" / f"fold{self.fold}" / "val_trans.txt",
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError("Missing ATCOSIM paths:\n" + "\n".join(missing_paths))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of "
                    f"devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        fold_dir = self.data_dir / "transcriptions" / f"fold{self.fold}"
        if stage in (None, "fit", "validate"):
            if self.data_train is None:
                self.data_train = ATCOSIMDataset(
                    transcription_file=fold_dir / "train_trans.txt",
                    audio_dir=self.data_dir / "train",
                )
            if self.data_val is None:
                self.data_val = ATCOSIMDataset(
                    transcription_file=fold_dir / "val_trans.txt",
                    audio_dir=self.data_dir / "train",
                )

        if stage in (None, "test", "predict") and self.data_test is None:
            self.data_test = ATCOSIMDataset(
                transcription_file=self.data_dir / "transcriptions" / "test_trans.txt",
                audio_dir=self.data_dir / "test",
            )

    def _dataloader(self, dataset: ATCOSIMDataset, shuffle: bool) -> DataLoader[Dict[str, Any]]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            collate_fn=atcosim_collate_fn,
        )

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.data_train is None:
            raise RuntimeError("Training dataset is not initialized. Call setup('fit') first.")
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.data_val is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup('fit') first.")
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.data_test is None:
            raise RuntimeError("Test dataset is not initialized. Call setup('test') first.")
        return self._dataloader(self.data_test, shuffle=False)

    def predict_dataloader(self) -> DataLoader[Dict[str, Any]]:
        if self.data_test is None:
            raise RuntimeError("Prediction dataset is not initialized. Call setup('predict') first.")
        return self._dataloader(self.data_test, shuffle=False)
