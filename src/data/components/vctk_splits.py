"""VCTK sample discovery and pluggable split strategies."""

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Protocol, Sequence


@dataclass(frozen=True)
class VCTKSample:
    utterance_id: str
    sentence_id: str
    speaker_id: str
    text: str
    audio_path: Path


@dataclass(frozen=True)
class VCTKSplit:
    train: tuple[VCTKSample, ...]
    val: tuple[VCTKSample, ...]
    test: tuple[VCTKSample, ...]


class VCTKSplitStrategy(Protocol):
    def split(self, samples: Sequence[VCTKSample]) -> VCTKSplit: ...


def discover_vctk_samples(data_dir: Path, microphone: str) -> list[VCTKSample]:
    samples: list[VCTKSample] = []
    transcript_root = data_dir / "txt"
    audio_root = data_dir / "wav48_silence_trimmed"
    speaker_ids = sorted(
        path.name
        for path in transcript_root.iterdir()
        if path.is_dir() and (audio_root / path.name).is_dir() and path.name.startswith("p")
    )

    for speaker_id in speaker_ids:
        for transcript_path in sorted((transcript_root / speaker_id).glob("*.txt")):
            utterance_id = transcript_path.stem
            audio_path = audio_root / speaker_id / f"{utterance_id}_{microphone}.flac"
            if not audio_path.is_file():
                audio_path = audio_path.with_suffix(".wav")
            if not audio_path.is_file():
                continue
            text = transcript_path.read_text(encoding="utf-8").strip()
            if text:
                samples.append(
                    VCTKSample(
                        utterance_id=utterance_id,
                        sentence_id=utterance_id.rsplit("_", maxsplit=1)[-1],
                        speaker_id=speaker_id,
                        text=text,
                        audio_path=audio_path,
                    )
                )

    if not samples:
        raise ValueError(f"No VCTK samples found in {data_dir}")
    return samples


class RatioSpeakerSplit:
    """Split sorted speakers into disjoint train, validation, and test sets."""

    def __init__(self, val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
        if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
            raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, samples: Sequence[VCTKSample]) -> VCTKSplit:
        speakers = sorted({sample.speaker_id for sample in samples})
        if len(speakers) < 3:
            raise ValueError("VCTK requires at least three speakers for train/val/test splits")

        val_count = max(1, round(len(speakers) * self.val_ratio))
        test_count = max(1, round(len(speakers) * self.test_ratio))
        if val_count + test_count >= len(speakers):
            raise ValueError("VCTK split ratios leave no training speakers")

        train_end = len(speakers) - val_count - test_count
        val_end = len(speakers) - test_count
        train_speakers = set(speakers[:train_end])
        val_speakers = set(speakers[train_end:val_end])
        test_speakers = set(speakers[val_end:])
        return VCTKSplit(
            train=tuple(sample for sample in samples if sample.speaker_id in train_speakers),
            val=tuple(sample for sample in samples if sample.speaker_id in val_speakers),
            test=tuple(sample for sample in samples if sample.speaker_id in test_speakers),
        )


class UnseenSpeakerSentenceSplit:
    """Reserve specified speakers and sentence IDs while holding out seen-speaker utterances."""

    def __init__(
        self,
        evaluation_speaker_ids: Sequence[str],
        evaluation_sentence_ids: Sequence[str],
        val_ratio: float = 0.1,
        seed: int = 12345,
    ) -> None:
        if not evaluation_speaker_ids or not evaluation_sentence_ids:
            raise ValueError("Evaluation speakers and sentences must not be empty")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        self.evaluation_speaker_ids = tuple(evaluation_speaker_ids)
        self.evaluation_sentence_ids = tuple(
            sentence_id.rsplit("_", maxsplit=1)[-1] for sentence_id in evaluation_sentence_ids
        )
        self.val_ratio = val_ratio
        self.seed = seed

    def split(self, samples: Sequence[VCTKSample]) -> VCTKSplit:
        available_speakers = {sample.speaker_id for sample in samples}
        available_sentences = {sample.sentence_id for sample in samples}
        evaluation_speakers = set(self.evaluation_speaker_ids)
        evaluation_sentences = set(self.evaluation_sentence_ids)

        missing_speakers = sorted(evaluation_speakers - available_speakers)
        missing_sentences = sorted(evaluation_sentences - available_sentences)
        if missing_speakers:
            raise ValueError(f"Unknown evaluation speakers: {missing_speakers}")
        if missing_sentences:
            raise ValueError(f"Unknown evaluation sentences: {missing_sentences}")

        test = tuple(
            sample
            for sample in samples
            if sample.speaker_id in evaluation_speakers
            and sample.sentence_id in evaluation_sentences
        )
        test_pairs = {(sample.speaker_id, sample.sentence_id) for sample in test}
        missing_test_pairs = sorted(
            (speaker_id, sentence_id)
            for speaker_id in evaluation_speakers
            for sentence_id in evaluation_sentences
            if (speaker_id, sentence_id) not in test_pairs
        )
        if missing_test_pairs:
            formatted_pairs = ", ".join(
                f"{speaker_id}_{sentence_id}"
                for speaker_id, sentence_id in missing_test_pairs
            )
            raise ValueError(f"Missing evaluation utterances: {formatted_pairs}")

        seen_samples = [
            sample
            for sample in samples
            if sample.speaker_id not in evaluation_speakers
            and sample.sentence_id not in evaluation_sentences
        ]

        grouped: dict[str, list[VCTKSample]] = {}
        for sample in seen_samples:
            grouped.setdefault(sample.speaker_id, []).append(sample)

        train: list[VCTKSample] = []
        val: list[VCTKSample] = []
        for speaker_id, speaker_samples in sorted(grouped.items()):
            ordered = sorted(speaker_samples, key=lambda sample: sample.utterance_id)
            Random(f"{self.seed}:{speaker_id}").shuffle(ordered)
            val_count = max(2, round(len(ordered) * self.val_ratio))
            if len(ordered) - val_count < 2:
                raise ValueError(
                    f"Speaker {speaker_id} needs at least four non-evaluation utterances"
                )
            val.extend(ordered[:val_count])
            train.extend(ordered[val_count:])

        test_counts = {
            speaker_id: sum(sample.speaker_id == speaker_id for sample in test)
            for speaker_id in evaluation_speakers
        }
        insufficient_test_speakers = sorted(
            speaker_id for speaker_id, count in test_counts.items() if count < 2
        )
        if insufficient_test_speakers:
            raise ValueError(
                "Evaluation speakers need at least two selected sentences: "
                + ", ".join(insufficient_test_speakers)
            )

        return VCTKSplit(train=tuple(train), val=tuple(val), test=test)
