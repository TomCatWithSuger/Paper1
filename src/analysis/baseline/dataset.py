"""核验数据划分并构建固定测试清单。"""

# ====================
# 1. Imports
# ====================


import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from src.analysis.baseline.runs import RunSummary, _nested_value
from src.data.components.vctk_splits import (
    UnseenSpeakerSentenceSplit,
    discover_vctk_samples,
)
from src.data.vctk_datamodule import VCTKDataset

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _resolve_vctk_root(data_root: Path) -> Path:
    candidates = (data_root, data_root / "VCTK")
    for candidate in candidates:
        if (candidate / "txt").is_dir() and (candidate / "wav48_silence_trimmed").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"未找到 VCTK 目录；已检查: {', '.join(str(path) for path in candidates)}"
    )


def _shared_split_config(runs: Sequence[RunSummary]) -> dict[str, Any]:
    keys = (
        "evaluation_speaker_ids",
        "evaluation_sentence_ids",
        "val_ratio",
        "seed",
    )
    values = {
        key: [_nested_value(run.config, f"data.split_strategy.{key}") for run in runs]
        for key in keys
    }
    mismatched = [
        key for key, items in values.items() if any(item != items[0] for item in items[1:])
    ]
    if mismatched:
        raise ValueError(f"B1/B2/B3 数据划分配置不一致: {', '.join(mismatched)}")
    return {key: items[0] for key, items in values.items()}


def _build_test_manifest(
    vctk_root: Path,
    runs: Sequence[RunSummary],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_config = _shared_split_config(runs)
    microphone = _nested_value(runs[0].config, "data.microphone")
    samples = discover_vctk_samples(vctk_root, str(microphone))
    strategy = UnseenSpeakerSentenceSplit(**split_config)
    split = strategy.split(samples)
    dataset = VCTKDataset(
        samples=split.test,
        sample_rate=int(_nested_value(runs[0].config, "data.sample_rate")),
        max_duration_seconds=float(_nested_value(runs[0].config, "data.max_duration_seconds")),
        randomize=False,
        reference_mode="different_speaker",
    )

    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for index, source in enumerate(dataset.samples):
        reference = dataset.samples[dataset._reference_index(index, source.speaker_id)]
        relative_audio = source.audio_path.relative_to(vctk_root)
        row = {
            "index": index,
            "source_id": source.utterance_id,
            "source_speaker_id": source.speaker_id,
            "sentence_id": source.sentence_id,
            "reference_id": reference.utterance_id,
            "reference_speaker_id": reference.speaker_id,
            "text": source.text,
            "audio_path": str(relative_audio),
        }
        rows.append(row)
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")

    dataset_summary = {
        "data_root_argument": str(vctk_root.parent),
        "resolved_vctk_root": str(vctk_root),
        "discovered_samples": len(samples),
        "train_samples": len(split.train),
        "validation_samples": len(split.val),
        "test_samples": len(split.test),
        "test_speakers": len({sample.speaker_id for sample in split.test}),
        "test_sentences": len({sample.sentence_id for sample in split.test}),
        "test_manifest_sha256": digest.hexdigest(),
        "metadata_modified": False,
    }
    return rows, dataset_summary
