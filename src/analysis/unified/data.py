"""准备固定转换配对、缓存特征与 Mel 批次。"""

# ====================
# 1. Imports
# ====================


from pathlib import Path
from typing import Any, Sequence

import torch

from src.analysis.unified.checkpoints import EvaluationSpec
from src.analysis.unified.records import _read_csv
from src.data.components.vctk_splits import VCTKSample, discover_vctk_samples
from src.data.vctk_datamodule import VCTKCollator, VCTKDataset

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _load_specs(plan_path: Path) -> list[EvaluationSpec]:
    specs = []
    for row in _read_csv(plan_path):
        if row["status"] != "待运行统一转换评估":
            continue
        run_dir = Path(row["run_dir"]).expanduser().resolve()
        specs.append(
            EvaluationSpec(
                experiment_id=row["experiment_id"],
                model=row["model"],
                nfe=int(row["nfe"]),
                run_dir=run_dir,
            )
        )
    if not specs:
        raise ValueError(f"实验计划中没有待评估项目: {plan_path}")
    return specs


def _load_manifest_samples(
    manifest_path: Path,
    vctk_root: Path,
    microphone: str,
) -> tuple[list[dict[str, str]], dict[str, VCTKSample]]:
    rows = _read_csv(manifest_path)
    samples = discover_vctk_samples(vctk_root, microphone)
    by_id = {sample.utterance_id: sample for sample in samples}
    missing = sorted(
        utterance_id
        for row in rows
        for utterance_id in (row["source_id"], row["reference_id"])
        if utterance_id not in by_id
    )
    if missing:
        raise FileNotFoundError(f"测试清单样本不存在: {', '.join(sorted(set(missing)))}")
    return rows, by_id


def _validate_feature_cache(manifest: Sequence[dict[str, str]], feature_cache_dir: Path) -> None:
    utterance_ids = {
        utterance_id
        for row in manifest
        for utterance_id in (row["source_id"], row["reference_id"])
    }
    missing = [
        f"{kind}/{utterance_id}.pt"
        for kind in ("content", "speaker")
        for utterance_id in sorted(utterance_ids)
        if not (feature_cache_dir / kind / f"{utterance_id}.pt").is_file()
    ]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        remainder = "" if len(missing) <= 20 else f"\n- 其余 {len(missing) - 20} 个"
        raise FileNotFoundError(f"统一评估缺少 {len(missing)} 个缓存文件:\n{preview}{remainder}")


def _collator(config: dict[str, Any]) -> VCTKCollator:
    data = config["data"]
    return VCTKCollator(
        sample_rate=int(data["sample_rate"]),
        n_fft=int(data["n_fft"]),
        win_length=int(data["win_length"]),
        hop_length=int(data["hop_length"]),
        n_mels=int(data["n_mels"]),
        f_min=float(data["f_min"]),
        f_max=float(data["f_max"]),
        log_mel_floor=float(data["log_mel_floor"]),
    )


def _dataset(
    samples: Sequence[VCTKSample],
    config: dict[str, Any],
    feature_cache_dir: Path,
) -> VCTKDataset:
    data = config["data"]
    return VCTKDataset(
        samples=samples,
        sample_rate=int(data["sample_rate"]),
        max_duration_seconds=float(data["max_duration_seconds"]),
        randomize=False,
        reference_mode="different_speaker",
        feature_cache_dir=feature_cache_dir,
    )


def _prepare_pair(
    row: dict[str, str],
    by_id: dict[str, VCTKSample],
    config: dict[str, Any],
    feature_cache_dir: Path,
    collator: VCTKCollator,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    source = by_id[row["source_id"]]
    reference = by_id[row["reference_id"]]
    pair_dataset = _dataset((source, reference), config, feature_cache_dir)
    source_item = pair_dataset[0]
    if source_item["reference_id"] != reference.utterance_id:
        raise ValueError(f"固定参考不匹配: {source.utterance_id} -> {source_item['reference_id']}")
    reference_item = pair_dataset[1]
    source_batch = collator([source_item])
    reference_batch = collator([reference_item])
    target_frames = int(reference_batch["mel_lengths"][0].item())
    target_mel = reference_batch["mel_spectrograms"][0, :, :target_frames]
    audio_samples = int(source_batch["lengths"][0].item())
    return source_batch, target_mel, audio_samples
