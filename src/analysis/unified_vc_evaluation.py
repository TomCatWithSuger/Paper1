"""在固定 VCTK 清单上统一评估 B1、B2、B3 的 Mel 质量与推理效率。"""

# ====================
# 1. Imports
# ====================

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import hydra
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from src.data.components.vctk_splits import (
    UnseenSpeakerSentenceSplit,
    VCTKSample,
    discover_vctk_samples,
)
from src.data.vctk_datamodule import VCTKCollator, VCTKDataset

# ====================
# 2. Definitions
# ====================


@dataclass(frozen=True)
class EvaluationSpec:
    """定义一个 checkpoint 与 NFE 组合。"""

    experiment_id: str
    model: str
    nfe: int
    run_dir: Path


PER_SAMPLE_FIELDS = (
    "experiment_id",
    "model",
    "nfe",
    "sample_index",
    "source_id",
    "source_speaker_id",
    "reference_id",
    "reference_speaker_id",
    "sentence_id",
    "text_length",
    "source_frames",
    "target_frames",
    "audio_seconds",
    "seed",
    "mel_l1",
    "mel_l2_rmse",
    "mel_mse",
    "latency_ms",
    "rtf",
    "generated_path",
)


# ====================
# 3. Utilities
# ====================


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _align_target(target: torch.Tensor, frames: int) -> torch.Tensor:
    return F.interpolate(
        target.unsqueeze(0),
        size=frames,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def _mean_or_nan(values: Iterable[float]) -> float:
    items = list(values)
    return mean(items) if items else math.nan


# ====================
# 4. Internal Functions
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


def _model_config(run_dir: Path) -> dict[str, Any]:
    value = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效 Hydra 配置: {run_dir}")
    return value


def _best_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        path for path in (run_dir / "checkpoints").glob("*.ckpt") if path.name != "last.ckpt"
    )
    if len(checkpoints) != 1:
        raise ValueError(f"期望唯一最佳 checkpoint，实际找到 {len(checkpoints)} 个: {run_dir}")
    return checkpoints[0]


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = _model_config(run_dir)
    model = hydra.utils.instantiate(OmegaConf.create(config["model"]))
    with torch.serialization.safe_globals([UnseenSpeakerSentenceSplit]):
        checkpoint = torch.load(_best_checkpoint(run_dir), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"无效 Lightning checkpoint: {run_dir}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    return model, config


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


def _generate(
    model: torch.nn.Module,
    source_batch: dict[str, Any],
    nfe: int,
) -> torch.Tensor:
    method = model.method
    target_mask = source_batch["mel_attention_mask"]
    condition = method.encode_condition(
        source_batch["content_features"],
        source_batch["content_feature_lengths"],
        source_batch["reference_speaker_features"],
        target_mask,
    )
    kwargs: dict[str, Any] = {"integration_steps": nfe}
    if method.requires_source_mels:
        kwargs["source_mels"] = source_batch["mel_spectrograms"]
    return method.sample(condition, target_mask, **kwargs)


def _warm_up(
    model: torch.nn.Module,
    source_batch: dict[str, Any],
    nfe: int,
    seed: int,
    device: torch.device,
) -> None:
    source_batch = _to_device(source_batch, device)
    _seed(seed, device)
    with torch.inference_mode():
        _generate(model, source_batch, nfe)
    _synchronize(device)


def _evaluate_sample(
    model: torch.nn.Module,
    source_batch: dict[str, Any],
    target_mel: torch.Tensor,
    audio_samples: int,
    sample_rate: int,
    spec: EvaluationSpec,
    row: dict[str, str],
    seed: int,
    device: torch.device,
    generated_dir: Path,
) -> dict[str, Any]:
    source_batch = _to_device(source_batch, device)
    target_mel = target_mel.to(device)
    _seed(seed, device)
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = _generate(model, source_batch, spec.nfe)
    _synchronize(device)
    latency_seconds = time.perf_counter() - started

    source_frames = int(source_batch["mel_lengths"][0].item())
    generated = generated[0, :, :source_frames]
    aligned_target = _align_target(target_mel, source_frames)
    difference = generated - aligned_target
    mel_l1 = float(difference.abs().mean().item())
    mel_mse = float(difference.square().mean().item())
    mel_l2_rmse = math.sqrt(mel_mse)
    audio_seconds = audio_samples / sample_rate

    experiment_dir = generated_dir / spec.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    generated_path = experiment_dir / f"{row['source_id']}__{row['reference_id']}.pt"
    torch.save(
        {
            "mel": generated.detach().to(device="cpu", dtype=torch.float16),
            "source_id": row["source_id"],
            "reference_id": row["reference_id"],
            "sentence_id": row["sentence_id"],
            "seed": seed,
            "nfe": spec.nfe,
        },
        generated_path,
    )
    return {
        "experiment_id": spec.experiment_id,
        "model": spec.model,
        "nfe": spec.nfe,
        "sample_index": int(row["index"]),
        "source_id": row["source_id"],
        "source_speaker_id": row["source_speaker_id"],
        "reference_id": row["reference_id"],
        "reference_speaker_id": row["reference_speaker_id"],
        "sentence_id": row["sentence_id"],
        "text_length": len(row["text"]),
        "source_frames": source_frames,
        "target_frames": int(target_mel.size(1)),
        "audio_seconds": audio_seconds,
        "seed": seed,
        "mel_l1": mel_l1,
        "mel_l2_rmse": mel_l2_rmse,
        "mel_mse": mel_mse,
        "latency_ms": latency_seconds * 1000.0,
        "rtf": latency_seconds / audio_seconds,
        "generated_path": str(generated_path),
    }


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _append_ledger(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _aggregate(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["experiment_id"]), []).append(record)
    rows = []
    for experiment_id, items in sorted(grouped.items()):
        first = items[0]
        rows.append(
            {
                "experiment_id": experiment_id,
                "model": first["model"],
                "nfe": first["nfe"],
                "sample_count": len(items),
                "mel_l1_mean": _mean_or_nan(float(item["mel_l1"]) for item in items),
                "mel_l1_median": median(float(item["mel_l1"]) for item in items),
                "mel_l2_rmse_mean": _mean_or_nan(float(item["mel_l2_rmse"]) for item in items),
                "latency_ms_mean": _mean_or_nan(float(item["latency_ms"]) for item in items),
                "rtf_mean": _mean_or_nan(float(item["rtf"]) for item in items),
            }
        )
    return rows


def _difference_rows(
    records: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    complete_b1 = [
        row for row in aggregates if row["model"] == "B1" and int(row["sample_count"]) == 100
    ]
    b2_ids = {
        str(row["experiment_id"])
        for row in aggregates
        if row["model"] == "B2" and int(row["sample_count"]) == 100
    }
    if not complete_b1 or not b2_ids:
        return [], None
    best_b1 = min(complete_b1, key=lambda row: float(row["mel_l1_mean"]))
    best_id = str(best_b1["experiment_id"])
    b2_id = sorted(b2_ids)[0]
    indexed = {
        (str(record["experiment_id"]), str(record["source_id"])): record for record in records
    }
    rows = []
    for source_id in sorted(
        source
        for experiment_id, source in indexed
        if experiment_id == best_id and (b2_id, source) in indexed
    ):
        b1 = indexed[(best_id, source_id)]
        b2 = indexed[(b2_id, source_id)]
        delta_l1 = float(b2["mel_l1"]) - float(b1["mel_l1"])
        delta_l2 = float(b2["mel_l2_rmse"]) - float(b1["mel_l2_rmse"])
        rows.append(
            {
                "source_id": source_id,
                "source_speaker_id": b1["source_speaker_id"],
                "reference_id": b1["reference_id"],
                "reference_speaker_id": b1["reference_speaker_id"],
                "sentence_id": b1["sentence_id"],
                "text_length": b1["text_length"],
                "b1_experiment_id": best_id,
                "b1_mel_l1": b1["mel_l1"],
                "b2_mel_l1": b2["mel_l1"],
                "delta_mel_l1": delta_l1,
                "delta_mel_l2_rmse": delta_l2,
                "category": "提升" if delta_l1 < 0 else "下降" if delta_l1 > 0 else "不变",
            }
        )
    return rows, best_id


def _report(
    device: torch.device,
    records: Sequence[dict[str, Any]],
    aggregates: Sequence[dict[str, Any]],
    differences: Sequence[dict[str, Any]],
    best_b1: str | None,
) -> str:
    lines = [
        "# B1/B2/B3 统一生成评估",
        "",
        f"- 设备：`{device}`",
        "- Mel 对齐：将参考目标 Mel 线性插值到源语音帧数。",
        "- latency/RTF 范围：缓存条件特征到生成 Mel，不包含离线特征提取与声码器。",
        "- CER/WER 与生成语音说话人相似度：当前没有声码器，未计算。",
        "",
        "## 聚合结果",
        "",
        "| 实验 | 模型 | NFE | 样本数 | Mel L1 | Mel L2 RMSE | latency ms | RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['experiment_id']} | {row['model']} | {row['nfe']} | "
            f"{row['sample_count']} | {float(row['mel_l1_mean']):.6f} | "
            f"{float(row['mel_l2_rmse_mean']):.6f} | "
            f"{float(row['latency_ms_mean']):.3f} | {float(row['rtf_mean']):.6f} |"
        )
    lines.extend(["", "## B1 → B2 样本差异", ""])
    if not differences:
        lines.append("完整 100 组结果尚未产生，暂不选择 B1 最优 NFE，也不输出样本差异结论。")
    else:
        improved = sum(row["category"] == "提升" for row in differences)
        declined = sum(row["category"] == "下降" for row in differences)
        unchanged = len(differences) - improved - declined
        lines.extend(
            [
                f"按平均 Mel L1 选择的 B1 基线为 `{best_b1}`。",
                f"B2 相对该基线：提升 {improved}，下降 {declined}，不变 {unchanged}。",
                "逐样本结果见 `b1_b2_sample_differences.csv`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "Mel 距离可用于检查声学重构趋势，但不能替代 CER/WER、说话人相似度或 MOS。",
            "只有接入统一声码器并补全语音域指标后，才能回答 MeanFlow 是否适合 VC。",
            "",
        ]
    )
    return "\n".join(lines)


# ====================
# 5. Core Logic
# ====================


def _execute(
    plan_path: Path,
    manifest_path: Path,
    vctk_root: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    device_name: str,
    base_seed: int,
    limit: int | None,
    experiment_ids: Sequence[str] | None,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated_mels"
    ledger_path = output_dir / "evaluation_records.jsonl"
    per_sample_path = output_dir / "per_sample_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    difference_path = output_dir / "b1_b2_sample_differences.csv"
    report_path = output_dir / "unified_evaluation_report.md"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 评估，但当前 PyTorch 环境不可用 CUDA")

    specs = _load_specs(plan_path.expanduser().resolve())
    if experiment_ids:
        selected = set(experiment_ids)
        specs = [spec for spec in specs if spec.experiment_id in selected]
        missing = sorted(selected - {spec.experiment_id for spec in specs})
        if missing:
            raise ValueError(f"未知或不可运行实验: {', '.join(missing)}")

    first_config = _model_config(specs[0].run_dir)
    microphone = str(first_config["data"]["microphone"])
    manifest, by_id = _load_manifest_samples(
        manifest_path.expanduser().resolve(), vctk_root.expanduser().resolve(), microphone
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        manifest = manifest[:limit]
    resolved_feature_cache = feature_cache_dir.expanduser().resolve()
    _validate_feature_cache(manifest, resolved_feature_cache)

    records = _load_ledger(ledger_path)
    completed = {(str(record["experiment_id"]), str(record["source_id"])) for record in records}
    for spec in specs:
        model, config = _load_model(spec.run_dir, device)
        collator = _collator(config)
        sample_rate = int(config["data"]["sample_rate"])
        pending_rows = [
            row for row in manifest if (spec.experiment_id, row["source_id"]) not in completed
        ]
        if pending_rows:
            warm_batch, _, _ = _prepare_pair(
                pending_rows[0], by_id, config, resolved_feature_cache, collator
            )
            _warm_up(model, warm_batch, spec.nfe, base_seed - 1, device)
        for row in manifest:
            key = (spec.experiment_id, row["source_id"])
            if key in completed:
                continue
            source_batch, target_mel, audio_samples = _prepare_pair(
                row, by_id, config, resolved_feature_cache, collator
            )
            seed = base_seed + int(row["index"])
            record = _evaluate_sample(
                model,
                source_batch,
                target_mel,
                audio_samples,
                sample_rate,
                spec,
                row,
                seed,
                device,
                generated_dir,
            )
            _append_ledger(ledger_path, record)
            records.append(record)
            completed.add(key)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    relevant_ids = {spec.experiment_id for spec in specs}
    relevant_sources = {row["source_id"] for row in manifest}
    selected_records = [
        record
        for record in records
        if record["experiment_id"] in relevant_ids and record["source_id"] in relevant_sources
    ]
    selected_records.sort(key=lambda row: (str(row["experiment_id"]), int(row["sample_index"])))
    aggregates = _aggregate(selected_records)
    differences, best_b1 = _difference_rows(selected_records, aggregates)

    _write_csv(per_sample_path, selected_records, PER_SAMPLE_FIELDS)
    _write_csv(
        aggregate_path,
        aggregates,
        (
            "experiment_id",
            "model",
            "nfe",
            "sample_count",
            "mel_l1_mean",
            "mel_l1_median",
            "mel_l2_rmse_mean",
            "latency_ms_mean",
            "rtf_mean",
        ),
    )
    _write_csv(
        difference_path,
        differences,
        (
            "source_id",
            "source_speaker_id",
            "reference_id",
            "reference_speaker_id",
            "sentence_id",
            "text_length",
            "b1_experiment_id",
            "b1_mel_l1",
            "b2_mel_l1",
            "delta_mel_l1",
            "delta_mel_l2_rmse",
            "category",
        ),
    )
    report_path.write_text(
        _report(device, selected_records, aggregates, differences, best_b1),
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "per_sample": per_sample_path,
        "aggregate": aggregate_path,
        "differences": difference_path,
        "ledger": ledger_path,
        "generated_mels": generated_dir,
    }


# ====================
# 6. Public API
# ====================


def evaluate_unified(
    plan_path: Path,
    manifest_path: Path,
    vctk_root: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    base_seed: int = 12345,
    limit: int | None = None,
    experiment_ids: Sequence[str] | None = None,
) -> dict[str, Path]:
    """执行可恢复的固定清单统一 Mel 生成与效率评估。"""
    return _execute(
        plan_path,
        manifest_path,
        vctk_root,
        feature_cache_dir,
        output_dir,
        device,
        base_seed,
        limit,
        experiment_ids,
    )
