"""在固定 VCTK 清单上统一评估 B1、B2、B3 的 Mel 质量与推理效率。"""

# ====================
# 1. Imports
# ====================


import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.analysis.unified.checkpoints import (
    CHECKPOINT_SAFE_GLOBALS as CHECKPOINT_SAFE_GLOBALS,
)
from src.analysis.unified.checkpoints import PER_SAMPLE_FIELDS as PER_SAMPLE_FIELDS
from src.analysis.unified.checkpoints import EvaluationSpec as EvaluationSpec
from src.analysis.unified.checkpoints import _best_checkpoint as _best_checkpoint
from src.analysis.unified.checkpoints import _load_model as _load_model
from src.analysis.unified.checkpoints import _model_config as _model_config
from src.analysis.unified.data import _collator as _collator
from src.analysis.unified.data import _dataset as _dataset
from src.analysis.unified.data import _load_manifest_samples as _load_manifest_samples
from src.analysis.unified.data import _load_specs as _load_specs
from src.analysis.unified.data import _prepare_pair as _prepare_pair
from src.analysis.unified.data import _validate_feature_cache as _validate_feature_cache
from src.analysis.unified.identity import _build_run_identity as _build_run_identity
from src.analysis.unified.identity import _checkpoint_sha256 as _checkpoint_sha256
from src.analysis.unified.identity import (
    _feature_cache_fingerprint as _feature_cache_fingerprint,
)
from src.analysis.unified.identity import _identity_mismatch as _identity_mismatch
from src.analysis.unified.identity import _load_identity as _load_identity
from src.analysis.unified.identity import _sha256_file as _sha256_file
from src.analysis.unified.identity import _source_version as _source_version
from src.analysis.unified.identity import _write_identity as _write_identity
from src.analysis.unified.records import _append_ledger as _append_ledger
from src.analysis.unified.records import _load_ledger as _load_ledger
from src.analysis.unified.records import _read_csv as _read_csv
from src.analysis.unified.records import _write_csv as _write_csv
from src.analysis.unified.reporting import _aggregate as _aggregate
from src.analysis.unified.reporting import _difference_rows as _difference_rows
from src.analysis.unified.reporting import _mean_or_nan as _mean_or_nan
from src.analysis.unified.reporting import _report as _report

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


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
    resolved_manifest_path = manifest_path.expanduser().resolve()
    full_manifest, by_id = _load_manifest_samples(
        resolved_manifest_path, vctk_root.expanduser().resolve(), microphone
    )
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须为正数")

    resolved_feature_cache = feature_cache_dir.expanduser().resolve()
    manifest = full_manifest if limit is None else full_manifest[:limit]
    _validate_feature_cache(manifest, resolved_feature_cache)

    identity_path = output_dir / "evaluation_identity.json"
    current_identity = _build_run_identity(
        specs,
        resolved_manifest_path,
        full_manifest,
        resolved_feature_cache,
        first_config,
        device_name,
        base_seed,
    )
    records = _load_ledger(ledger_path)
    existing_identity = _load_identity(identity_path)
    if existing_identity is None:
        if records:
            raise ValueError("旧评估账本缺少身份信息，拒绝续跑以避免误复用历史结果")
        _write_identity(identity_path, current_identity)
    else:
        differences = _identity_mismatch(existing_identity, current_identity)
        if differences:
            raise ValueError(f"评估身份不一致，拒绝续跑: {', '.join(differences)}")
        registered = existing_identity.get("experiments", {})
        unregistered = {str(record["experiment_id"]) for record in records} - set(registered)
        if unregistered:
            raise ValueError(f"评估账本缺少实验身份，拒绝续跑: {', '.join(sorted(unregistered))}")
        current_identity["experiments"] = {**registered, **current_identity["experiments"]}
        _write_identity(identity_path, current_identity)

    completed = {
        (str(record["experiment_id"]), str(record["source_id"]), str(record["reference_id"]))
        for record in records
    }
    for spec in specs:
        model, config = _load_model(spec.run_dir, device)
        collator = _collator(config)
        sample_rate = int(config["data"]["sample_rate"])
        pending_rows = [
            row
            for row in manifest
            if (spec.experiment_id, row["source_id"], row["reference_id"]) not in completed
        ]
        if pending_rows:
            warm_batch, _, _ = _prepare_pair(
                pending_rows[0], by_id, config, resolved_feature_cache, collator
            )
            _warm_up(model, warm_batch, spec.nfe, base_seed - 1, device)
        for row in manifest:
            key = (spec.experiment_id, row["source_id"], row["reference_id"])
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
    # 每个 (实验, 源, 参考) 在结果中最多保留最早写入的一条，避免重复计入聚合。
    deduped_records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for record in selected_records:
        identity_key = (
            str(record["experiment_id"]),
            str(record["source_id"]),
            str(record["reference_id"]),
        )
        if identity_key in seen_keys:
            continue
        seen_keys.add(identity_key)
        deduped_records.append(record)
    selected_records = deduped_records
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
# 对外提供的函数
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
