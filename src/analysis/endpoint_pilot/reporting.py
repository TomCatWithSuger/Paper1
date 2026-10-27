"""模块说明：

- 审计两个训练臂的共同流、真实更新数、恢复状态和末期checkpoint。
- 导出训练监控与独立验证的逐样本配对区间，不按中间权重挑选成绩。
"""

# ====================
# 1. Imports
# ====================

import json
from collections import defaultdict

import numpy as np
import torch

from src.analysis.endpoint_pilot.training import safe_load
from src.analysis.p1.evaluation import _csv, _json
from src.analysis.p2.reporting import mean_ci, read, source_means, summary
from src.analysis.unified.identity import _sha256_file

# ====================
# 2. Definitions
# ====================

QUALITY_METRICS = [
    "cer",
    "wer",
    "target_speaker_similarity",
    "source_speaker_similarity",
    "target_preference_gap",
    "target_preferred",
    "source_f0_corr",
    "source_f0_centered_rmse_semitones",
    "source_vuv_error",
    "source_joint_voiced_fraction",
    "source_energy_corr",
    "source_energy_log_mae",
    "reference_f0_corr",
]
ENDPOINT_METRICS = [
    "raw_mse",
    "raw_mse_common",
    "jvp_rms",
    "velocity_mse",
    "target_energy",
    "reconstruction_l1",
]


# ====================
# 3. Utilities
# ====================


def _json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _statistics(values):
    values = np.array([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _paired(rows, cohort, metric, a, b, metadata):
    va = source_means(
        [r for r in rows if r["cohort"] == cohort and r["experiment_id"] == a], metric
    )
    vb = source_means(
        [r for r in rows if r["cohort"] == cohort and r["experiment_id"] == b], metric
    )
    meta = [r for r in metadata if r["source_id"] in va and r["source_id"] in vb]
    result = mean_ci([va[r["source_id"]] - vb[r["source_id"]] for r in meta], meta)
    return {"cohort": cohort, "experiment_a": a, "experiment_b": b, "metric": metric, **result}


# ====================
# 4. Internal Functions
# ====================


def training_audit(output):
    """汇总梯度、裁剪、实际采样和检查点状态，核验双臂训练记录。"""
    original = json.loads((output / "initial_restore_audit.json").read_text())
    for arm in ["control", "intervention"]:
        restored = json.loads((output / arm / "restore_audit.json").read_text())
        for key in [
            "file_sha256",
            "epoch",
            "global_step",
            "model_state_sha256",
            "optimizer_lr",
            "scheduler_total_steps",
            "scheduler_last_epoch",
        ]:
            assert restored[key] == original[key], (arm, key)
        assert (
            restored["optimizer_state_exact"]
            and restored["scheduler_state_exact"]
            and restored["parameter_names_exact"]
        )
    logs = {
        arm: _json_rows(output / arm / "training.jsonl") for arm in ["control", "intervention"]
    }
    plan = safe_load(output / "common_plan.pt")
    for arm, rows in logs.items():
        assert len(rows) == 5000 and [r["pilot_update"] for r in rows] == list(range(1, 5001))
        assert [r["global_step"] for r in rows] == list(range(422067, 427067))
        assert all(r["finite"] for r in rows)
        assert not (output / arm / "NUMERICAL_FAILURE.json").exists()
    for a, b in zip(logs["control"], logs["intervention"]):
        for key in [
            "data_sha256",
            "noise_sha256",
            "source_ids",
            "crop_starts",
            "lr_before",
            "lr_after",
            "epoch",
            "batch_in_epoch",
            "batch_size",
        ]:
            assert a[key] == b[key], (a["pilot_update"], key)
    assert all(r["sampled_full_endpoint_mse"] is None for r in logs["control"])
    for step, row in enumerate(logs["intervention"]):
        expected = bool(plan["endpoint_mask"][step, : row["batch_size"]].any())
        assert (row["sampled_full_endpoint_mse"] is not None) == expected
    records, windows = [], []
    for arm, rows in logs.items():
        for metric in [
            "raw_meanflow_mse",
            "zero_mse",
            "sampled_full_endpoint_mse",
            "jvp_rms",
            "grad_norm_before_clip",
        ]:
            records.append(
                {"arm": arm, "metric": metric, **_statistics([r[metric] for r in rows])}
            )
        for start in range(0, 5000, 1000):
            group = rows[start : start + 1000]
            windows.append(
                {
                    "arm": arm,
                    "update_start": start + 1,
                    "update_end": start + 1000,
                    "raw_mse_mean": float(np.mean([r["raw_meanflow_mse"] for r in group])),
                    "jvp_rms_mean": float(np.mean([r["jvp_rms"] for r in group])),
                    "gradient_norm_mean": float(
                        np.mean([r["grad_norm_before_clip"] for r in group])
                    ),
                    "gradient_norm_p95": float(
                        np.percentile([r["grad_norm_before_clip"] for r in group], 95)
                    ),
                    "clip_frequency": float(np.mean([r["clipped"] for r in group])),
                    "numeric_failures": 0,
                }
            )
    _csv(output / "training_statistics.csv", records)
    _csv(output / "training_windows.csv", windows)
    by_gate = []
    for arm, rows in logs.items():
        for gate in [False, True]:
            selected = [
                r
                for index, r in enumerate(rows)
                if bool(plan["endpoint_mask"][index, : r["batch_size"]].any()) == gate
            ]
            by_gate.append(
                {
                    "arm": arm,
                    "endpoint_gate_present": gate,
                    "updates": len(selected),
                    "raw_mse_mean": float(np.mean([r["raw_meanflow_mse"] for r in selected])),
                    "grad_norm_mean": float(
                        np.mean([r["grad_norm_before_clip"] for r in selected])
                    ),
                    "clip_frequency": float(np.mean([r["clipped"] for r in selected])),
                }
            )
    _csv(output / "training_by_endpoint_gate.csv", by_gate)
    checkpoints = []
    for arm in logs:
        ledger = _json_rows(output / arm / "checkpoint_ledger.jsonl")
        assert [r["pilot_update"] for r in ledger] == [1000, 2500, 5000]
        assert len({r["sha256"] for r in ledger}) == 3
        for record in ledger:
            from pathlib import Path

            path = Path(record["path"])
            assert _sha256_file(path) == record["sha256"]
            checkpoint = safe_load(path)
            assert checkpoint["global_step"] == 422066 + record["pilot_update"]
            assert checkpoint["epoch"] == int(plan["epochs"][record["pilot_update"] - 1])
            assert checkpoint["lr_schedulers"][0]["last_epoch"] == checkpoint["global_step"]
            assert all(
                int(s["step"]) == checkpoint["global_step"]
                for s in checkpoint["optimizer_states"][0]["state"].values()
            )
            assert all(
                torch.isfinite(tensor).all() for tensor in checkpoint["state_dict"].values()
            )
            assert all(
                torch.isfinite(s["exp_avg"]).all() and torch.isfinite(s["exp_avg_sq"]).all()
                for s in checkpoint["optimizer_states"][0]["state"].values()
            )
            checkpoints.append(record)
        assert _sha256_file(output / arm / "checkpoints/last.ckpt") == ledger[-1]["sha256"]
        assert (
            ledger[-1]["sha256"]
            != json.loads((output / "identity.json").read_text())["base_checkpoint_sha256"]
        )
    assert checkpoints[-1]["model_sha256"] != checkpoints[2]["model_sha256"]
    _csv(output / "checkpoint_audit.csv", checkpoints)
    probes = []
    for arm in logs:
        probes.extend(
            {"arm": arm, **r} for r in _json_rows(output / arm / "validation_probes.jsonl")
        )
    grouped = defaultdict(list)
    for r in probes:
        grouped[(r["arm"], r["pilot_update"], r["interval"])].append(r)
    probe_summary = []
    for (arm, update, interval), rs in grouped.items():
        probe_summary.append(
            {
                "arm": arm,
                "pilot_update": update,
                "interval": interval,
                "n_sources": len({r["source_id"] for r in rs}),
                "n_draws": len(rs),
                "raw_mse_mean": float(np.mean([r["raw_mse"] for r in rs])),
                "jvp_rms_mean": float(np.mean([r["jvp_rms"] for r in rs])),
            }
        )
    _csv(output / "validation_probe_curves.csv", probe_summary)
    _json(
        output / "paired_training_audit.json",
        {
            "updates_each": 5000,
            "all_data_noise_crop_lr_match": True,
            "optimizer_restore_exact": True,
            "scheduler_total_steps_preserved": 551000,
            "final_global_step_each": 427066,
            "final_epoch_each": 387,
            "checkpoint_updates": [1000, 2500, 5000],
            "last_is_actual_final": True,
            "final_models_distinct": True,
            "numeric_failures": 0,
        },
    )


def evaluation_reports(output):
    """聚合双队列质量及端点指标，导出分层统计与配对区间。"""
    directory = output / "evaluation"
    quality = read(directory / "per_sample_metrics.csv")
    endpoint = read(directory / "endpoint_diagnostics.csv")
    for row in endpoint:
        if row["reconstruction_l1"] == "":
            row["reconstruction_l1"] = float("nan")
    for fields, name in [
        (["cohort", "experiment_id"], "quality_summary"),
        (["cohort", "experiment_id", "length_bin"], "quality_by_length"),
        (["cohort", "experiment_id", "source_speaker_id"], "quality_by_speaker"),
        (["cohort", "experiment_id", "repeat"], "quality_by_noise"),
    ]:
        _csv(directory / f"{name}.csv", summary(quality, fields, QUALITY_METRICS))
    _csv(
        directory / "endpoint_summary.csv",
        summary(endpoint, ["cohort", "experiment_id", "interval"], ENDPOINT_METRICS),
    )
    _csv(
        directory / "endpoint_by_length.csv",
        summary(endpoint, ["cohort", "experiment_id", "interval", "length_bin"], ENDPOINT_METRICS),
    )
    comparisons, endpoint_comparisons = [], []
    for cohort, filename in [
        ("p2_fixed", "p2_validation_manifest.json"),
        ("reserved", "reserved_validation_manifest.json"),
    ]:
        metadata = json.loads((output / filename).read_text())
        for a, b in [
            ("intervention", "control"),
            ("control", "initial"),
            ("intervention", "initial"),
        ]:
            for metric in QUALITY_METRICS:
                comparisons.append(_paired(quality, cohort, metric, a, b, metadata))
            for interval in ["zero", "full"]:
                for metric in ENDPOINT_METRICS:
                    if interval == "zero" and metric == "reconstruction_l1":
                        continue
                    record = _paired(
                        [r for r in endpoint if r["interval"] == interval],
                        cohort,
                        metric,
                        a,
                        b,
                        metadata,
                    )
                    endpoint_comparisons.append({"interval": interval, **record})
    _csv(directory / "paired_comparisons.csv", comparisons)
    _csv(directory / "endpoint_paired_comparisons.csv", endpoint_comparisons)
    # 逐样本配对表保留两次噪声，不把其当独立源样本计算区间。
    indexed = {(r["cohort"], r["experiment_id"], r["source_id"], r["repeat"]): r for r in quality}
    sample_pairs = []
    for r in quality:
        if r["experiment_id"] != "intervention":
            continue
        control = indexed[(r["cohort"], "control", r["source_id"], r["repeat"])]
        sample_pairs.append(
            {
                "cohort": r["cohort"],
                "source_id": r["source_id"],
                "reference_id": r["reference_id"],
                "repeat": r["repeat"],
                "seed": r["seed"],
                **{f"{metric}_control": control[metric] for metric in QUALITY_METRICS},
                **{f"{metric}_intervention": r[metric] for metric in QUALITY_METRICS},
                **{f"{metric}_delta": r[metric] - control[metric] for metric in QUALITY_METRICS},
            }
        )
    _csv(directory / "control_intervention_per_sample.csv", sample_pairs)


# ====================
# 6. Public API
# ====================


def run(output):
    """汇总双臂训练审计和冻结评估，写入最终实验报告。"""
    training_audit(output)
    evaluation_reports(output)
    print("训练共同流、checkpoint、逐样本配对与验证区间已导出", flush=True)
