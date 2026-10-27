"""模块说明：

- 导出 P2 分层统计、成对区间及端点误差/转换质量的关联。
- 重复噪声先在源语句内平均，按说话人与实际文本重采样。
"""

# ====================
# 1. Imports
# ====================

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from src.analysis.p1.evaluation import _csv, _json
from src.analysis.p2.common import load_manifest
from src.analysis.vocoder.metrics import _edit_distance

# ====================
# 2. Definitions
# ====================

INTERVAL_METRICS = [
    "raw_mse",
    "raw_mse_common",
    "velocity_mse",
    "velocity_mse_common",
    "jvp_correction_rms",
    "target_energy",
    "raw_over_target_energy",
    "correction_squared",
    "cross_term",
]
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
    "applied_content_cer",
    "applied_content_wer",
    "applied_target_similarity",
    "self_mel_l1",
    "donor_cer",
    "donor_wer",
]


# ====================
# 3. Utilities
# ====================


def read(path):
    """读取 CSV 记录，将可解析数值转为浮点并保留文本字段。"""
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def weights_for(metadata):
    """按源说话人与真实文本构造固定种子的乘积重采样权重。"""
    speakers = sorted({r["source_speaker_id"] for r in metadata})
    texts = sorted({r["source_text"] for r in metadata})
    si = np.array([speakers.index(r["source_speaker_id"]) for r in metadata])
    ti = np.array([texts.index(r["source_text"]) for r in metadata])
    rng = np.random.default_rng(20261025)
    sw = rng.multinomial(len(speakers), np.full(len(speakers), 1 / len(speakers)), size=2000)
    tw = rng.multinomial(len(texts), np.full(len(texts), 1 / len(texts)), size=2000)
    return sw[:, si] * tw[:, ti]


def mean_ci(values, metadata):
    """过滤非有限值并计算源语句级均值及聚类重采样区间。"""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return {
            "n_sources": 0,
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    weights = weights_for(metadata)[:, valid]
    values = values[valid]
    denominator = weights.sum(1)
    samples = weights[denominator > 0] @ values / denominator[denominator > 0]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "n_sources": len(values),
        "mean": float(values.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def summary(rows, group_fields, metrics):
    """按指定字段聚合指标，分别报告行数、源数及有效指标数。"""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for key, group in groups.items():
        value = dict(zip(group_fields, key))
        value.update(
            n_rows=len(group),
            n_sources=len({r["source_id"] for r in group}),
            n_draws=len({(r["source_id"], r.get("repeat", 0)) for r in group}),
        )
        for metric in metrics:
            array = np.asarray([r[metric] for r in group], dtype=float)
            finite = array[np.isfinite(array)]
            if len(finite):
                value[f"{metric}_mean"] = float(finite.mean())
                value[f"{metric}_median"] = float(np.median(finite))
                value[f"{metric}_valid_n"] = len(finite)
        if "valid_frames" in group[0]:
            value["valid_frames_total"] = sum(r["valid_frames"] for r in group)
            value["raw_mse_frame_weighted"] = (
                sum(r["raw_mse"] * r["valid_frames"] for r in group) / value["valid_frames_total"]
            )
        result.append(value)
    return result


def source_means(rows, metric):
    """先在同一源语句内平均重复噪声的指定指标。"""
    groups = defaultdict(list)
    for row in rows:
        value = row[metric]
        if np.isfinite(value):
            groups[row["source_id"]].append(value)
    return {uid: float(np.mean(values)) for uid, values in groups.items()}


# ====================
# 4. Internal Functions
# ====================


def interval_reports(output, metadata):
    """汇总区间、位置和句长分层，并输出相对零区间的配对差。"""
    for name in ["d1", "d1_center_matched", "d1_boundary"]:
        rows = read(output / name / "per_sample_intervals.csv")
        for row in rows:
            row["r_bin"] = min(int(row["r"] * 4), 3)
            row["t_bin"] = min(int(row["t"] * 4), 3)
        for fields, filename in [
            (["interval_group"], "interval_summary"),
            (["interval_group", "length_bin"], "interval_by_length"),
            (["r_bin", "t_bin", "length_bin"], "r_t_length_bins"),
            (["r", "t", "delta"], "grid_summary"),
            (["interval_group", "partition"], "interval_by_partition"),
        ]:
            _csv(output / name / f"{filename}.csv", summary(rows, fields, INTERVAL_METRICS))
        if name == "d1":
            for field in ["r_bin", "t_bin"]:
                _csv(
                    output / name / f"{field}_summary.csv",
                    summary(rows, [field], INTERVAL_METRICS),
                )
        comparisons = []
        baseline = [r for r in rows if r["interval_group"] == "zero"]
        for group in ["short", "medium", "long", "full"]:
            selected = [r for r in rows if r["interval_group"] == group]
            for metric in ["raw_mse_common", "velocity_mse_common", "raw_over_target_energy"]:
                a, b = source_means(selected, metric), source_means(baseline, metric)
                ids = [
                    r["source_id"] for r in metadata if r["source_id"] in a and r["source_id"] in b
                ]
                meta = [r for r in metadata if r["source_id"] in ids]
                stats = mean_ci([a[r["source_id"]] - b[r["source_id"]] for r in meta], meta)
                comparisons.append(
                    {"group_a": group, "group_b": "zero", "metric": metric, **stats}
                )
        _csv(output / name / "paired_vs_zero.csv", comparisons)


def quality_reports(output, metadata):
    """补充身份偏好及替换内容指标，生成质量分层与配对统计。"""
    directory = output / "conversion"
    rows = read(directory / "per_sample_metrics.csv")
    normalized_texts = {r["source_id"]: r["normalized_reference"] for r in rows}
    for row in rows:
        donor = normalized_texts[row["donor_id"]]
        hypothesis = row["normalized_hypothesis"]
        row["donor_cer"] = _edit_distance(
            donor.replace(" ", ""), hypothesis.replace(" ", "")
        ) / len(donor.replace(" ", ""))
        row["donor_wer"] = _edit_distance(donor.split(), hypothesis.split()) / len(donor.split())
        row["target_preference_gap"] = (
            row["target_speaker_similarity"] - row["source_speaker_similarity"]
        )
        row["target_preferred"] = float(row["target_preference_gap"] > 0)
    _csv(directory / "per_sample_enriched.csv", rows)
    for fields, filename in [
        (["experiment_id"], "quality_summary"),
        (["experiment_id", "length_bin"], "quality_by_length"),
        (["experiment_id", "partition"], "quality_by_partition"),
        (["experiment_id", "repeat"], "quality_by_noise"),
        (["experiment_id", "source_speaker_id"], "quality_by_speaker"),
    ]:
        _csv(directory / f"{filename}.csv", summary(rows, fields, QUALITY_METRICS))
    pairs = [
        ("b2_base", "b1_anchor"),
        ("b2_content_donor", "b2_base"),
        ("b2_speaker_source", "b2_base"),
    ]
    times = ["b3_best_s080", "b3_best_s090", "b3_best_s095", "b3_best_s100"]
    pairs += [(times[i], times[j]) for i in range(len(times)) for j in range(i + 1, len(times))]
    comparisons = []
    for a, b in pairs:
        for subset in ["all", "diagnostic_A", "confirmation_B", "noise_0", "noise_1"]:
            selected = rows
            if subset.startswith("noise"):
                selected = [r for r in selected if r["repeat"] == int(subset[-1])]
            elif subset != "all":
                selected = [r for r in selected if r["partition"] == subset]
            for metric in QUALITY_METRICS:
                va = source_means([r for r in selected if r["experiment_id"] == a], metric)
                vb = source_means([r for r in selected if r["experiment_id"] == b], metric)
                meta = [r for r in metadata if r["source_id"] in va and r["source_id"] in vb]
                stats = mean_ci([va[r["source_id"]] - vb[r["source_id"]] for r in meta], meta)
                comparisons.append(
                    {
                        "experiment_a": a,
                        "experiment_b": b,
                        "subset": subset,
                        "metric": metric,
                        **stats,
                    }
                )
    _csv(directory / "paired_comparisons.csv", comparisons)
    return rows


def endpoint_associations(output, metadata, quality):
    """关联逐句端点残差和转换质量，输出相关性及重采样区间。"""
    rows = [
        r for r in read(output / "d1/per_sample_intervals.csv") if r["interval_group"] == "full"
    ]
    quality = [r for r in quality if r["experiment_id"] == "b2_base"]
    records = []
    for metric in [
        "raw_mse_common",
        "velocity_mse_common",
        "jvp_correction_rms",
        "terminal_reconstruction_l1",
    ]:
        x = source_means(rows, metric)
        y = source_means(quality, "cer")
        meta = [r for r in metadata if r["source_id"] in x and r["source_id"] in y]
        xv = np.array([x[r["source_id"]] for r in meta])
        yv = np.array([y[r["source_id"]] for r in meta])
        duration = np.log(np.array([r["audio_seconds"] for r in meta]))

        def partial(a, b, d):
            matrix = np.column_stack([np.ones(len(d)), d])
            a = a - matrix @ np.linalg.lstsq(matrix, a, rcond=None)[0]
            b = b - matrix @ np.linalg.lstsq(matrix, b, rcond=None)[0]
            return float(np.corrcoef(a, b)[0, 1])

        boot, boot_partial = [], []
        for weights in weights_for(meta):
            ids = np.repeat(np.arange(len(meta)), weights)
            if len(ids) >= 4 and len(set(ids)) > 3:
                boot.append(float(spearmanr(xv[ids], yv[ids]).statistic))
                boot_partial.append(partial(xv[ids], yv[ids], duration[ids]))
        records.append(
            {
                "metric": metric,
                "outcome": "conversion_cer",
                "n_sources": len(meta),
                "spearman": float(spearmanr(xv, yv).statistic),
                "spearman_ci_low": float(np.nanpercentile(boot, 2.5)),
                "spearman_ci_high": float(np.nanpercentile(boot, 97.5)),
                "partial_pearson_log_duration": partial(xv, yv, duration),
                "partial_ci_low": float(np.nanpercentile(boot_partial, 2.5)),
                "partial_ci_high": float(np.nanpercentile(boot_partial, 97.5)),
                "causal_evidence": False,
            }
        )
    _csv(output / "d1/endpoint_quality_associations.csv", records)


def loss_reports(output, metadata):
    """聚合 B3 损失分解和先验分层，并审计早末期权重可比性。"""
    rows = read(output / "d2/loss_components.csv")
    metrics = [
        "raw_mse",
        "raw_mse_common",
        "adaptive_loss",
        "ssim_loss",
        "total_loss",
        "jvp_correction_rms",
        "velocity_mse",
        "zero_reconstruction_l1",
        "error_energy",
        "adaptive_gradient_weight",
        "margin_hit",
    ]
    for fields, name in [
        (["checkpoint"], "loss_summary"),
        (["checkpoint", "length_bin"], "loss_by_length"),
        (["checkpoint", "partition"], "loss_by_partition"),
        (["checkpoint", "pseudo"], "loss_by_prior"),
        (["checkpoint", "source_speaker_id"], "loss_by_speaker"),
    ]:
        _csv(output / "d2" / f"{name}.csv", summary(rows, fields, metrics))
    best = json.loads((output / "d2/best_model.json").read_text())
    last = json.loads((output / "d2/last_model.json").read_text())
    _json(
        output / "d2/checkpoint_comparison_validity.json",
        {
            "best": best,
            "last": last,
            "same_file_hash": best["sha256"] == last["sha256"],
            "can_compare_early_vs_final": best["sha256"] != last["sha256"]
            and last["epoch"] == 499,
            "interpretation": "现存last实际仍为epoch40，不能据此判断epoch499转换或选模一致性",
        },
    )
    # 同批伪源构造耦合多个样本，单模型分量区间按16个微批次成组重采样。
    component_intervals = []
    selected = [r for r in rows if r["checkpoint"] == "best"]
    blocks = sorted({r["batch_index"] for r in selected})
    draws = np.random.default_rng(20261025).integers(0, len(blocks), (2000, len(blocks)))
    for metric in metrics:
        values = np.array(
            [
                np.mean([r[metric] for r in selected if r["batch_index"] == block])
                for block in blocks
            ]
        )
        low, high = np.percentile(values[draws].mean(1), [2.5, 97.5])
        component_intervals.append(
            {
                "checkpoint": "best",
                "metric": metric,
                "n_sources": len({r["source_id"] for r in selected}),
                "n_draws": len(selected),
                "n_microbatch_blocks": len(blocks),
                "mean": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_unit": "microbatch4_shared_prior",
            }
        )
    _csv(output / "d2/component_intervals.csv", component_intervals)
    summaries = []
    for label in ["best", "last"]:
        audit = read(output / f"d2/{label}_original_val_pairing_audit.csv")
        count = sum(r["batch_size"] for r in audit)
        summaries.append(
            {
                "checkpoint": label,
                "audit_kind": "original_validation_pairing_algorithm",
                "n_samples": count,
                "n_batches": len(audit),
                "requested_pseudo": sum(r["requested_pseudo"] for r in audit),
                "effective_pseudo": sum(r["effective_pseudo"] for r in audit),
                "fallback_pure_noise": sum(r["fallback_pure_noise"] for r in audit),
                "intentional_pure_noise": sum(r["intentional_pure_noise"] for r in audit),
                "effective_fraction": sum(r["effective_pseudo"] for r in audit) / count,
            }
        )
        selected = [r for r in rows if r["checkpoint"] == label]
        effective = sum(r["pseudo"] == "True" for r in selected)
        summaries.append(
            {
                "checkpoint": label,
                "audit_kind": "actual_native_loss_prior",
                "n_samples": len(selected),
                "n_batches": len({(r["batch_index"], r["repeat"]) for r in selected}),
                "requested_pseudo": len(selected) // 2,
                "effective_pseudo": effective,
                "fallback_pure_noise": len(selected) // 2 - effective,
                "intentional_pure_noise": len(selected) // 2,
                "effective_fraction": effective / len(selected),
            }
        )
    _csv(output / "d2/pseudo_source_summary.csv", summaries)


# ====================
# 6. Public API
# ====================


def run(output):
    """从已有诊断产物生成区间、质量、相关性及损失分解报告。"""
    metadata = load_manifest(output)
    interval_reports(output, metadata)
    loss_reports(output, metadata)
    quality = quality_reports(output, metadata)
    endpoint_associations(output, metadata, quality)
    print("P2原始统计、分层、成对区间与关联分析已导出", flush=True)
