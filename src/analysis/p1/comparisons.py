"""模块说明：

- 从固定配对的逐样本结果计算方法差异与分层汇总。
- 用源说话人和语句分别重采样，避免将相关转换当成独立样本。
"""

# ====================
# 1. Imports
# ====================

import csv
from pathlib import Path

import numpy as np

# ====================
# 2. Definitions
# ====================

COMPARISONS = (
    ("nfe_b2_01", "nfe_b1_01"),
    ("nfe_b3_01", "nfe_b2_01"),
    ("nfe_b3_01", "nfe_b1_01"),
    ("nfe_b1_05", "nfe_b1_01"),
    ("nfe_b1_10", "nfe_b1_01"),
    ("nfe_b1_50", "nfe_b1_01"),
)
METRICS = (
    "cer",
    "wer",
    "target_speaker_similarity",
    "source_speaker_similarity",
    "mel_l1",
    "source_f0_corr",
    "source_f0_centered_rmse_semitones",
    "source_vuv_error",
    "source_energy_corr",
)


# ====================
# 3. Utilities
# ====================


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ====================
# 5. Core Logic
# ====================


def paired_comparisons(path: Path):
    """生成 2000 次双轴重采样的配对差异区间与说话人分层。"""
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    indexed = {(r["experiment_id"], r["source_id"]): r for r in rows}
    sources = sorted({r["source_id"] for r in rows})
    metadata = [indexed[("nfe_b1_01", uid)] for uid in sources]
    speakers = sorted({r["source_speaker_id"] for r in metadata})
    sentences = sorted({r["sentence_id"] for r in metadata})
    si = np.array([speakers.index(r["source_speaker_id"]) for r in metadata])
    ti = np.array([sentences.index(r["sentence_id"]) for r in metadata])
    rng = np.random.default_rng(20261024)
    sw = rng.multinomial(len(speakers), np.full(len(speakers), 1 / len(speakers)), size=2000)
    tw = rng.multinomial(len(sentences), np.full(len(sentences), 1 / len(sentences)), size=2000)
    weights = sw[:, si] * tw[:, ti]
    comparisons = []
    for a, b in COMPARISONS:
        for metric in METRICS:
            delta = np.array(
                [
                    float(indexed[(a, uid)][metric]) - float(indexed[(b, uid)][metric])
                    for uid in sources
                ]
            )
            valid = np.isfinite(delta)
            if not valid.any():
                continue
            w = weights[:, valid]
            numerator = w @ delta[valid]
            denominator = w.sum(axis=1)
            bootstrap = numerator[denominator > 0] / denominator[denominator > 0]
            low, high = np.percentile(bootstrap, [2.5, 97.5])
            comparisons.append(
                {
                    "experiment_a": a,
                    "experiment_b": b,
                    "metric": metric,
                    "paired_n": int(valid.sum()),
                    "mean_a_minus_b": float(delta[valid].mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "fraction_a_greater": float((delta[valid] > 0).mean()),
                    "resamples": 2000,
                    "resampling": "source_speaker_and_sentence_independent_counts",
                }
            )
    _write(path.parent / "paired_comparisons.csv", comparisons)
    strata = []
    for experiment in sorted({r["experiment_id"] for r in rows}):
        for speaker in speakers:
            group = [
                r
                for r in rows
                if r["experiment_id"] == experiment and r["source_speaker_id"] == speaker
            ]
            row = {"experiment_id": experiment, "source_speaker_id": speaker, "n": len(group)}
            for metric in METRICS:
                values = np.array([float(r[metric]) for r in group])
                values = values[np.isfinite(values)]
                row[metric] = float(values.mean()) if len(values) else float("nan")
            strata.append(row)
    _write(path.parent / "speaker_strata.csv", strata)
    return comparisons


# ====================
# 6. Public API
# ====================


def run(path: Path):
    """读取 P1 产物并写入配对比较与分层统计。"""
    return paired_comparisons(path)
