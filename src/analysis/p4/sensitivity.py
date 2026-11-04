"""模块说明：对 P4 总体配对差补充源/目标说话人和文本的三向聚类敏感性分析。"""

# ====================
# 1. Imports
# ====================

import json
from pathlib import Path

import numpy as np

from src.analysis.p4.identity import (
    CONFIGS,
    METRICS,
    SEED,
    read_csv,
    sha256,
    source_means,
    speaker_genders,
    validate_rows,
    write_csv,
)

# ====================
# 3. Utilities
# 通用工具函数
# ====================


def crossed_ci(values, metadata):
    """按源说话人、目标说话人和实际文本重采样，固定语句内噪声均值。"""
    weights = np.ones((5000, len(metadata)))
    rng = np.random.default_rng(SEED)
    for field in ("source_speaker_id", "reference_speaker_id", "source_text"):
        levels = sorted({row[field] for row in metadata})
        indices = [levels.index(row[field]) for row in metadata]
        draws = rng.multinomial(len(levels), np.full(len(levels), 1 / len(levels)), 5000)
        weights *= draws[:, indices]
    denominator = weights.sum(axis=1)
    valid = denominator > 0
    samples = (weights[valid] @ np.asarray(values)) / denominator[valid]
    return tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def run(output):
    """为两份评分补充源与目标说话人及文本的三向聚类敏感性区间。"""
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    genders = speaker_genders(protocol["speaker_info"])
    results, inputs = [], {}
    for scorer, name in [
        ("wavlm_ecapa", "existing_per_sample.csv"),
        ("campplus", "campplus/per_sample.csv"),
    ]:
        path = output / name
        if scorer == "campplus" and not (output / "campplus/COMPLETE.json").is_file():
            raise RuntimeError("必须先完成独立评分")
        inputs[name] = sha256(path)
        means = source_means(validate_rows(read_csv(path), genders))
        indexed = {(row["experiment_id"], row["source_id"]): row for row in means}
        for a, b in ((CONFIGS[0], CONFIGS[2]), (CONFIGS[1], CONFIGS[2]), (CONFIGS[0], CONFIGS[1])):
            metadata = [row for row in means if row["experiment_id"] == a]
            for metric in METRICS:
                values = [row[metric] - indexed[(b, row["source_id"])][metric] for row in metadata]
                lo, hi = crossed_ci(values, metadata)
                results.append(
                    {
                        "scorer": scorer,
                        "config_a": a,
                        "config_b": b,
                        "metric": metric,
                        "n_sources": len(metadata),
                        "difference": float(np.mean(values)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
    destination = output / "source_target_text_sensitivity.csv"
    if destination.exists():
        raise FileExistsError(destination)
    write_csv(destination, results)
    audit = {
        "input_sha256": inputs,
        "code_sha256": sha256(__file__),
        "seed": SEED,
        "bootstrap_repeats": 5000,
        "clusters": ["source_speaker_id", "reference_speaker_id", "source_text"],
        "purpose": "补充目标说话人复用的敏感性核查；探索性分析，非跨训练种子或多重比较校正",
    }
    (output / "sensitivity_protocol.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source/sensitivity.py").write_bytes(Path(__file__).read_bytes())
