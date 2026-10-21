"""聚合 Mel 与效率指标并生成样本差异报告。"""

# ====================
# 1. Imports
# ====================


import math
from statistics import mean, median
from typing import Any, Iterable, Sequence

import torch

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _mean_or_nan(values: Iterable[float]) -> float:
    items = list(values)
    return mean(items) if items else math.nan


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
