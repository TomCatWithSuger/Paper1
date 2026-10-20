"""读写波形评估记录并聚合指标、生成报告。"""

# ====================
# 1. Imports
# ====================


import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from src.analysis.vocoder.definitions import (
    ENERGY_HOP,
    ENERGY_WIN,
    F0_FRAME_TIME,
    F0_FREQ_HIGH,
    F0_FREQ_LOW,
)

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _mean_or_nan(values: Sequence[float]) -> float:
    """计算有限值均值，无有效值时返回 NaN。"""
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    """读取统一评估记录（jsonl）。"""
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _append_ledger(path: Path, record: dict[str, Any]) -> None:
    """追加写入一条评估记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按实验聚合波形指标。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["experiment_id"]), []).append(row)

    aggregates = []
    for experiment_id, items in grouped.items():
        first = items[0]
        aggregates.append(
            {
                "experiment_id": experiment_id,
                "model": first["model"],
                "nfe": first["nfe"],
                "sample_count": len(items),
                "wer_mean": _mean_or_nan([float(item["wer"]) for item in items]),
                "cer_mean": _mean_or_nan([float(item["cer"]) for item in items]),
                "spk_sim_mean": _mean_or_nan([float(item["spk_sim"]) for item in items]),
                "f0_corr_mean": _mean_or_nan([float(item["f0_corr"]) for item in items]),
                "f0_log_mae_mean": _mean_or_nan([float(item["f0_log_mae"]) for item in items]),
                "energy_corr_mean": _mean_or_nan([float(item["energy_corr"]) for item in items]),
                "energy_log_mae_mean": _mean_or_nan(
                    [float(item["energy_log_mae"]) for item in items]
                ),
                "vocoder_latency_ms_mean": _mean_or_nan(
                    [float(item["vocoder_latency_ms"]) for item in items]
                ),
                "vocoder_rtf_mean": _mean_or_nan([float(item["vocoder_rtf"]) for item in items]),
                "e2e_latency_ms_mean": _mean_or_nan(
                    [float(item["e2e_latency_ms"]) for item in items]
                ),
                "e2e_rtf_mean": _mean_or_nan([float(item["e2e_rtf"]) for item in items]),
            }
        )
    aggregates.sort(key=lambda row: (str(row["model"]), float(row["nfe"])))
    return aggregates


def _report(
    aggregates: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
    device: str,
) -> str:
    """生成波形域评估报告。"""
    lines = [
        "# Vocoder 波形域评估报告",
        "",
        "## 1. 设置",
        "",
        "- 声码器：BSC-LT/vocos-mel-22khz（冻结预训练 Vocos，80 维 Mel→22.05kHz，对所有系统统一使用）。",
        "- Mel 适配：log 功率 Mel ÷2 转对数幅度，并经 HTK→slaney 频带线性插值后送入 Vocos。",
        "- WER/CER：openai/whisper-small（英文转写，文本归一化后按词/字符编辑距离计算）。",
        "- 说话人相似度：MeanVC 冻结 WavLM Large + ECAPA-TDNN 余弦相似度（目标为参考音频）。",
        f"- F0：torchaudio detect_pitch_frequency（{F0_FREQ_LOW}-{F0_FREQ_HIGH}Hz，帧长 {F0_FRAME_TIME}s，双侧有声帧 log-F0 相关与 MAE）。",
        f"- 能量：{ENERGY_WIN}/{ENERGY_HOP} 分帧 log-RMS 的相关与 MAE。",
        "- RTF：mel 生成延迟（统一评估记录）+ 声码器解码延迟，除以音频时长；声码器 RTF 单独列出。",
        f"- 设备：{device}。",
        "",
        "## 2. 重建对照（声码器上限）",
        "",
        "真实参考 Mel 经同一适配链路与 Vocos 重建后的指标（评估声码器+适配链路本身）：",
        "",
        "| 指标 | 参考音频(ASR) | 重建音频 |",
        "|---|---|---|",
    ]
    wer_orig = _mean_or_nan([float(row["wer_original"]) for row in control_rows])
    wer_recon = _mean_or_nan([float(row["wer_recon"]) for row in control_rows])
    cer_orig = _mean_or_nan([float(row["cer_original"]) for row in control_rows])
    cer_recon = _mean_or_nan([float(row["cer_recon"]) for row in control_rows])
    spk = _mean_or_nan([float(row["spk_sim"]) for row in control_rows])
    f0c = _mean_or_nan([float(row["f0_corr"]) for row in control_rows])
    ec = _mean_or_nan([float(row["energy_corr"]) for row in control_rows])
    lines.append(f"| WER | {wer_orig:.4f} | {wer_recon:.4f} |")
    lines.append(f"| CER | {cer_orig:.4f} | {cer_recon:.4f} |")
    lines.append(f"| 说话人相似度 | 1.0000 | {spk:.4f} |")
    lines.append(f"| F0 相关 | 1.0000 | {f0c:.4f} |")
    lines.append(f"| 能量相关 | 1.0000 | {ec:.4f} |")

    lines += [
        "",
        "## 3. 各系统波形指标",
        "",
        "| 实验 | 模型 | NFE | WER | CER | 说话人相似度 | F0 相关 | F0 log-MAE | 能量相关 | 声码器 RTF | 端到端 RTF |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['experiment_id']} | {row['model']} | {row['nfe']} "
            f"| {float(row['wer_mean']):.4f} | {float(row['cer_mean']):.4f} "
            f"| {float(row['spk_sim_mean']):.4f} | {float(row['f0_corr_mean']):.4f} "
            f"| {float(row['f0_log_mae_mean']):.4f} | {float(row['energy_corr_mean']):.4f} "
            f"| {float(row['vocoder_rtf_mean']):.6f} | {float(row['e2e_rtf_mean']):.6f} |"
        )
    lines += [
        "",
        "## 4. 说明与边界",
        "",
        "- 所有系统共享同一冻结 Vocos 声码器与同一 Mel 适配，系统间差异仅来自生成的 Mel。",
        "- WER/CER 反映可懂度与内容保持；说话人相似度反映音色迁移；F0/能量轮廓反映韵律保真。",
        "- 重建对照量化了声码器链路的上限，生成结果指标应对照该上限解读。",
        "- RTF 不含离线 MeanVC 特征提取（与统一评估口径一致）。",
        "",
    ]
    return "\n".join(lines)
