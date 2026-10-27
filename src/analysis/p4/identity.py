"""模块说明：

- 核验 B3 三档源强度的完整配对，按说话人和性别分析身份与内容。
- 先在语句内平均噪声，再生成配对区间和不含配置标签的听测清单。
"""

# ====================
# 1. Imports
# ====================

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================

CONFIGS = ("b3_best_s080", "b3_best_s090", "b3_best_s095")
METRICS = (
    "cer",
    "target_speaker_similarity",
    "source_speaker_similarity",
    "target_preference_gap",
    "target_preferred",
)
STRATA = ("all", "source_speaker_id", "reference_speaker_id", "gender_pair", "length_bin")
SEED = 20261026


# ====================
# 3. Utilities
# 通用工具函数
# ====================


def read_csv(path):
    """读取本地 CSV 并保留原始字段，交由协议核验函数解析。"""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    """按首行字段顺序写入统一记录，不隐式丢弃额外字段。"""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    """流式计算本地文件指纹，用于模型及音频输入一致性核验。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speaker_genders(path):
    """读取 VCTK 明示的 M/F 元数据，不从声音或编号推断。"""
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith("p") and fields[2] in ("M", "F"):
            result[fields[0]] = fields[2]
    return result


def paired_ci(values, metadata):
    """按源说话人和实际文本乘积重采样；不把重复噪声当独立语句。"""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    weights = np.ones((2000, len(metadata)), dtype=float)
    for field in ("source_speaker_id", "source_text"):
        levels = sorted({r[field] for r in metadata})
        index = [levels.index(r[field]) for r in metadata]
        draws = rng.multinomial(len(levels), np.full(len(levels), 1 / len(levels)), 2000)
        weights *= draws[:, index]
    denominator = weights.sum(axis=1)
    valid = denominator > 0
    samples = (weights[valid] @ values) / denominator[valid]
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


# ====================
# 4. Internal Functions
# 内部业务 / 功能实现
# ====================


def validate_rows(rows, genders, expected_sources=64):
    """核验三档配置、两噪声及元数据完整配对，并计算身份偏好。"""
    selected = [dict(row) for row in rows if row["experiment_id"] in CONFIGS]
    indexed = {}
    for row in selected:
        repeat_value = float(row["repeat"])
        if repeat_value not in (0, 1):
            raise ValueError("重复噪声编号必须为0或1")
        repeat = int(repeat_value)
        key = (row["experiment_id"], row["source_id"], repeat)
        if key in indexed or repeat not in (0, 1):
            raise ValueError(f"重复或非法噪声配对：{key}")
        row["repeat"] = repeat
        expected_time = dict(zip(CONFIGS, (0.8, 0.9, 0.95)))[row["experiment_id"]]
        if float(row["source_time"]) != expected_time:
            raise ValueError(f"配置与source_time不匹配：{key}")
        if row["source_speaker_id"] == row["reference_speaker_id"]:
            raise ValueError("P4要求跨说话人转换配对")
        for metric in METRICS[:3]:
            row[metric] = float(row[metric])
            if not np.isfinite(row[metric]):
                raise ValueError(f"评分缺失：{key}, {metric}")
        row["target_preference_gap"] = (
            row["target_speaker_similarity"] - row["source_speaker_similarity"]
        )
        row["target_preferred"] = float(row["target_preference_gap"] > 0)
        row["gender_pair"] = (
            genders[row["source_speaker_id"]] + "→" + genders[row["reference_speaker_id"]]
        )
        if not Path(row["wav_path"]).is_file():
            raise FileNotFoundError(row["wav_path"])
        indexed[key] = row
    ids = sorted({row["source_id"] for row in selected})
    if len(ids) != expected_sources or len(indexed) != expected_sources * 6:
        raise ValueError("三档配置必须各含完整源语句及两个噪声")
    metadata_fields = (
        "reference_id",
        "source_speaker_id",
        "reference_speaker_id",
        "source_text",
        "normalized_reference",
        "donor_id",
        "length_bin",
        "partition",
    )
    for uid in ids:
        anchor = indexed[(CONFIGS[0], uid, 0)]
        if float(anchor["seed"]) == float(indexed[(CONFIGS[0], uid, 1)]["seed"]):
            raise ValueError("两个噪声必须使用不同种子")
        for config in CONFIGS:
            for repeat in (0, 1):
                row = indexed[(config, uid, repeat)]
                if any(row[field] != anchor[field] for field in metadata_fields):
                    raise ValueError(f"配对元数据不一致：{uid}")
                if float(row["seed"]) != float(indexed[(CONFIGS[0], uid, repeat)]["seed"]):
                    raise ValueError(f"配置间噪声种子不一致：{uid}")
    return selected


def source_means(rows):
    """先在每个源语句内平均两个噪声，避免扩大独立样本数。"""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment_id"], row["source_id"])].append(row)
    result = []
    for group in groups.values():
        row = {
            key: group[0][key]
            for key in (
                "experiment_id",
                "source_id",
                "source_speaker_id",
                "reference_speaker_id",
                "source_text",
                "gender_pair",
                "length_bin",
            )
        }
        row.update({metric: float(np.mean([r[metric] for r in group])) for metric in METRICS})
        result.append(row)
    return result


def statistics(rows):
    """按说话人、性别组合和长度汇总指标并计算配对变化区间。"""
    means = source_means(rows)
    summaries, comparisons = [], []
    indexed = {(r["experiment_id"], r["source_id"]): r for r in means}
    for field in STRATA:
        levels = ["all"] if field == "all" else sorted({r[field] for r in means})
        for level in levels:
            group = [r for r in means if field == "all" or r[field] == level]
            for config in CONFIGS:
                subset = [r for r in group if r["experiment_id"] == config]
                for metric in METRICS:
                    summaries.append(
                        {
                            "stratum": field,
                            "level": level,
                            "experiment_id": config,
                            "metric": metric,
                            "n_sources": len(subset),
                            "mean": float(np.mean([r[metric] for r in subset])),
                        }
                    )
            for a, b in (
                (CONFIGS[0], CONFIGS[2]),
                (CONFIGS[1], CONFIGS[2]),
                (CONFIGS[0], CONFIGS[1]),
            ):
                metadata = [r for r in group if r["experiment_id"] == a]
                for metric in METRICS:
                    values = [r[metric] - indexed[(b, r["source_id"])][metric] for r in metadata]
                    lo, hi = paired_ci(values, metadata)
                    comparisons.append(
                        {
                            "stratum": field,
                            "level": level,
                            "config_a": a,
                            "config_b": b,
                            "metric": metric,
                            "n_sources": len(values),
                            "n_source_speakers": len({r["source_speaker_id"] for r in metadata}),
                            "n_target_speakers": len(
                                {r["reference_speaker_id"] for r in metadata}
                            ),
                            "difference": float(np.mean(values)),
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "fraction_delta_positive": float(np.mean(np.asarray(values) > 0)),
                            "fraction_delta_negative": float(np.mean(np.asarray(values) < 0)),
                        }
                    )
    return summaries, comparisons


def original_path(dataset, uid):
    """定位指定语句的 VCTK 原始 mic1 音频，缺失时明确报错。"""
    path = Path(dataset) / "wav48_silence_trimmed" / uid.split("_")[0] / f"{uid}_mic1.flac"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def listening_manifest(rows, dataset, output):
    """每个性别组合最多四句，固定噪声0；组织者密钥与受试者表分开。"""
    rng = np.random.default_rng(SEED)
    groups = defaultdict(list)
    for row in rows:
        if row["experiment_id"] == CONFIGS[0] and row["repeat"] == 0:
            groups[row["gender_pair"]].append(row)
    indexed = {(r["experiment_id"], r["source_id"], r["repeat"]): r for r in rows}
    public, private = [], []
    assets = output / "listening" / "audio"
    assets.mkdir(parents=True)
    selected = []
    for gender in sorted(groups):
        candidates = sorted(groups[gender], key=lambda r: r["source_id"])
        selected.extend(candidates[i] for i in rng.permutation(len(candidates))[:4])
    for trial, index in enumerate(rng.permutation(len(selected)), 1):
        row = selected[index]
        # 盲听资产使用匿名文件名；原始路径仅出现在组织者表。
        for role, uid in [("source", row["source_id"]), ("target", row["reference_id"])]:
            shutil.copyfile(original_path(dataset, uid), assets / f"{trial:02d}_{role}.flac")
        for label, config in zip("ABC", rng.permutation(CONFIGS)):
            item = indexed[(config, row["source_id"], 0)]
            name = f"{trial:02d}_{label}.wav"
            shutil.copyfile(item["wav_path"], assets / name)
            public.append(
                {
                    "trial": trial,
                    "candidate": label,
                    "audio": f"audio/{name}",
                    "source": f"audio/{trial:02d}_source.flac",
                    "target": f"audio/{trial:02d}_target.flac",
                    "transcription": "",
                    "target_similarity_1_to_5": "",
                    "source_similarity_1_to_5": "",
                    "naturalness_1_to_5": "",
                }
            )
            private.append(
                {
                    "trial": trial,
                    "candidate": label,
                    "experiment_id": config,
                    "source_id": row["source_id"],
                    "reference_id": row["reference_id"],
                    "gender_pair": row["gender_pair"],
                    "repeat": 0,
                    "source_text": row["source_text"],
                }
            )
    write_csv(output / "listening/participant.csv", public)
    write_csv(output / "listening/organizer_key.csv", private)


# ====================
# 5. Core Logic
# 核心流程
# ====================


def analyze(input_csv, speaker_info, dataset, output):
    """核验已有评分和波形，保存输入指纹、分层统计及匿名盲听材料。"""
    rows = validate_rows(read_csv(input_csv), speaker_genders(speaker_info))
    # 全部输入先核验，避免用缺失波形的样本形成选择偏差。
    audio = {Path(row["wav_path"]) for row in rows}
    audio.update(
        original_path(dataset, row[field])
        for row in rows
        for field in ("source_id", "reference_id", "donor_id")
    )
    protocol = {
        "input_csv": str(input_csv.resolve()),
        "input_sha256": sha256(input_csv),
        "speaker_info": str(speaker_info.resolve()),
        "speaker_info_sha256": sha256(speaker_info),
        "dataset": str(dataset.resolve()),
        "configs": CONFIGS,
        "n_outputs": len(rows),
        "n_sources": 64,
        "seed": SEED,
        "bootstrap": "2000次源说话人×实际文本乘积重采样，先平均两个噪声；目标说话人分层为描述性分析；未校正多重比较",
        "independent_scoring": "pending",
        "listening": "prepared_not_performed",
        "audio_sha256": {str(path.resolve()): sha256(path) for path in sorted(audio)},
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    project = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__),
        Path(__file__).with_name("scoring.py"),
        project / "scripts/analysis/evaluate_p4_identity.py",
    ]
    protocol["code_sha256"] = {str(path): sha256(path) for path in sources}
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(output / "existing_per_sample.csv", rows)
    summaries, comparisons = statistics(rows)
    write_csv(output / "existing_summary.csv", summaries)
    write_csv(output / "existing_paired.csv", comparisons)
    listening_manifest(rows, dataset, output)
    return protocol


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def run(input_csv, speaker_info, dataset, output):
    """对已有转换结果执行 P4 分层诊断，使用全新的输出目录。"""
    return analyze(Path(input_csv), Path(speaker_info), Path(dataset), Path(output))
