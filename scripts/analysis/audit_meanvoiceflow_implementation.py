"""模块说明：

- 直接提取本地论文文本，保存实际训练配置、历史指标和检查点审计。
- 用已有 B3 权重复现条件 padding 问题并核对修复，不训练、不生成音频。
"""

# ====================
# 1. Imports
# ====================

import argparse
import csv
import hashlib
import json
import shutil
import subprocess  # nosec B404 - 仅用参数数组调用本地论文文本提取工具。
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.unified.checkpoints import CHECKPOINT_SAFE_GLOBALS
from src.models.components.conditioning.meanvc import CachedConditionEncoder

RUN = ROOT / "logs/train/runs/2026-10-21_14-29-28"


# ====================
# 3. Utilities
# 通用工具函数
# ====================


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint(path):
    with torch.serialization.safe_globals(list(CHECKPOINT_SAFE_GLOBALS)):
        return torch.load(path, map_location="cpu", weights_only=True)


# ====================
# 4. Internal Functions
# 内部业务 / 功能实现
# ====================


def _padding_probe(payload, row):
    parameters = {
        key.split("condition_encoder.", 1)[1]: value
        for key, value in payload["state_dict"].items()
        if "condition_encoder." in key
    }
    cache = ROOT / "data/meanvc_features/vctk"
    content = torch.load(cache / "content" / f"{row['source_id']}.pt", weights_only=True).float()[
        None
    ]
    speaker = torch.load(
        cache / "speaker" / f"{row['reference_id']}.pt", weights_only=True
    ).float()[None]
    frames = torch.load(row["generated_path"], weights_only=True)["mel"].shape[-1]
    lengths = torch.tensor([content.shape[-1]])
    mask = torch.ones(1, frames, dtype=torch.bool)
    padded = F.pad(mask, (0, frames), value=False)
    versions = {}
    for name, enabled in [("legacy", False), ("masked", True)]:
        encoder = CachedConditionEncoder(mask_padding_in_normalization=enabled).eval()
        encoder.load_state_dict(parameters, strict=True)
        with torch.no_grad():
            plain = encoder(content, lengths, speaker, mask)
            longer = encoder(content, lengths, speaker, padded)[:, :, :frames]
        versions[name] = (plain, longer)
    result = {
        "source": row["source_id"],
        "reference": row["reference_id"],
        "valid_frames": frames,
        "added_padding_frames": frames,
    }
    for name, (plain, padded_value) in versions.items():
        difference = plain - padded_value
        result[name] = {
            "max_abs_diff": float(difference.abs().max()),
            "rms_diff": float(difference.square().mean().sqrt()),
        }
    result["unpadded_legacy_and_fixed_bitwise_equal"] = torch.equal(
        versions["legacy"][0], versions["masked"][0]
    )
    assert result["legacy"]["max_abs_diff"] > 0.01
    assert result["masked"]["max_abs_diff"] < 1e-5
    assert result["unpadded_legacy_and_fixed_bitwise_equal"]
    return result


# ====================
# 5. Core Logic
# 核心流程
# ====================


def _execute(output):
    paper = next((ROOT / "docs/papers").glob("*MeanVoiceFlow*.pdf"))
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("缺少本地 pdftotext，不会自动安装或下载")
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(  # nosec B603 - 已解析本地程序路径，独立参数不经过命令解释器。
        [executable, "-layout", str(paper), str(output / "paper.txt")], check=True
    )
    config = OmegaConf.load(RUN / ".hydra/config.yaml")
    quality_path = ROOT / "analysis_results/p1_latest_20261024/full/aggregate_metrics.csv"
    latency_path = ROOT / "analysis_results/p1_latest_20261024/latency_interleaved/aggregate.csv"
    quality = {row["experiment_id"]: row for row in _rows(quality_path)}
    latency = {row["experiment_id"]: row for row in _rows(latency_path)}
    best_path = RUN / "checkpoints/epoch_040.ckpt"
    last_path = RUN / "checkpoints/last.ckpt"
    best = _checkpoint(best_path)
    row = next(
        row
        for row in _rows(
            ROOT / "analysis_results/p2_mechanisms_20261024/conversion/per_sample_enriched.csv"
        )
        if row["experiment_id"] == "b3_best_s095"
    )
    probe = _padding_probe(best, row)
    checkpoints = {
        "best": {
            "epoch": best["epoch"],
            "global_step": best["global_step"],
            "sha256": _hash(best_path),
        }
    }
    del best
    last = _checkpoint(last_path)
    checkpoints["last"] = {
        "epoch": last["epoch"],
        "global_step": last["global_step"],
        "sha256": _hash(last_path),
    }
    selected = ["nfe_b1_01", "nfe_b1_50", "nfe_b3_01"]
    result = {
        "paper": str(paper),
        "paper_sha256": _hash(paper),
        "extraction": "pdftotext -layout；本地文本层，无OCR或视觉模型",
        "run_config_sha256": _hash(RUN / ".hydra/config.yaml"),
        "training_method": OmegaConf.to_container(config.model.method),
        "training_optimizer": OmegaConf.to_container(config.model.optimizer),
        "historical_padding_flag": config.model.method.condition_encoder.get(
            "mask_padding_in_normalization", False
        ),
        "checkpoint_audit": checkpoints,
        "padding_probe": probe,
        "recorded_p1": {
            key: {
                "cer": float(quality[key]["cer_mean"]),
                "target_similarity": float(quality[key]["target_speaker_similarity_mean"]),
                "interleaved_latency_ms": float(latency[key]["mean_ms"]),
            }
            for key in selected
        },
        "quality_input_sha256": _hash(quality_path),
        "latency_input_sha256": _hash(latency_path),
        "limitations": "没有新训练或重新测质量；padding修复的质量收益未知；历史权重继续按原配置加载",
    }
    sources = [
        Path(__file__),
        ROOT / "src/models/components/conditioning/meanvc.py",
        ROOT / "src/models/methods/generative/flow/mean_flow/meanvoiceflow.py",
    ]
    result["source_sha256"] = {str(path): _hash(path) for path in sources}
    (output / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, output / "source" / path.name)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(probe, ensure_ascii=False, indent=2))
    print(output)


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def main():
    """解析审计输出目录，执行本地论文与已有权重的一致性检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _execute(args.output_dir.resolve())


# ====================
# 7. Entry Point
# 可选：直接运行该文件
# ====================

if __name__ == "__main__":
    main()
