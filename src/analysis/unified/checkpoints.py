"""定义评估任务并安全加载模型检查点。"""

# ====================
# 1. Imports
# ====================


from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode

from src.data.components.vctk_splits import UnseenSpeakerSentenceSplit

# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================


@dataclass(frozen=True)
class EvaluationSpec:
    """定义一个 checkpoint 与 NFE 组合。"""

    experiment_id: str
    model: str
    nfe: int
    run_dir: Path


CHECKPOINT_SAFE_GLOBALS = (
    UnseenSpeakerSentenceSplit,
    ListConfig,
    ContainerMetadata,
    Any,
    list,
    defaultdict,
    dict,
    int,
    AnyNode,
    Metadata,
)


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
# 4. Internal Functions
# 内部业务实现
# ====================


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
    with torch.serialization.safe_globals(list(CHECKPOINT_SAFE_GLOBALS)):
        checkpoint = torch.load(_best_checkpoint(run_dir), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError(f"无效 Lightning checkpoint: {run_dir}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    return model, config
