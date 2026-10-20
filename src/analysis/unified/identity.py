"""计算评估身份并核验断点续跑条件。"""

# ====================
# 1. Imports
# ====================


import hashlib
import json
import shutil
import subprocess  # nosec B404 - 仅执行固定参数的本地版本控制命令
from pathlib import Path
from typing import Any, Sequence

from src.analysis.unified.checkpoints import (
    EvaluationSpec,
    _best_checkpoint,
    _model_config,
)

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(run_dir: Path) -> str:
    return _sha256_file(_best_checkpoint(run_dir))


def _feature_cache_fingerprint(manifest: Sequence[dict[str, str]], feature_cache_dir: Path) -> str:
    digest = hashlib.sha256()
    utterances = sorted(
        {
            utterance_id
            for row in manifest
            for utterance_id in (row["source_id"], row["reference_id"])
        }
    )
    for utterance_id in utterances:
        for kind in ("content", "speaker"):
            path = feature_cache_dir / kind / f"{utterance_id}.pt"
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _source_version() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        return "unknown"
    try:
        git_executable = str(Path(git_executable).resolve())
        commit = subprocess.run(
            [git_executable, "rev-parse", "HEAD"],
            shell=False,  # nosec B603 - 绝对路径、固定参数且禁用命令解释器
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            [git_executable, "status", "--porcelain"],
            shell=False,  # nosec B603 - 绝对路径、固定参数且禁用命令解释器
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _build_run_identity(
    specs: Sequence[EvaluationSpec],
    manifest_path: Path,
    manifest: Sequence[dict[str, str]],
    feature_cache_dir: Path,
    first_config: dict[str, Any],
    device_name: str,
    base_seed: int,
) -> dict[str, Any]:
    data = first_config["data"]
    experiments = {
        spec.experiment_id: {
            "model": spec.model,
            "nfe": spec.nfe,
            "checkpoint_sha256": _checkpoint_sha256(spec.run_dir),
            "config_sha256": hashlib.sha256(
                json.dumps(_model_config(spec.run_dir), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        for spec in specs
    }
    return {
        "source_version": _source_version(),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_seed": base_seed,
        "timing_device": device_name,
        "timing_scope": "cached_condition_to_mel",
        "mel_params": {
            "sample_rate": int(data["sample_rate"]),
            "n_fft": int(data["n_fft"]),
            "win_length": int(data["win_length"]),
            "hop_length": int(data["hop_length"]),
            "n_mels": int(data["n_mels"]),
            "f_min": float(data["f_min"]),
            "f_max": float(data["f_max"]),
            "log_mel_floor": float(data["log_mel_floor"]),
        },
        "feature_cache_fingerprint": _feature_cache_fingerprint(manifest, feature_cache_dir),
        "experiments": experiments,
    }


def _identity_mismatch(existing: dict[str, Any], current: dict[str, Any]) -> list[str]:
    differences = [
        key
        for key in sorted(set(existing) | set(current))
        if key != "experiments" and existing.get(key) != current.get(key)
    ]
    existing_experiments = existing.get("experiments", {})
    current_experiments = current.get("experiments", {})
    for experiment_id in sorted(current_experiments):
        if (
            experiment_id in existing_experiments
            and existing_experiments[experiment_id] != current_experiments[experiment_id]
        ):
            differences.append(f"experiments.{experiment_id}")
    return differences


def _load_identity(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效评估身份文件: {path}")
    return value


def _write_identity(path: Path, identity: dict[str, Any]) -> None:
    path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
