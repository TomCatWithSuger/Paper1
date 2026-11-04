"""模块说明：提供离线评估报告测试的合成模型、日志资源和显式共享夹具。"""

# ====================
# 1. Imports
# ====================

import csv
import hashlib
import json
import math
import socket
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from torch.utils.tensorboard.writer import SummaryWriter

from src.analysis import baseline_validation as baseline
from src.analysis import unified_vc_evaluation as unified
from src.analysis import vocoder_evaluation as vocoder
from src.analysis.vocoder import audio as vocoder_audio
from src.analysis.vocoder import models as vocoder_models

# ====================
# 3. Utilities
# ====================


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(path: Path, label, config):
    (path / ".hydra").mkdir(parents=True)
    (path / ".hydra" / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    (path / "tags.log").write_text(repr([baseline.RUN_LABEL_TAGS[label]]), encoding="utf-8")
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "last.ckpt").touch()
    (path / "checkpoints" / "epoch_002.ckpt").touch()
    writer = SummaryWriter(str(path / "tensorboard" / "version_0"))
    try:
        for tag, values in {
            "train/loss": [8.0, 4.0, 5.0],
            "val/loss": [6.0, 2.0, 3.0],
            "test/loss": [1.5],
        }.items():
            for step, value in enumerate(values):
                writer.add_scalar(tag, value, global_step=step)
    finally:
        writer.close()


class _SyntheticMethod:
    """提供可核验的模型边界，保留真实数据处理与评估流程。"""

    def __init__(self, requires_source_mels, offset):
        self.requires_source_mels = requires_source_mels
        self.offset = offset
        self.calls = []

    def encode_condition(self, content, lengths, speaker, mask):
        assert content.shape[:2] == (1, 3)
        assert lengths.tolist() == [5]
        assert speaker.shape == (1, 2)
        return torch.full((1, 2, mask.size(1)), math.log(1e-5) + self.offset)

    def sample(self, condition, mask, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert mask.all()
        assert ("source_mels" in kwargs) == self.requires_source_mels
        if self.requires_source_mels:
            assert kwargs["source_mels"].shape == condition.shape
        self.calls.append((torch.initial_seed(), kwargs["integration_steps"]))
        return condition


@pytest.fixture(name="offline_boundaries", autouse=True)
def offline_boundaries(monkeypatch):
    """禁止测试联网或意外进入真实模型加载。"""

    def forbidden(*args, **kwargs):
        pytest.fail("测试不得联网或加载真实模型")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(unified, "_load_model", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture(name="evaluation_data")
def evaluation_data(tmp_path):
    """构造三个实验、八条语音及四条固定测试配对。"""
    config = {
        "seed": 7,
        "data": {
            "microphone": "mic1",
            "sample_rate": 8000,
            "max_duration_seconds": 1.0,
            "n_fft": 32,
            "win_length": 32,
            "hop_length": 16,
            "n_mels": 2,
            "f_min": 0.0,
            "f_max": 4000.0,
            "log_mel_floor": 1e-5,
            "test_reference_mode": "different_speaker",
            "split_strategy": {
                "evaluation_speaker_ids": ["p200", "p300"],
                "evaluation_sentence_ids": ["005", "006"],
                "val_ratio": 0.5,
                "seed": 7,
            },
        },
        "model": {"method": {"condition_encoder": {"content_dim": 3}}},
    }
    runs = tmp_path / "runs"
    for label in baseline.RUN_LABEL_TAGS:
        _write_run(runs / f"2026-02-{label[-1]}_{label}", label, config)
    _write_run(runs / "2026-01-01_old_B1", "B1", {**config, "seed": 999})
    (runs / "unrelated").mkdir()
    (runs / "notes.txt").write_text("忽略非实验文件", encoding="utf-8")
    root = tmp_path / "data" / "VCTK"
    cache = tmp_path / "cache"
    for kind in ("content", "speaker"):
        (cache / kind).mkdir(parents=True)
    for speaker, sentences in (
        ("p100", range(1, 5)),
        ("p200", (5, 6)),
        ("p300", (5, 6)),
    ):
        for sentence in sentences:
            utterance = f"{speaker}_{sentence:03d}"
            text_dir = root / "txt" / speaker
            audio_dir = root / "wav48_silence_trimmed" / speaker
            text_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / f"{utterance}.txt").write_text(
                f"测试，句子 {sentence}\n", encoding="utf-8"
            )
            sf.write(audio_dir / f"{utterance}_mic1.wav", np.zeros(64 + 16 * sentence), 8000)
            torch.save(torch.ones(3, 5), cache / "content" / f"{utterance}.pt")
            torch.save(torch.tensor([1.0, 2.0]), cache / "speaker" / f"{utterance}.pt")
    return SimpleNamespace(
        root=root, runs=runs, cache=cache, config=config, output=tmp_path / "audit"
    )


@pytest.fixture(name="audit")
def audit(evaluation_data):
    paths = baseline.analyze_baselines(
        evaluation_data.runs, evaluation_data.root.parent, evaluation_data.output
    )
    return SimpleNamespace(**vars(evaluation_data), paths=paths)


@pytest.fixture(name="model_boundary")
def model_boundary(monkeypatch):
    loaded = []

    def load(run_dir, device):
        config = unified._model_config(run_dir)
        label = run_dir.name[-2:]
        method = _SyntheticMethod(label == "B3", {"B1": 1.0, "B2": 2.0, "B3": 3.0}[label])
        loaded.append((run_dir, device, method))
        return SimpleNamespace(method=method), config

    monkeypatch.setattr(unified, "_load_model", load)
    return loaded


def _evaluate(audit, output, **kwargs):
    return unified.evaluate_unified(
        audit.paths["plan"],
        audit.paths["manifest"],
        audit.root,
        audit.cache,
        output,
        **kwargs,
    )


def _write_identity(audit, output, experiment_ids, base_seed=12345, device="cpu"):
    """为预置账本构造与 _execute 一致的评估身份文件。"""
    selected = set(experiment_ids)
    specs = [
        spec for spec in unified._load_specs(audit.paths["plan"]) if spec.experiment_id in selected
    ]
    rows = unified._read_csv(audit.paths["manifest"])
    identity = unified._build_run_identity(
        specs,
        audit.paths["manifest"],
        rows,
        audit.cache,
        unified._model_config(specs[0].run_dir),
        device,
        base_seed,
    )
    (output / "evaluation_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False), encoding="utf-8"
    )


def _run_dir_for(audit, experiment_id):
    for spec in unified._load_specs(audit.paths["plan"]):
        if spec.experiment_id == experiment_id:
            return spec.run_dir
    raise AssertionError(experiment_id)
