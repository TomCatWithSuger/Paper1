"""模块说明：验证源码身份及离线 ASR、声码器资源边界，不加载真实模型。"""

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
from src.analysis.unified import identity
from src.analysis.vocoder import audio as vocoder_audio
from src.analysis.vocoder import models as vocoder_models
from tests.helpers.evaluation_reports import _evaluate as _evaluate
from tests.helpers.evaluation_reports import _read_csv as _read_csv
from tests.helpers.evaluation_reports import _run_dir_for as _run_dir_for
from tests.helpers.evaluation_reports import _SyntheticMethod as _SyntheticMethod
from tests.helpers.evaluation_reports import _write_csv as _write_csv
from tests.helpers.evaluation_reports import _write_identity as _write_identity
from tests.helpers.evaluation_reports import _write_run as _write_run
from tests.helpers.evaluation_reports import audit as _audit_fixture
from tests.helpers.evaluation_reports import evaluation_data as _evaluation_data_fixture
from tests.helpers.evaluation_reports import model_boundary as _model_boundary_fixture
from tests.helpers.evaluation_reports import (
    offline_boundaries as _offline_boundaries_fixture,
)

# ====================
# 6. Public API
# ====================


@pytest.mark.parametrize("dirty", ["", " M changed.py"])
def test_source_version_uses_fixed_git_commands(monkeypatch, tmp_path, dirty):
    executable = tmp_path / "git"
    calls = []
    monkeypatch.setattr(identity.shutil, "which", lambda name: str(executable))

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="commit-id\n" if len(calls) == 1 else dirty)

    monkeypatch.setattr(identity.subprocess, "run", run)
    assert identity._source_version() == f"commit-id{'-dirty' if dirty else ''}"
    assert [command for command, _ in calls] == [
        [str(executable.resolve()), "rev-parse", "HEAD"],
        [str(executable.resolve()), "status", "--porcelain"],
    ]
    for _, kwargs in calls:
        assert kwargs == {
            "shell": False,
            "capture_output": True,
            "text": True,
            "check": True,
            "timeout": 5,
        }


def test_source_version_missing_git_does_not_start_process(monkeypatch):
    monkeypatch.setattr(identity.shutil, "which", lambda name: None)

    def forbidden(*args, **kwargs):
        pytest.fail("未找到可执行文件时不得启动进程")

    monkeypatch.setattr(identity.subprocess, "run", forbidden)
    assert identity._source_version() == "unknown"


@pytest.mark.parametrize(
    "error", [OSError("unavailable"), identity.subprocess.TimeoutExpired("git", 5)]
)
def test_source_version_handles_command_errors(monkeypatch, tmp_path, error):
    monkeypatch.setattr(identity.shutil, "which", lambda name: str(tmp_path / "git"))

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(identity.subprocess, "run", run)
    assert identity._source_version() == "unknown"


@pytest.mark.parametrize("existing_file", [False, True])
def test_asr_missing_local_directory_skips_metrics(tmp_path, existing_file, caplog):
    model_path = tmp_path / "whisper"
    if existing_file:
        model_path.touch()
    scorer = vocoder_models._AsrScorer(model_path, torch.device("cpu"))
    assert not scorer.available
    assert "跳过 WER/CER" in caplog.text
    assert str(model_path) in caplog.text
    scores = scorer.score(torch.zeros(100), "hello")
    assert math.isnan(scores["wer"]) and math.isnan(scores["cer"])
    assert scores["asr_hyp"] == ""
    assert scorer.transcribe(torch.zeros(100)) == ""


def test_asr_only_loads_local_models(monkeypatch, tmp_path):
    calls = []
    model = torch.nn.Linear(1, 1)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(normalizer=None))

    def load_processor(path, **kwargs):
        calls.append((path, kwargs))
        return processor

    def load_model(path, **kwargs):
        calls.append((path, kwargs))
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            WhisperProcessor=SimpleNamespace(from_pretrained=load_processor),
            WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    scorer = vocoder_models._AsrScorer(tmp_path, torch.device("cpu"))
    assert calls == [(str(tmp_path.resolve()), {"local_files_only": True})] * 2
    assert scorer.processor is processor
    assert scorer.model is model
    assert not model.training
    assert scorer.available


@pytest.mark.parametrize("failure_stage", ["processor", "model", "dependency"])
def test_asr_unavailable_local_resources_skip_metrics(
    monkeypatch, tmp_path, caplog, failure_stage
):
    def unavailable(*args, **kwargs):
        raise OSError("模型文件不完整")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        (
            None
            if failure_stage == "dependency"
            else SimpleNamespace(
                WhisperProcessor=SimpleNamespace(
                    from_pretrained=(
                        unavailable
                        if failure_stage == "processor"
                        else lambda *args, **kwargs: object()
                    )
                ),
                WhisperForConditionalGeneration=SimpleNamespace(from_pretrained=unavailable),
            )
        ),
    )
    scorer = vocoder_models._AsrScorer(tmp_path, torch.device("cpu"))
    assert not scorer.available
    assert "跳过 WER/CER" in caplog.text
    assert math.isnan(scorer.score(torch.zeros(100), "hello")["wer"])


def test_vocoder_without_whisper_continues_and_can_resume_asr(monkeypatch, tmp_path, caplog):
    waveform = torch.sin(torch.arange(2205) * (2 * math.pi * 440 / 22050)) * 0.25

    class LocalVocos(torch.nn.Module):
        def decode(self, mel):
            return waveform.unsqueeze(0)

    monkeypatch.setattr(vocoder, "_load_vocos", lambda *args: LocalVocos())
    monkeypatch.setattr(
        vocoder,
        "_SpeakerScorer",
        lambda *args: SimpleNamespace(
            embedding=lambda *args, **kwargs: torch.ones(2),
            similarity=lambda *args: 0.75,
        ),
    )
    monkeypatch.setattr(vocoder, "_load_audio", lambda *args: waveform)
    monkeypatch.setattr(
        vocoder,
        "discover_vctk_samples",
        lambda *args: [SimpleNamespace(utterance_id="reference", audio_path=tmp_path / "ref.wav")],
    )
    monkeypatch.setattr(
        vocoder,
        "_contour_metrics",
        lambda *args: {
            "f0_corr": 0.8,
            "f0_log_mae": 0.1,
            "energy_corr": 0.9,
            "energy_log_mae": 0.2,
        },
    )
    mel_path = tmp_path / "mel.pt"
    torch.save({"mel": torch.zeros(80, 9)}, mel_path)
    unified_dir = tmp_path / "unified"
    vocoder._append_ledger(
        unified_dir / "evaluation_records.jsonl",
        {
            "experiment_id": "b1",
            "model": "B1",
            "nfe": 1,
            "sample_index": 0,
            "source_id": "source",
            "reference_id": "reference",
            "audio_seconds": 0.1,
            "latency_ms": 2,
            "generated_path": str(mel_path),
        },
    )
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [{"source_id": "source", "reference_id": "reference", "text": "hello"}],
    )
    options = {
        "unified_dir": unified_dir,
        "manifest_path": manifest,
        "vctk_root": tmp_path,
        "vocos_config": tmp_path / "vocos.yaml",
        "vocos_weights": tmp_path / "vocos.bin",
        "asr_model_dir": tmp_path / "missing-whisper",
        "speaker_source": tmp_path / "speaker.py",
        "speaker_checkpoint": tmp_path / "speaker.pt",
        "wavlm_checkpoint": tmp_path / "wavlm.pt",
        "output_dir": tmp_path / "output",
        "device": "cpu",
    }
    summary = vocoder.evaluate_vocoder_metrics(**options)
    assert "跳过 WER/CER" in caplog.text
    assert math.isnan(summary["aggregates"][0]["wer_mean"])
    assert summary["aggregates"][0]["spk_sim_mean"] == 0.75
    assert "跳过 WER/CER" in summary["report"].read_text(encoding="utf-8")
    assert Path(_read_csv(summary["per_sample"])[0]["wav_path"]).is_file()
    assert math.isnan(float(_read_csv(summary["control"])[0]["wer_recon"]))
    vocoder.evaluate_vocoder_metrics(**options)
    ledger = options["output_dir"] / "evaluation_records.jsonl"
    assert len(vocoder._load_ledger(ledger)) == 1

    monkeypatch.setattr(
        vocoder,
        "_AsrScorer",
        lambda *args: SimpleNamespace(
            available=True,
            score=lambda *args: {"wer": 0.25, "cer": 0.125, "asr_hyp": "hello"},
        ),
    )
    summary = vocoder.evaluate_vocoder_metrics(**options)
    assert summary["aggregates"][0]["sample_count"] == 1
    assert summary["aggregates"][0]["wer_mean"] == 0.25
    assert len(_read_csv(summary["per_sample"])) == 1
    assert len(vocoder._load_ledger(ledger)) == 2
    vocoder.evaluate_vocoder_metrics(**options)
    assert len(vocoder._load_ledger(ledger)) == 2


def test_vocoder_audio_roundtrip_and_mel_adapter(tmp_path):
    waveform = torch.sin(torch.arange(2205) * (2 * math.pi * 440 / 22050)) * 0.25
    path = tmp_path / "audio" / "sample.wav"
    vocoder_audio._save_waveform(path, waveform)
    restored = vocoder_audio._load_audio(path)
    assert restored.shape == waveform.shape
    torch.testing.assert_close(restored, waveform, rtol=0, atol=1 / 32768)

    filterbank = vocoder_audio._ours_mel_filterbank()
    matrix = vocoder_audio._build_mel_adapter()
    assert filterbank.shape == (80, 513)
    assert matrix.shape == (80, 80)
    assert (filterbank >= 0).all() and (matrix >= 0).all()
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(80))

    mel = vocoder_audio._project_log_mel(restored, filterbank)
    adapted = vocoder_audio._adapt_log_mel(mel, matrix)
    assert adapted.shape == mel.shape == (80, waveform.numel() // 256 + 1)
    assert torch.isfinite(adapted).all()
    torch.testing.assert_close(adapted, (matrix @ (mel * 0.5)).clamp_min(math.log(1e-7)))

    silence_mel = vocoder_audio._project_log_mel(torch.zeros_like(waveform), filterbank)
    torch.testing.assert_close(silence_mel, torch.full_like(silence_mel, math.log(1e-5)))
    clamped = vocoder_audio._adapt_log_mel(torch.full((80, 1), -1000.0), matrix)
    torch.testing.assert_close(clamped, torch.full((80, 1), math.log(1e-7)))
