"""模块说明：用本地合成资源验证 P1 预检、完整评分、续跑和配对比较。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.analysis.p1 import comparisons, evaluation
from src.analysis.p4.identity import read_csv
from tests.helpers.diagnostic_models import (
    AnalyticModel,
    LocalAsr,
    LocalSpeaker,
    LocalVocos,
    diagnostic_batch,
)

pytestmark = pytest.mark.usefixtures("legacy_text_encoding")


@pytest.fixture()
def p1_runtime(tmp_path, monkeypatch):
    """只替换模型、数据加载和 GPU 边界，保留评估编排与文件读写。"""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    script = root / "scripts/analysis/evaluate_p1_latest.py"
    script.parent.mkdir(parents=True)
    script.write_text("# 本地源码快照\n", encoding="utf-8")
    (root / "src/example.py").write_text("# 本地源码快照\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "preflight.json").write_text("{}", encoding="utf-8")
    (output / "test_manifest.csv").write_text("fixture", encoding="utf-8")
    manifest = [
        {
            "index": 0,
            "source_id": "p1_001",
            "reference_id": "p2_001",
            "text": "hello world",
            "source_speaker_id": "p1",
            "sentence_id": "001",
        }
    ]
    by_id = {uid: SimpleNamespace(text="hello world") for uid in ("p1_001", "p2_001")}
    monkeypatch.setattr(evaluation, "ROOT", root)
    monkeypatch.setattr(
        evaluation, "_gpu_state", lambda path: path.write_text("CPU fixture", encoding="utf-8")
    )
    monkeypatch.setattr(evaluation, "_load_manifest_samples", lambda *args: (manifest, by_id))
    monkeypatch.setattr(evaluation, "_model_config", lambda *args: {"data": {}})
    monkeypatch.setattr(
        evaluation,
        "_dataset",
        lambda *args: [{"waveform": torch.sin(torch.arange(2205) * 0.1) * 0.1} for _ in range(2)],
    )
    monkeypatch.setattr(evaluation, "_AsrScorer", LocalAsr)
    speaker = LocalSpeaker()
    monkeypatch.setattr(evaluation, "_SpeakerScorer", lambda *args: speaker)
    weights = root / "pretrained/meanvc/wavlm_large_finetune.pth"
    weights.parent.mkdir(parents=True)
    torch.save({"model": speaker.encoder.model.state_dict()}, weights)
    vocos = LocalVocos()
    monkeypatch.setattr(evaluation, "_load_vocos", lambda *args: vocos)
    monkeypatch.setattr(evaluation, "from_vocos_weights", lambda *args: lambda mel: mel.float())
    monkeypatch.setattr(evaluation, "_ours_mel_filterbank", lambda: None)
    monkeypatch.setattr(evaluation, "_project_log_mel", lambda *args: torch.ones(80, 8))
    monkeypatch.setattr(evaluation, "pitch_track", lambda waveform: {"f0": np.ones(4)})
    monkeypatch.setattr(evaluation, "compare_tracks", lambda *args, **kwargs: {"f0_corr": 1.0})
    generated = tmp_path / "mel.pt"
    torch.save({"mel": torch.ones(80, 8)}, generated)
    records = [
        {
            **manifest[0],
            "experiment_id": experiment,
            "generated_path": str(generated),
            "latency_ms": 2.0,
            "audio_seconds": 0.1,
            "mel_l1": 0.0,
        }
        for experiment in sorted({item for pair in comparisons.COMPARISONS for item in pair})
    ]

    def generate(*args):
        directory = args[4]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evaluation_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )

    monkeypatch.setattr(evaluation, "evaluate_unified", generate)
    return root, output, manifest, vocos


def test_smoke_full_and_resume(p1_runtime, monkeypatch):
    """六配置完整评分后重复运行不解码已有生成结果，仍核验真实对照。"""
    _, output, _, vocos = p1_runtime
    with pytest.raises(AssertionError, match="冒烟"):
        evaluation.run(output, "full", "cpu")
    evaluation.run(output, "smoke", "cpu")
    passed = json.loads((output / "smoke_corrected/PASSED.json").read_text(encoding="utf-8"))
    assert passed["generated"] == 6 and passed["controls"] == 4
    evaluation.run(output, "full", "cpu")
    metrics = read_csv(output / "full/per_sample_metrics.csv")
    assert len(metrics) == 6
    assert all(
        float(row["cer"]) == 0 and float(row["target_speaker_similarity"]) == 1 for row in metrics
    )
    assert all(float(row["cached_to_wave_latency_ms"]) >= 2 for row in metrics)
    assert all(Path(row["wav_path"]).is_file() for row in metrics)
    assert (output / "full/code_snapshot/src/example.py").is_file()
    decode = Mock(wraps=vocos.decode)
    monkeypatch.setattr(vocos, "decode", decode)
    evaluation.run(output, "full", "cpu")
    assert decode.call_count == 5  # 三次预热加两条真实语音重建。
    ledger = (output / "full/waveform_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 6


def test_paired_comparisons_filter_missing_values(tmp_path):
    """方法差值先逐源配对；全缺失指标不产生区间，分层保留 NaN。"""
    rows = []
    experiments = sorted({item for pair in comparisons.COMPARISONS for item in pair})
    for index, experiment in enumerate(experiments):
        for speaker in ("p1", "p2"):
            for sentence in ("001", "002"):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "source_id": f"{speaker}_{sentence}",
                        "source_speaker_id": speaker,
                        "sentence_id": sentence,
                        **dict.fromkeys(comparisons.METRICS, float(index)),
                        "wer": float("nan"),
                    }
                )
    path = tmp_path / "metrics.csv"
    evaluation._csv(path, rows)
    result = comparisons.run(path)
    assert len(result) == len(comparisons.COMPARISONS) * (len(comparisons.METRICS) - 1)
    for row in result:
        expected = experiments.index(row["experiment_a"]) - experiments.index(row["experiment_b"])
        assert row["mean_a_minus_b"] == row["ci95_low"] == row["ci95_high"] == expected
        assert row["paired_n"] == 4
    strata = read_csv(tmp_path / "speaker_strata.csv")
    assert len(strata) == 12
    assert all(np.isnan(float(row["wer"])) for row in strata)


def test_preflight_audits_source_path_cache_and_weights(p1_runtime, monkeypatch):
    """用真实前向钩子核验 B3 源扩散，实际读写缓存和模型状态文件。"""
    root, output, manifest, vocos = p1_runtime
    output = output.parent / "preflight"
    manifest *= 100
    for index, row in enumerate(manifest):
        manifest[index] = {**row, "index": index, "source_id": f"p1_{index:03d}"}
    by_id = {row["source_id"]: SimpleNamespace() for row in manifest}
    by_id["p2_001"] = SimpleNamespace()
    monkeypatch.setattr(evaluation, "_load_manifest_samples", lambda *args: (manifest, by_id))
    baseline = root / "analysis_results/baseline_validation"
    baseline.mkdir(parents=True)
    (baseline / "test_manifest.csv").write_text("fixture", encoding="utf-8")
    cache = root / "cache"
    monkeypatch.setattr(evaluation, "CACHE", cache)
    for kind, shape in (("content", (256, 2)), ("speaker", (256,))):
        (cache / kind).mkdir(parents=True)
        for uid in by_id:
            torch.save(torch.ones(shape), cache / kind / f"{uid}.pt")
    for label, run in evaluation.RUNS.items():
        run_dir = root / "logs/train/runs" / run
        (run_dir / ".hydra").mkdir(parents=True)
        (run_dir / ".hydra/config.yaml").write_text("data: {}", encoding="utf-8")
        (run_dir / evaluation.BEST[label]).write_bytes(b"checkpoint fixture")
    monkeypatch.setattr(
        evaluation, "_best_checkpoint", lambda directory: next(directory.glob("*.ckpt"))
    )
    monkeypatch.setattr(evaluation, "_load_model", lambda *args: (AnalyticModel(), {"data": {}}))
    monkeypatch.setattr(evaluation, "_validate_feature_cache", lambda *args: None)
    monkeypatch.setattr(
        evaluation, "_prepare_pair", lambda *args: (diagnostic_batch(), None, 2205)
    )
    monkeypatch.setattr(
        evaluation,
        "_collator",
        lambda *args: SimpleNamespace(_log_mel=lambda wave: torch.ones(80, 8)),
    )

    def generate(model, batch, nfe):
        source = batch["mel_spectrograms"]
        times = torch.full((1,), 0.95)
        states = model.method.interpolate(source, torch.randn_like(source), times)
        return model.method.network(states, {"source_time": times, "end_time": times})

    monkeypatch.setattr(evaluation, "_generate", generate)
    for name in ("vocos/config.yaml", "meanvc/ecapa_tdnn.py", "meanvc/wavlm_large.pt"):
        path = root / "pretrained" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    (root / "pretrained/whisper-small").mkdir()
    (root / "pretrained/whisper-small/config.json").write_text("{}", encoding="utf-8")
    torch.save(
        {**vocos.state_dict(), "feature_extractor.extra": torch.ones(1)},
        root / "pretrained/vocos/pytorch_model.bin",
    )
    evaluation.run(output, "preflight", "cpu")
    audit = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert audit["models"]["B3"]["diffused_real_source_verified"] is True
    assert all(row["same_seed_repeat_identical"] for row in audit["models"].values())
    assert len(audit["cache_files"]) == 202
    assert audit["vocos"]["missing_keys"] == []
    assert audit["vocos"]["ignored_keys"] == ["feature_extractor.extra"]
    assert len(read_csv(output / "experiment_plan.csv")) == 6
