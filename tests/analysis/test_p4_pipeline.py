"""模块说明：验证 P4 从本地音频、配对表到盲听与独立评分审计的完整流程。"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.analysis.p4 import identity, scoring, sensitivity


@pytest.fixture()
def p4_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """满足正式的 64 源语句协议，所有文件均在临时目录生成。"""
    dataset = tmp_path / "dataset"
    waveform = np.sin(np.arange(3200) * 0.1).astype(np.float32) * 0.1
    for speaker, names in (("p1", [f"{i:03d}" for i in range(64)] + ["999"]), ("p2", ["001"])):
        directory = dataset / "wav48_silence_trimmed" / speaker
        directory.mkdir(parents=True)
        for name in names:
            sf.write(directory / f"{speaker}_{name}_mic1.flac", waveform, 16000)
    info = tmp_path / "speaker-info.txt"
    info.write_text("ID AGE GENDER\np1 20 F\np2 30 M\n")
    rows = []
    for config, source_time in zip(identity.CONFIGS, (0.8, 0.9, 0.95)):
        for repeat in (0, 1):
            wav = tmp_path / f"{config}_{repeat}.wav"
            sf.write(wav, waveform, 16000)
            for index in range(64):
                rows.append(
                    {
                        "experiment_id": config,
                        "source_id": f"p1_{index:03d}",
                        "reference_id": "p2_001",
                        "source_speaker_id": "p1",
                        "reference_speaker_id": "p2",
                        "repeat": repeat,
                        "seed": 123 + repeat,
                        "source_time": source_time,
                        "source_text": f"Sentence {index % 2}.",
                        "normalized_reference": f"sentence {index % 2}",
                        "donor_id": "p1_999",
                        "length_bin": "short",
                        "partition": "diagnostic_A",
                        "wav_path": str(wav),
                        "cer": 0.1,
                        "target_speaker_similarity": 0.4 + 0.2 * repeat,
                        "source_speaker_similarity": 0.5,
                    }
                )
    table = tmp_path / "scores.csv"
    identity.write_csv(table, rows)
    return table, info, dataset, tmp_path / "output"


class LocalEmbeddings:
    """只替换模型推理边界，保留正式的指纹验证和统计流程。"""

    runtime_version = "test-runtime"
    torch_version = "test-torch"

    def __init__(self, model: Path) -> None:
        self.model = model

    def embedding(self, path: str | Path) -> np.ndarray:
        """源与同说话人供体同向，目标与生成音频同向。"""
        return np.array([1.0, 0.0]) if "p1_" in Path(path).name else np.array([0.0, 1.0])


def test_analysis_scoring_and_sensitivity(p4_inputs, monkeypatch) -> None:
    """公开入口生成真实产物，并验证盲听匿名化与独立评分统计。"""
    table, info, dataset, output = p4_inputs
    protocol = identity.run(table, info, dataset, output)
    assert protocol["n_outputs"] == 384 and protocol["n_sources"] == 64
    assert protocol["input_sha256"] == identity.sha256(table)
    assert protocol["independent_scoring"] == "pending"
    participants = identity.read_csv(output / "listening/participant.csv")
    organizer = identity.read_csv(output / "listening/organizer_key.csv")
    assert len(participants) == len(organizer) == 12
    assert {row["candidate"] for row in participants} == {"A", "B", "C"}
    assert all("experiment_id" not in row and "source_id" not in row for row in participants)
    for row in participants:
        assert (output / "listening" / row["audio"]).is_file()
        assert "b3" not in row["audio"]
    summaries = identity.read_csv(output / "existing_summary.csv")
    preference = next(row for row in summaries if row["metric"] == "target_preferred")
    assert float(preference["mean"]) == 0.5
    assert int(preference["n_sources"]) == 64

    model = output.parent / "campplus.onnx"
    model.write_bytes(b"local model fixture")
    monkeypatch.setattr(scoring, "CampPlus", LocalEmbeddings)
    audit = scoring.run(output, model)
    assert audit["n_outputs"] == 384
    assert audit["n_control_pairs"] == 64
    assert audit["same_speaker_preferred_fraction"] == 1
    assert audit["model_sha256"] == identity.sha256(model)
    scores = identity.read_csv(output / "campplus/per_sample.csv")
    assert all(float(row["target_speaker_similarity"]) == 1 for row in scores)
    assert all(float(row["source_speaker_similarity"]) == 0 for row in scores)
    archive = np.load(output / "campplus/embeddings.npz")
    assert archive["embeddings"].shape == (audit["n_embeddings"], 2)
    saved = json.loads((output / "protocol.json").read_text())
    assert saved["independent_scoring"] == "completed_campplus"
    with pytest.raises(FileExistsError):
        scoring.run(output, model)

    sensitivity.run(output)
    intervals = identity.read_csv(output / "source_target_text_sensitivity.csv")
    assert len(intervals) == 30
    assert {row["scorer"] for row in intervals} == {"wavlm_ecapa", "campplus"}
    assert all(float(row["difference"]) == pytest.approx(0) for row in intervals)
    assert all(float(row["ci95_low"]) == pytest.approx(0) for row in intervals)
    saved = json.loads((output / "sensitivity_protocol.json").read_text())
    assert saved["clusters"] == ["source_speaker_id", "reference_speaker_id", "source_text"]
    assert (output / "source/sensitivity.py").is_file()
    with pytest.raises(FileExistsError):
        sensitivity.run(output)


@pytest.mark.parametrize("changed", ["table", "info", "audio"])
def test_independent_scoring_rejects_changed_inputs(p4_inputs, monkeypatch, changed) -> None:
    """任何上游指纹变化都必须在创建评分会话前被拒绝。"""
    table, info, dataset, output = p4_inputs
    protocol = identity.run(table, info, dataset, output)
    paths = {"table": table, "info": info, "audio": Path(next(iter(protocol["audio_sha256"])))}
    with paths[changed].open("ab") as stream:
        stream.write(b"changed")
    calls = []
    monkeypatch.setattr(scoring, "CampPlus", lambda model: calls.append(model))
    with pytest.raises(ValueError, match="变化"):
        scoring.run(output, output / "missing.onnx")
    assert not calls
    assert not (output / "campplus").exists()


def test_sensitivity_requires_completed_scoring(p4_inputs) -> None:
    """独立评分尚未完成时不得生成混合来源的最终区间。"""
    table, info, dataset, output = p4_inputs
    identity.run(table, info, dataset, output)
    with pytest.raises(RuntimeError, match="必须先完成独立评分"):
        sensitivity.run(output)
    assert not (output / "source_target_text_sensitivity.csv").exists()
