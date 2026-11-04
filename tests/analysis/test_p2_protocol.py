"""模块说明：验证 P2 固定验证协议、保留队列与共同训练数据流的可复现性。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from src.analysis.endpoint_pilot import data, runtime
from src.analysis.p2 import common
from src.data.components.vctk_splits import VCTKSample, VCTKSplit
from tests.helpers.diagnostic_models import AnalyticModel, diagnostic_batch


@pytest.fixture()
def protocol_resources(tmp_path, monkeypatch):
    """16 个说话人各有八条有效语句与一条超长语句，另设互斥训练和测试集。"""
    root = tmp_path / "project"
    root.mkdir()
    wav = root / "short.wav"
    long_wav = root / "long.wav"
    sf.write(wav, np.sin(np.arange(32000) * 0.1) * 0.1, 16000)
    sf.write(long_wav, np.zeros(192000), 16000)
    samples = []
    for speaker_index in range(16):
        speaker = f"p{speaker_index:02d}"
        for sentence in range(8):
            text = f"hello {sentence}"
            if sentence in (2, 6):
                text += f" from {speaker}"
            samples.append(
                VCTKSample(f"{speaker}_{sentence:03d}", f"{sentence:03d}", speaker, text, wav)
            )
        samples.append(VCTKSample(f"{speaker}_999", "999", speaker, "hello 0", long_wav))
    train = VCTKSample("train_001", "001", "train", "training", wav)
    test = VCTKSample("test_001", "001", "test", "testing", wav)
    split = VCTKSplit((train,), tuple(samples), (test,))
    by_id = {sample.utterance_id: sample for sample in samples}
    cache = root / "cache"
    for kind, shape in (("content", (256, 40)), ("speaker", (256,))):
        (cache / kind).mkdir(parents=True)
        for uid in by_id:
            torch.save(torch.ones(shape), cache / kind / f"{uid}.pt")
    for label, run in common.RUNS.items():
        directory = root / "logs/train/runs" / run
        (directory / ".hydra").mkdir(parents=True)
        (directory / ".hydra/config.yaml").write_text("data: {}")
        (directory / "checkpoints").mkdir()
        for name in ("epoch_382.ckpt", "epoch_040.ckpt", "last.ckpt"):
            torch.save(
                {"state_dict": AnalyticModel().state_dict(), "epoch": 40, "global_step": 100},
                directory / "checkpoints" / name,
            )
    prior = root / "analysis_results/p1_latest_20261024"
    prior.mkdir(parents=True)
    resource = root / "resource.bin"
    resource.write_bytes(b"resource fixture")
    (prior / "preflight.json").write_text(
        json.dumps({"resource_files": {str(resource): common._sha256_file(resource)}})
    )
    script = root / "scripts/analysis/evaluate_p2.py"
    script.parent.mkdir(parents=True)
    script.write_text("# 测试源码\n")
    (root / "src").mkdir()
    (root / "src/local.py").write_text("# 测试源码\n")
    config = {"data": {"sample_rate": 16000, "max_duration_seconds": 0.1}}
    monkeypatch.setattr(common, "ROOT", root)
    monkeypatch.setattr(common, "CACHE", cache)
    monkeypatch.setattr(data, "CACHE", cache)
    monkeypatch.setattr(common, "load_samples", lambda: (config, split, by_id))
    monkeypatch.setattr(common, "_gpu_state", lambda path: path.write_text("CPU fixture"))
    return root, config, split, by_id


def test_frozen_protocol_and_reserved_cohort_are_disjoint(protocol_resources, tmp_path):
    """完整准备流程记录指纹与代码快照；保留队列排除所有已有角色的语句。"""
    _, _, split, _ = protocol_resources
    output = tmp_path / "protocol"
    common.run(output)
    manifest = common.load_manifest(output)
    assert len(manifest) == 64
    assert len({row["source_id"] for row in manifest}) == 64
    assert len({row["source_speaker_id"] for row in manifest}) == 16
    assert {row["partition"] for row in manifest} == {"diagnostic_A", "confirmation_B"}
    assert any(row["reference_same_text"] for row in manifest)
    assert any(not row["reference_same_text"] for row in manifest)
    assert all(row["source_id"] != row["donor_id"] for row in manifest)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["no_train_or_test_utterance_used"] is True
    assert protocol["split_counts"] == {"train": 1, "val": 144, "test": 1}
    identity = json.loads((output / "identity.json").read_text())
    assert len(identity["checkpoint_files"]) == 4
    assert len(identity["caches"]) == 128
    assert (output / "code_snapshot/scripts/analysis/evaluate_p2.py").is_file()
    reserved = runtime._reserved_manifest(manifest, split)
    original_ids = {
        row[key] for row in manifest for key in ("source_id", "reference_id", "donor_id")
    }
    reserved_ids = {
        row[key] for row in reserved for key in ("source_id", "reference_id", "donor_id")
    }
    assert len(reserved) == 64 and not original_ids & reserved_ids
    assert all(row["source_speaker_id"] != row["reference_speaker_id"] for row in reserved)
    assert all(row["partition"] == "reserved_validation" for row in reserved)
    with pytest.raises(FileExistsError):
        common.run(output)


def test_checkpoint_loading_is_strict_and_frozen(protocol_resources, monkeypatch):
    """严格加载本地状态并冻结参数，文件身份包含实际检查点指纹。"""
    _, _, _, _ = protocol_resources
    monkeypatch.setattr(common, "_model_config", lambda *args: {"model": {}})
    monkeypatch.setattr(common.hydra.utils, "instantiate", lambda *args: AnalyticModel())
    model, _, audit = common.load_checkpoint("B2", "epoch_382.ckpt", torch.device("cpu"))
    assert not model.training
    assert audit["strict_load"] is True and audit["global_step"] == 100
    assert audit["sha256"] == common._sha256_file(Path(audit["path"]))
    torch.save(
        {"state_dict": {"unexpected": torch.ones(1)}, "epoch": 1, "global_step": 1}, audit["path"]
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        common.load_checkpoint("B2", "epoch_382.ckpt", torch.device("cpu"))


def test_planned_crops_restore_rng_and_partial_batches(protocol_resources):
    """实际读取缓存与音频，重放裁剪不消耗全局随机流，末批不读取占位索引。"""
    _, config, split, _ = protocol_resources
    plan = data.make_plan(5, base_epoch=2, updates=4, batch_size=3)
    assert plan["batch_sizes"].tolist() == [3, 2, 3, 2]
    batches = data.PlannedBatches(plan, start=1, stop=3)
    assert len(batches) == 2 and list(batches) == [[3, 4], [6, 7, 8]]
    dataset = data.PlannedDataset(split.val[:5], config, plan)
    state = torch.random.get_rng_state().clone()
    first, second = dataset[0], dataset[0]
    assert len(dataset) == 12
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(first["waveform"], second["waveform"])
    assert first["pilot_crop_start"] == second["pilot_crop_start"]
    assert first["pilot_position"] == 0
    assert first["waveform"].numel() == 1600
    with pytest.raises(AssertionError):
        dataset[5]
    counts = data.plan_counts(plan)
    assert counts["sample_draws"] == 10
    assert counts["zero_draws"] + counts["control_nonzero_draws"] == 10
    assert (
        counts["intervention_original_nonzero_draws"] + counts["intervention_endpoint_draws"]
        == counts["control_nonzero_draws"]
    )
    original = plan["r"].clone()
    starts, _ = data.planned_times(plan, 0, "control")
    starts.zero_()
    torch.testing.assert_close(plan["r"], original)
    with pytest.raises(ValueError):
        data.planned_times(plan, 0, "unknown")


def test_fixed_probe_restores_mode_and_has_deterministic_derivatives(monkeypatch):
    """相同源和种子返回相同诊断，探针不改变训练状态。"""
    model = AnalyticModel().train()
    monkeypatch.setattr(runtime, "batch_for", lambda *args: diagnostic_batch())
    rows = [{"index": 0, "source_id": "p1_0"}]
    first = runtime.fixed_probe(model, rows, {}, None, torch.device("cpu"))
    second = runtime.fixed_probe(model, rows, {}, None, torch.device("cpu"))
    assert model.training and first == second
    assert len(first) == 4
    assert all(
        row["jvp_rms"] == 0 and row["reconstruction_l1"] is None
        for row in first
        if row["interval"] == "zero"
    )
    assert all(row["jvp_rms"] > 0 for row in first if row["interval"] == "full")
