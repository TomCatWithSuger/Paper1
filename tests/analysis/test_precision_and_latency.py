"""模块说明：用解析线性场验证有限差分校验与交错延迟记录的协议。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.analysis.p1 import latency
from src.analysis.p2 import precision
from src.analysis.p4.identity import read_csv
from tests.helpers.diagnostic_models import AnalyticModel, diagnostic_batch

pytestmark = pytest.mark.usefixtures("legacy_text_encoding")


def test_precision_matches_analytic_jvp_and_endpoint_difference(tmp_path, monkeypatch):
    """两种差分方案均应匹配线性场的解析 JVP，端点差分保留在时间域内。"""
    rows = [{"source_id": str(index)} for index in range(16)]
    monkeypatch.setattr(precision, "load_manifest", lambda *args: rows)
    monkeypatch.setattr(precision, "load_samples", lambda: ({}, None, {}))
    monkeypatch.setattr(precision, "prepared_items", lambda *args: {})
    monkeypatch.setattr(precision, "_collator", lambda *args: None)
    monkeypatch.setattr(
        precision, "batch_for", lambda selected, *args: diagnostic_batch(len(selected), 130)
    )
    model = AnalyticModel()
    monkeypatch.setattr(precision, "load_checkpoint", lambda *args: (model, {}, {}))
    directory = tmp_path / "d1"
    directory.mkdir()
    # 被测流程会设置全局 TF32 开关，测试后恢复调用方状态。
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        precision.run(tmp_path, torch.device("cpu"))
        precision.run_endpoint(tmp_path, torch.device("cpu"))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before[0]
        torch.backends.cudnn.allow_tf32 = before[1]
    central = read_csv(directory / "precision_fd_comparison.csv")
    assert len(central) == 2 * 4 * 4
    assert {row["tf32"] for row in central} == {"True", "False"}
    assert all(float(row["relative_l2"]) < 0.001 for row in central)
    drift = read_csv(directory / "precision_jvp_drift.csv")
    assert all(float(row["relative_l2"]) == 0 for row in drift)
    backward = read_csv(directory / "endpoint_finite_difference.csv")
    assert len(backward) == 64
    assert all(row["scheme"] == "second_order_backward_inside_domain" for row in backward)
    audit = json.loads((directory / "endpoint_fd_check.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True and audit["n_sources"] == 16
    assert audit["max_relative_l2"] < 0.001


def test_interleaved_latency_has_balanced_calls_and_seed_replay(tmp_path, monkeypatch):
    """计时器与设备边界可控，验证每个配对的六配置三次重复及分位数。"""
    (tmp_path / "full").mkdir()
    (tmp_path / "full/PASSED.json").write_text("{}", encoding="utf-8")
    manifest = [
        {"index": index, "source_id": f"p1_{index}", "reference_id": "p2_0"} for index in range(2)
    ]
    monkeypatch.setattr(latency, "_load_manifest_samples", lambda *args: (manifest, {}))
    monkeypatch.setattr(latency, "_load_model", lambda *args: (AnalyticModel(), {}))
    monkeypatch.setattr(latency, "_collator", lambda *args: None)
    monkeypatch.setattr(latency, "_prepare_pair", lambda *args: (diagnostic_batch(), None, 22050))
    monkeypatch.setattr(latency, "_to_device", lambda batch, device: batch)
    monkeypatch.setattr(
        latency, "_gpu_state", lambda path: path.write_text("CPU fixture", encoding="utf-8")
    )
    sync = Mock()
    seed = Mock()
    generate = Mock(return_value=torch.ones(1, 80, 8))
    monkeypatch.setattr(latency.torch.cuda, "synchronize", sync)
    monkeypatch.setattr(latency, "_seed", seed)
    monkeypatch.setattr(latency, "_generate", generate)
    times = iter(index * 0.002 for index in range(72))
    monkeypatch.setattr(latency, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    target = latency.run(tmp_path)
    records = read_csv(target / "per_call.csv")
    aggregates = read_csv(target / "aggregate.csv")
    assert len(records) == 36 and len(aggregates) == 6
    assert generate.call_count == 54 and sync.call_count == 73
    assert all(float(row["latency_ms"]) == pytest.approx(2) for row in records)
    assert all(float(row["rtf"]) == pytest.approx(0.002) for row in records)
    assert all(int(row["calls"]) == 6 for row in aggregates)
    for source in manifest:
        selected = [row for row in records if row["source_id"] == source["source_id"]]
        for repeat in range(3):
            group = [row for row in selected if int(row["repeat"]) == repeat]
            assert {int(row["order"]) for row in group} == set(range(6))
    assert [call.args[0] for call in seed.call_args_list] == [latency.SEED] * 18 + [
        latency.SEED + 1
    ] * 18
    protocol = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["repeats"] == 3 and protocol["shared_gpu"] is True
    with pytest.raises(FileExistsError):
        latency.run(tmp_path)
