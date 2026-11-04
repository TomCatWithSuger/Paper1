"""模块说明：用真实序列化状态验证优化器恢复与双臂 5000 步训练记录审计。"""

import json
import shutil
from functools import partial
from pathlib import Path

import pytest
import torch

from src.analysis.endpoint_pilot import reporting, runtime
from src.analysis.p4.identity import read_csv
from src.optim.warmup_cosine import WarmupCosineLR

pytestmark = pytest.mark.usefixtures("legacy_text_encoding")


def test_restore_verifies_adam_scheduler_and_parameter_order(tmp_path, monkeypatch):
    """先做一次真实 Adam 更新，再从文件恢复并逐项比较所有状态。"""

    def make_model():
        model = torch.nn.Linear(2, 1)
        setattr(model, "optimizer_factory", partial(torch.optim.Adam, lr=0.01))
        return model

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=0)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "initial.ckpt"
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "epoch": 3,
        "global_step": 1,
        "voice_parameter_names": [name for name, _ in model.named_parameters()],
    }
    torch.save(payload, path)
    monkeypatch.setattr(runtime, "_model_config", lambda *args: {"model": {}})
    monkeypatch.setattr(runtime.hydra.utils, "instantiate", lambda *args: make_model())
    restored, adam, schedule, _, audit = runtime.restore(path, torch.device("cpu"), path)
    assert runtime.tree_equal(restored.state_dict(), model.state_dict())
    assert runtime.tree_equal(adam.state_dict(), optimizer.state_dict())
    assert runtime.tree_equal(schedule.state_dict(), scheduler.state_dict())
    assert audit["adam_parameter_states"] == 2
    assert audit["optimizer_state_exact"] and audit["scheduler_state_exact"]
    assert audit["scheduler_last_epoch"] == audit["global_step"] == 1
    assert restored.training
    assert all(parameter.requires_grad for parameter in restored.parameters())
    payload["voice_parameter_names"].reverse()
    torch.save(payload, path)
    with pytest.raises(AssertionError, match="参数名或顺序"):
        runtime.restore(path, torch.device("cpu"), path)


@pytest.mark.parametrize(
    "left,right",
    [
        (torch.ones(2), torch.ones(3)),
        (torch.ones(2), torch.ones(2, dtype=torch.float64)),
        ({"a": 1}, {"b": 1}),
        ([1, 2], (1, 2)),
        ([1, 2], [1]),
        (1, 2),
    ],
)
def test_state_equality_rejects_structural_mismatches(left, right):
    """恢复审计必须同时核验形状、精度、键名、容器类型及值。"""
    assert not runtime.tree_equal(left, right)


def _json_lines(path: Path, rows: list[dict]) -> None:
    """写入与正式训练相同的逐行 JSON 账本。"""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture()
def training_records(tmp_path):
    """构造正式长度的共同流以及包含真实指纹的六个检查点。"""
    initial = {
        "file_sha256": "base",
        "epoch": 382,
        "global_step": 422066,
        "model_state_sha256": "original",
        "optimizer_lr": 0.01,
        "scheduler_total_steps": 551000,
        "scheduler_last_epoch": 422066,
        "optimizer_state_exact": True,
        "scheduler_state_exact": True,
        "parameter_names_exact": True,
    }
    (tmp_path / "initial_restore_audit.json").write_text(json.dumps(initial), encoding="utf-8")
    (tmp_path / "identity.json").write_text(
        json.dumps({"base_checkpoint_sha256": "base"}), encoding="utf-8"
    )
    gate = torch.arange(5000) % 2 == 0
    torch.save(
        {"epochs": torch.full((5000,), 387), "endpoint_mask": gate[:, None]},
        tmp_path / "common_plan.pt",
    )
    for arm_index, arm in enumerate(("control", "intervention")):
        directory = tmp_path / arm
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "restore_audit.json").write_text(json.dumps(initial), encoding="utf-8")
        rows = [
            {
                "pilot_update": index + 1,
                "global_step": 422067 + index,
                "finite": True,
                "data_sha256": f"data-{index}",
                "noise_sha256": f"noise-{index}",
                "source_ids": ["p1_001"],
                "crop_starts": [0],
                "lr_before": 0.01,
                "lr_after": 0.01,
                "epoch": 387,
                "batch_in_epoch": index,
                "batch_size": 1,
                "raw_meanflow_mse": 1.0 + arm_index,
                "zero_mse": 1.0,
                "jvp_rms": 0.5,
                "sampled_full_endpoint_mse": 2.0 if arm_index and gate[index] else None,
                "grad_norm_before_clip": 2.0,
                "clipped": True,
            }
            for index in range(5000)
        ]
        _json_lines(directory / "training.jsonl", rows)
        ledger = []
        for update in (1000, 2500, 5000):
            checkpoint = directory / "checkpoints" / f"{update}.ckpt"
            step = 422066 + update
            state = {
                "global_step": step,
                "epoch": 387,
                "lr_schedulers": [{"last_epoch": step}],
                "optimizer_states": [
                    {
                        "state": {
                            0: {
                                "step": step,
                                "exp_avg": torch.ones(1),
                                "exp_avg_sq": torch.ones(1),
                            }
                        }
                    }
                ],
                "state_dict": {"weight": torch.tensor([float(update + arm_index)])},
            }
            torch.save(state, checkpoint)
            ledger.append(
                {
                    "arm": arm,
                    "pilot_update": update,
                    "path": str(checkpoint),
                    "sha256": runtime._sha256_file(checkpoint),
                    "model_sha256": runtime.tensor_hash(state["state_dict"]["weight"]),
                }
            )
        shutil.copyfile(checkpoint, directory / "checkpoints/last.ckpt")
        _json_lines(directory / "checkpoint_ledger.jsonl", ledger)
        _json_lines(
            directory / "validation_probes.jsonl",
            [
                {
                    "pilot_update": 5000,
                    "interval": interval,
                    "source_id": "p1_001",
                    "raw_mse": 1.0,
                    "jvp_rms": 0.5,
                }
                for interval in ("zero", "full")
            ],
        )
    return tmp_path


def test_training_audit_checks_common_stream_and_final_checkpoint(training_records):
    """从原始账本计算统计，并验证末期权重和公共随机流的审计结论。"""
    reporting.training_audit(training_records)
    audit = json.loads(
        (training_records / "paired_training_audit.json").read_text(encoding="utf-8")
    )
    assert audit["updates_each"] == 5000 and audit["last_is_actual_final"] is True
    assert audit["all_data_noise_crop_lr_match"] is True
    statistics = read_csv(training_records / "training_statistics.csv")
    control = next(
        row
        for row in statistics
        if row["arm"] == "control" and row["metric"] == "sampled_full_endpoint_mse"
    )
    assert int(control["n"]) == 0 and control["mean"] == ""
    intervention = next(
        row
        for row in statistics
        if row["arm"] == "intervention" and row["metric"] == "sampled_full_endpoint_mse"
    )
    assert int(intervention["n"]) == 2500 and float(intervention["mean"]) == 2
    windows = read_csv(training_records / "training_windows.csv")
    assert len(windows) == 10
    assert all(float(row["clip_frequency"]) == 1 for row in windows)
    checkpoints = read_csv(training_records / "checkpoint_audit.csv")
    assert len(checkpoints) == 6
    assert len(read_csv(training_records / "validation_probe_curves.csv")) == 4


@pytest.mark.parametrize("changed", ["noise", "restore", "checkpoint"])
def test_training_audit_rejects_inconsistent_evidence(training_records, changed):
    """随机流、恢复状态或检查点任一被修改，都不能给出审计通过结论。"""
    directory = training_records / "intervention"
    if changed == "noise":
        path = directory / "training.jsonl"
        rows = reporting._json_rows(path)
        rows[0]["noise_sha256"] = "different"
        _json_lines(path, rows)
    elif changed == "restore":
        path = directory / "restore_audit.json"
        restored = json.loads(path.read_text(encoding="utf-8"))
        restored["optimizer_state_exact"] = False
        path.write_text(json.dumps(restored), encoding="utf-8")
    else:
        (directory / "checkpoints/1000.ckpt").write_bytes(b"tampered")
    with pytest.raises(AssertionError):
        reporting.training_audit(training_records)
    assert not (training_records / "paired_training_audit.json").exists()
