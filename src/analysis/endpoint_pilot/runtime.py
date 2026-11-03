"""模块说明：

- 负责端点实验的训练状态恢复、固定探针、预检与检查点核验。
- 提供原训练入口复用的运行时操作；不启动训练，不下载资源。
"""

# ====================
# 1. Imports
# ====================

import copy
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.analysis.endpoint_pilot.data import (
    BATCH_SIZE,
    MICRO_BATCH,
    PILOT_STEPS,
    PilotCollator,
    PlannedBatches,
    PlannedDataset,
    batch_hash,
    make_plan,
    plan_counts,
    planned_times,
    tensor_hash,
)
from src.analysis.p1.evaluation import CACHE, ROOT, RUNS, _csv, _gpu_state, _json
from src.analysis.p2.common import (
    batch_for,
    load_manifest,
    load_samples,
    prepared_items,
)
from src.analysis.unified.checkpoints import CHECKPOINT_SAFE_GLOBALS, _model_config
from src.analysis.unified.data import _collator
from src.analysis.unified.identity import _sha256_file
from src.models.algorithms.autodiff import tensor_jvp
from src.optim.warmup_cosine import WarmupCosineLR

# ====================
# 6. Public API
# ====================


def safe_load(path):
    """仅在显式允许的配置类型范围内读取本地训练检查点。"""
    with torch.serialization.safe_globals(list(CHECKPOINT_SAFE_GLOBALS)):
        return torch.load(path, map_location="cpu", weights_only=True)


def tree_equal(left, right):
    """核验恢复后的值，不把仅load_state_dict未报错视为完整恢复。"""
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(tree_equal(v, right[k]) for k, v in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def model_hash(model):
    """对模型各状态张量的名称和内容计算统一指纹。"""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def _append(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def restore(path, device, config_checkpoint):
    """完整恢复可验证的优化状态；不复用未校准的Lightning循环计数。"""
    checkpoint = safe_load(path)
    config = _model_config(config_checkpoint.parent.parent)
    model = hydra.utils.instantiate(OmegaConf.create(config["model"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.train().requires_grad_(True).to(device)
    names = [name for name, _ in model.named_parameters()]
    assert names == checkpoint["voice_parameter_names"], "参数名或顺序不匹配，禁止声称完整Adam恢复"
    optimizer = model.optimizer_factory(params=model.parameters())
    state = checkpoint["lr_schedulers"][0]
    scheduler = WarmupCosineLR(optimizer, int(state["total_steps"]), int(state["warmup_steps"]))
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    scheduler.load_state_dict(state)
    assert tree_equal(
        optimizer.state_dict(), checkpoint["optimizer_states"][0]
    ), "Adam状态逐项核验失败"
    assert tree_equal(scheduler.state_dict(), state), "调度器状态逐项核验失败"
    assert int(scheduler.last_epoch) == int(checkpoint["global_step"])
    for parameter, value in optimizer.state.items():
        assert (
            value["exp_avg"].shape == parameter.shape
            and value["exp_avg_sq"].shape == parameter.shape
        )
        assert int(value["step"]) == int(checkpoint["global_step"])
    audit = {
        "checkpoint": str(path),
        "file_sha256": _sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "parameter_names_exact": True,
        "optimizer_state_exact": True,
        "scheduler_state_exact": True,
        "adam_parameter_states": len(optimizer.state),
        "optimizer_lr": optimizer.param_groups[0]["lr"],
        "scheduler_total_steps": scheduler.total_steps,
        "scheduler_last_epoch": scheduler.last_epoch,
        "model_state_sha256": model_hash(model),
        "original_rng_state_available": False,
        "lightning_fit_loop_restored": False,
        "loop_backend": "显式优化器更新循环；全局步数与数据epoch独立记录",
    }
    del checkpoint
    return model, optimizer, scheduler, config, audit


def native_planned_loss(method, batch, r, t, generator):
    """仅替换时间采样返回值，原生compute_loss、JVP及梯度目标完全保留。"""

    def sample(size, device, dtype, generator=None):
        assert size == len(r)
        return r.to(device=device, dtype=dtype), t.to(device=device, dtype=dtype)

    with patch.object(method, "_sample_time_intervals", side_effect=sample):
        return method.compute_loss(
            batch["mel_spectrograms"],
            batch["mel_attention_mask"],
            batch["content_features"],
            batch["content_feature_lengths"],
            batch["speaker_features"],
            generator=generator,
            return_diagnostics=True,
        )


def microbatch(batch, start, stop):
    """按样本范围切分批内张量和说话人元信息，供梯度累积使用。"""
    return {
        key: value[start:stop] if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _reserved_manifest(p2_manifest, split):
    """在查看pilot结果前冻结剩余验证语句，排除P2全部源/参考/替换ID。"""
    import soundfile as sf

    used = {r[key] for r in p2_manifest for key in ["source_id", "reference_id", "donor_id"]}
    speakers = sorted({r["source_speaker_id"] for r in p2_manifest})
    grouped, seconds = {}, {}
    for speaker in speakers:
        candidates = []
        for sample in split.val:
            if sample.speaker_id != speaker or sample.utterance_id in used:
                continue
            duration = sf.info(str(sample.audio_path)).duration
            if 2 <= duration <= 10:
                candidates.append(sample)
                seconds[sample.utterance_id] = duration
        candidates.sort(key=lambda s: (seconds[s.utterance_id], s.utterance_id))
        assert len(candidates) >= 4
        chosen = [candidates[int(q * (len(candidates) - 1))] for q in (0.2, 0.5, 0.8, 1.0)]
        assert len({s.utterance_id for s in chosen}) == 4
        grouped[speaker] = chosen
    rows = []
    for rank in range(4):
        for i, speaker in enumerate(speakers):
            source = grouped[speaker][rank]
            reference = grouped[speakers[(i + 1) % len(speakers)]][rank]
            donor = grouped[speaker][rank ^ 1]
            from src.analysis.p2.common import length_bin

            rows.append(
                {
                    "index": len(rows),
                    "source_id": source.utterance_id,
                    "source_speaker_id": speaker,
                    "sentence_id": source.sentence_id,
                    "reference_id": reference.utterance_id,
                    "reference_speaker_id": reference.speaker_id,
                    "donor_id": donor.utterance_id,
                    "source_text": source.text,
                    "donor_text": donor.text,
                    "reference_same_text": source.text == reference.text,
                    "audio_seconds": seconds[source.utterance_id],
                    "length_bin": length_bin(seconds[source.utterance_id]),
                    "partition": "reserved_validation",
                }
            )
    assert not used & {r[k] for r in rows for k in ["source_id", "reference_id", "donor_id"]}
    return rows


def fixed_probe(model, rows, items, collator, device):
    """用固定样本与噪声探测零区间和完整端点，随后恢复训练模式。"""
    previous = model.training
    model.eval()
    records = []
    with torch.no_grad():
        for row in rows:
            batch = batch_for([row], items, collator, device)
            x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
            condition = model.method.encode_condition(
                batch["content_features"],
                batch["content_feature_lengths"],
                batch["speaker_features"],
                mask,
            )
            for repeat in range(2):
                noise = torch.randn(
                    x.shape,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        20261024 + int(row["index"]) + 100000 * repeat
                    ),
                )
                velocity = noise - x
                for name, r, t in [("zero", 0.5, 0.5), ("full", 0.0, 1.0)]:
                    rs, ts = x.new_full((1,), r), x.new_full((1,), t)
                    z = model.method.interpolate(x, noise, ts)

                    def f(states, ends, starts):
                        return model.method(states, starts, ends, condition, mask)

                    prediction, derivative = tensor_jvp(
                        f,
                        (z, ts, rs),
                        (velocity, torch.ones_like(ts), torch.zeros_like(rs)),
                    )
                    correction = (t - r) * derivative
                    error = (prediction - (velocity - correction)).square()
                    records.append(
                        {
                            "source_id": row["source_id"],
                            "repeat": repeat,
                            "interval": name,
                            "raw_mse": float(error.mean()),
                            "jvp_rms": float(correction.square().mean().sqrt()),
                            "reconstruction_l1": (
                                float((noise - prediction - x).abs().mean())
                                if name == "full"
                                else None
                            ),
                        }
                    )
    model.train(previous)
    return records


def save_checkpoint(output, arm, model, optimizer, scheduler, plan, update, initial):
    """保存完整更新状态并维护末期副本，回读核验训练步数。"""
    epoch = int(plan["epochs"][update - 1])
    global_step = int(plan["base_global_step"]) + update
    destination = output / arm / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"epoch_{epoch:04d}_global_{global_step:07d}_pilot_{update:05d}.ckpt"
    value = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "voice_parameter_names": [name for name, _ in model.named_parameters()],
        "pilot_updates": update,
        "pilot_arm": arm,
        "pilot_batch_in_epoch": int(plan["batch_indices"][update - 1]),
        "base_checkpoint_sha256": initial["file_sha256"],
        "common_plan_sha256": _sha256_file(output / "common_plan.pt"),
        "loop_backend": "controlled_manual_optimizer_updates",
        "hyper_parameters": dict(model.hparams),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    last = destination / "last.ckpt"
    shutil.copyfile(path, destination / "last.tmp")
    os.replace(destination / "last.tmp", last)
    saved = safe_load(path)
    assert (
        saved["epoch"] == epoch
        and saved["global_step"] == global_step
        and saved["pilot_updates"] == update
    )
    assert all(
        int(s["step"]) == global_step for s in saved["optimizer_states"][0]["state"].values()
    )
    assert saved["lr_schedulers"][0]["last_epoch"] == global_step
    fingerprint = _sha256_file(path)
    assert fingerprint == _sha256_file(last) and fingerprint != initial["file_sha256"]
    record = {
        "arm": arm,
        "pilot_update": update,
        "epoch": epoch,
        "global_step": global_step,
        "path": str(path),
        "sha256": fingerprint,
        "model_sha256": model_hash(model),
        "last_matches_this_step": True,
    }
    _append(output / arm / "checkpoint_ledger.jsonl", record)
    return path


def preflight(output, device, base_checkpoint):
    """真实训练数据、最长padding形状反传；不执行任何优化器更新。"""
    import torch.nn.functional as F

    torch.set_num_threads(2)
    plan = safe_load(output / "common_plan.pt")
    model, optimizer, scheduler, config, audit = restore(base_checkpoint, device, base_checkpoint)
    _, split, _ = load_samples()
    dataset = PlannedDataset(split.train, config, plan)
    collator = PilotCollator(config)
    cpu = collator([dataset[i] for i in range(MICRO_BATCH)])
    before = model_hash(model)
    batch = {k: v.to(device) for k, v in cpu.items() if isinstance(v, torch.Tensor)}
    frames = 220500 // 256 + 1
    padding = frames - batch["mel_spectrograms"].shape[-1]
    if padding > 0:
        batch["mel_spectrograms"] = F.pad(batch["mel_spectrograms"], (0, padding))
        batch["mel_attention_mask"] = F.pad(batch["mel_attention_mask"], (0, padding), value=False)
    torch.cuda.reset_peak_memory_stats(device)
    r, t = planned_times(plan, 0, "intervention", 0, MICRO_BATCH)
    loss = native_planned_loss(
        model.method,
        batch,
        r,
        t,
        torch.Generator(device=device).manual_seed(int(plan["noise_seeds"][0, 0])),
    )[0]
    assert torch.isfinite(loss)
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    optimizer.zero_grad(set_to_none=True)
    assert model_hash(model) == before
    assert all(int(s["step"]) == audit["global_step"] for s in optimizer.state.values())
    result = {
        "real_samples": MICRO_BATCH,
        "padded_frames": batch["mel_spectrograms"].shape[-1],
        "loss": float(loss),
        "gradient_norm": norm,
        "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "model_unchanged": True,
        "optimizer_updates": 0,
        "optimizer_state_steps_unchanged": True,
        "passed": True,
    }
    _json(output / "preflight.json", result)
    print(result, flush=True)
