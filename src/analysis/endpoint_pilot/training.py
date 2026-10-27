"""模块说明：

- 从同一B2模型、Adam与原余弦调度器状态启动两个5000步训练臂。
- 唯一干预是原生时间采样输出；保留原MeanFlow损失与JVP目标。
- 保存逐步共同输入核验、梯度/数值监控以及可核验的真实末期权重。
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
from src.optim.warmup_cosine import WarmupCosineLR

# ====================
# 2. Definitions
# ====================

BASE_CHECKPOINT = ROOT / "logs/train/runs" / RUNS["B2"] / "checkpoints/epoch_382.ckpt"
P2_OUTPUT = ROOT / "analysis_results/p2_mechanisms_20261024"
CHECKPOINT_UPDATES = (1000, 2500, 5000)


# ====================
# 3. Utilities
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


def restore(path, device):
    """完整恢复可验证的优化状态；不复用未校准的Lightning循环计数。"""
    checkpoint = safe_load(path)
    config = _model_config(BASE_CHECKPOINT.parent.parent)
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


# ====================
# 4. Internal Functions
# ====================


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


def prepare(output):
    """核验初始状态并固定双臂共同随机计划及保留验证清单。"""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model, optimizer, scheduler, config, audit = restore(BASE_CHECKPOINT, torch.device("cpu"))
    _json(output / "initial_restore_audit.json", audit)
    data_config, split, _ = load_samples()
    assert len(split.train) == 35234
    plan = make_plan(len(split.train), audit["epoch"])
    plan["lr_before_update"] = torch.tensor(
        [
            scheduler.base_lrs[0] * scheduler._multiplier(audit["global_step"] + i)
            for i in range(PILOT_STEPS)
        ],
        dtype=torch.float64,
    )
    plan["base_global_step"] = audit["global_step"]
    plan["base_epoch"] = audit["epoch"]
    torch.save(plan, output / "common_plan.pt")
    _json(output / "common_plan_counts.json", plan_counts(plan))
    _csv(
        output / "training_samples.csv",
        [
            {
                "index": i,
                "utterance_id": s.utterance_id,
                "speaker_id": s.speaker_id,
                "audio_path": str(s.audio_path),
            }
            for i, s in enumerate(split.train)
        ],
    )
    p2_manifest = load_manifest(P2_OUTPUT)
    reserved = _reserved_manifest(p2_manifest, split)
    _json(output / "p2_validation_manifest.json", p2_manifest)
    _json(output / "reserved_validation_manifest.json", reserved)
    _csv(output / "reserved_validation_manifest.csv", reserved)
    val_ids, train_ids, test_ids = (
        {s.utterance_id for s in subset} for subset in [split.val, split.train, split.test]
    )
    for rows in [p2_manifest, reserved]:
        used = {r[k] for r in rows for k in ["source_id", "reference_id", "donor_id"]}
        assert used <= val_ids and not used & (train_ids | test_ids)
    for sample in split.train:
        for kind in ["content", "speaker"]:
            assert (CACHE / kind / f"{sample.utterance_id}.pt").is_file()
    resources = json.loads((P2_OUTPUT / "identity.json").read_text())["resources"]
    for filename, digest in resources.items():
        assert _sha256_file(Path(filename)) == digest
    identity = {
        "base_checkpoint_sha256": _sha256_file(BASE_CHECKPOINT),
        "common_plan_sha256": _sha256_file(output / "common_plan.pt"),
        "training_config_sha256": _sha256_file(
            BASE_CHECKPOINT.parent.parent / ".hydra/config.yaml"
        ),
        "resources": resources,
        "training_samples_sha256": _sha256_file(output / "training_samples.csv"),
        "p2_manifest_sha256": _sha256_file(output / "p2_validation_manifest.json"),
        "reserved_manifest_sha256": _sha256_file(output / "reserved_validation_manifest.json"),
    }
    _json(output / "identity.json", identity)
    shutil.copy2(
        BASE_CHECKPOINT.parent.parent / ".hydra/config.yaml", output / "base_training_config.yaml"
    )
    protocol = {
        "arms": ["control", "intervention"],
        "updates_per_arm": PILOT_STEPS,
        "hidden_channels": 512,
        "effective_batch": BATCH_SIZE,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": "完整32条先collate，微批按全有效元素比例累计，裁剪/Adam/scheduler每有效批仅一次；保留原最后不足32条批次",
        "gradient_clip_norm": 1.0,
        "control_probabilities": [0.75, 0.25, 0.0],
        "intervention_probabilities": [0.75, 0.23, 0.02],
        "endpoint_gate_given_nonzero": 0.08,
        "probabilities_not_forced_finite_counts": True,
        "loss": "原生ConditionalMeanFlow.compute_loss，adaptive_loss_power=0，无辅助loss",
        "scheduler_original_total_steps": 551000,
        "optimizer_restored_exact": True,
        "model_optimizer_scheduler_global_step_restored": True,
        "original_rng_not_saved": True,
        "randomness": "新共同计划；显式记录数据排列、裁剪种子、噪声种子、原r/t和端点门控；逐步比较实际输入/噪声hash",
        "precision": "完整FP32，TF32=False，两臂相同；不声称复原原运行未记录的硬件数值设置",
        "checkpoint_updates": CHECKPOINT_UPDATES,
        "primary_evaluation": "final5000，不按验证集选择中间权重",
        "validation": "P2固定64对+新增保留64对，每对2噪声；不用测试集",
        "capacity_decision": "两验证队列CER和目标音色须共同改善且源身份残留不增加；未通过则不继续hidden256",
        "training_started": False,
    }
    _json(output / "protocol.json", protocol)
    _gpu_state(output / "gpu_prepare.txt")
    del model, optimizer, scheduler
    print("准备完成：模型/Adam/调度器逐项恢复通过，共同5000步计划和两验证队列已冻结。", flush=True)


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

                    prediction, derivative = torch.func.jvp(
                        f, (z, ts, rs), (velocity, torch.ones_like(ts), torch.zeros_like(rs))
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


def train(output, arm, device, workers=8):
    """执行指定实验臂的固定步数训练并记录裁剪、探针和中间权重。"""
    torch.set_num_threads(2)
    assert json.loads((output / "preflight.json").read_text())["passed"]
    plan = safe_load(output / "common_plan.pt")
    assert plan["micro_batch"] == MICRO_BATCH
    assert (
        _sha256_file(output / "common_plan.pt")
        == json.loads((output / "identity.json").read_text())["common_plan_sha256"]
    )
    reference_rows = None
    if arm == "intervention":
        assert (output / "control/TRAINING_COMPLETE.json").is_file()
        reference_rows = [
            json.loads(line)
            for line in (output / "control/training.jsonl").read_text().splitlines()
        ]
        assert len(reference_rows) == PILOT_STEPS
    directory = output / arm
    directory.mkdir(exist_ok=False)
    source_hashes = {}
    for path in [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "scripts/analysis/run_endpoint_pilot.py",
    ]:
        relative = path.relative_to(ROOT)
        source_hashes[str(relative)] = _sha256_file(path)
        destination = directory / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    _json(
        directory / "runtime.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "tf32": False,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "source_sha256": source_hashes,
        },
    )
    model, optimizer, scheduler, config, initial = restore(BASE_CHECKPOINT, device)
    _json(directory / "restore_audit.json", initial)
    _, split, by_id = load_samples()
    data = PlannedDataset(split.train, config, plan)
    loader = DataLoader(
        data,
        batch_sampler=PlannedBatches(plan),
        collate_fn=PilotCollator(config),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
    )
    manifest = load_manifest(P2_OUTPUT)
    probe_rows = [manifest[i] for i in np.linspace(0, len(manifest) - 1, 8, dtype=int)]
    probe_items = prepared_items(probe_rows, config, by_id)
    collator = _collator(config)
    for record in fixed_probe(model, probe_rows, probe_items, collator, device):
        _append(
            directory / "validation_probes.jsonl",
            {"pilot_update": 0, "global_step": initial["global_step"], **record},
        )
    _gpu_state(directory / "gpu_start.txt")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    parameter_names = [name for name, _ in model.named_parameters()]
    clipping_count = 0
    first_data = None
    for step, cpu_batch in enumerate(loader):
        assert cpu_batch["pilot_positions"][0] == step * BATCH_SIZE
        size = int(plan["batch_sizes"][step])
        assert len(cpu_batch["ids"]) == size
        data_hash = batch_hash(cpu_batch)
        if reference_rows is not None:
            reference = reference_rows[step]
            assert (
                data_hash == reference["data_sha256"]
                and cpu_batch["ids"] == reference["source_ids"]
            )
            assert cpu_batch["pilot_crop_starts"] == reference["crop_starts"]
        if first_data is None:
            first_data = data_hash
        tensors = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
            if isinstance(value, torch.Tensor)
        }
        total_elements = (
            float(cpu_batch["mel_attention_mask"].sum()) * tensors["mel_spectrograms"].shape[1]
        )
        lr = optimizer.param_groups[0]["lr"]
        assert abs(lr - float(plan["lr_before_update"][step])) < 1e-15
        if reference_rows is not None:
            assert lr == reference_rows[step]["lr_before"]
        optimizer.zero_grad(set_to_none=True)
        squared_error_sum, zero_sum, full_sum, zero_elements, full_elements, jvp_squared = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        noise_hashes = []
        for chunk, start in enumerate(range(0, size, MICRO_BATCH)):
            stop = min(start + MICRO_BATCH, size)
            batch = microbatch(tensors, start, stop)
            count = (
                float(cpu_batch["mel_attention_mask"][start:stop].sum())
                * batch["mel_spectrograms"].shape[1]
            )
            r, t = planned_times(plan, step, arm, start, stop)
            seed = int(plan["noise_seeds"][step, chunk])
            generator = torch.Generator(device=device).manual_seed(seed)
            # 单独重放同一次原生标准正态抽样用于hash，不改变计算使用的generator。
            noise = torch.randn(
                batch["mel_spectrograms"].shape,
                device=device,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            noise_hashes.append(tensor_hash(noise))
            if reference_rows is not None:
                assert noise_hashes[-1] == reference_rows[step]["noise_sha256"][chunk]
            del noise
            loss, prediction, target, _, diagnostics = native_planned_loss(
                model.method, batch, r, t, generator
            )
            if (
                not torch.isfinite(loss)
                or not torch.isfinite(prediction).all()
                or not torch.isfinite(target).all()
            ):
                _json(
                    directory / "NUMERICAL_FAILURE.json",
                    {
                        "pilot_update_before_failure": step,
                        "chunk": chunk,
                        "global_step": initial["global_step"] + step,
                        "kind": "nonfinite_loss_or_forward",
                    },
                )
                raise FloatingPointError("出现非有限数，禁止跳过更新或悄然重置状态")
            (loss * (count / total_elements)).backward()
            error = (prediction.detach() - target).square()
            valid = batch["mel_attention_mask"][:, None].to(error.dtype)
            rz, tz = r.to(device), t.to(device)
            zm = valid * (rz == tz)[:, None, None]
            fm = valid * ((rz == 0) & (tz == 1))[:, None, None]
            squared_error_sum += float((error * valid).sum())
            zero_sum += float((error * zm).sum())
            full_sum += float((error * fm).sum())
            zero_elements += float(zm.sum()) * error.shape[1]
            full_elements += float(fm.sum()) * error.shape[1]
            jvp_squared += float(diagnostics["jvp_correction_rms"]) ** 2 * count
            del loss, prediction, target, diagnostics
        try:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
            )
        except RuntimeError:
            _json(
                directory / "NUMERICAL_FAILURE.json",
                {
                    "pilot_update_before_failure": step,
                    "global_step": initial["global_step"] + step,
                    "kind": "nonfinite_gradient",
                },
            )
            raise
        clipped = gradient_norm > 1.0
        clipping_count += int(clipped)
        optimizer.step()
        scheduler.step()
        update = step + 1
        record = {
            "pilot_update": update,
            "global_step": initial["global_step"] + update,
            "epoch": int(plan["epochs"][step]),
            "batch_in_epoch": int(plan["batch_indices"][step]),
            "batch_size": size,
            "data_sha256": data_hash,
            "noise_sha256": noise_hashes,
            "source_ids": cpu_batch["ids"],
            "crop_starts": cpu_batch["pilot_crop_starts"],
            "lr_before": lr,
            "lr_after": optimizer.param_groups[0]["lr"],
            "raw_meanflow_mse": squared_error_sum / total_elements,
            "zero_mse": zero_sum / zero_elements if zero_elements else None,
            "sampled_full_endpoint_mse": full_sum / full_elements if full_elements else None,
            "jvp_rms": math.sqrt(jvp_squared / total_elements),
            "grad_norm_before_clip": gradient_norm,
            "clipped": clipped,
            "clipping_frequency": clipping_count / update,
            "finite": True,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        _append(directory / "training.jsonl", record)
        if update % 100 == 0:
            print(
                f'{arm} update={update}/{PILOT_STEPS} global={record["global_step"]} loss={record["raw_meanflow_mse"]:.5f} grad={gradient_norm:.3f} clip_rate={clipping_count/update:.3f} elapsed={record["elapsed_seconds"]:.1f}s',
                flush=True,
            )
        if update % 250 == 0:
            for value in fixed_probe(model, probe_rows, probe_items, collator, device):
                _append(
                    directory / "validation_probes.jsonl",
                    {"pilot_update": update, "global_step": record["global_step"], **value},
                )
        if update in CHECKPOINT_UPDATES:
            save_checkpoint(output, arm, model, optimizer, scheduler, plan, update, initial)
    assert update == PILOT_STEPS
    assert parameter_names == [name for name, _ in model.named_parameters()]
    _json(
        directory / "TRAINING_COMPLETE.json",
        {
            "updates": PILOT_STEPS,
            "final_global_step": initial["global_step"] + PILOT_STEPS,
            "final_epoch": int(plan["epochs"][-1]),
            "clipped_steps": clipping_count,
            "clip_frequency": clipping_count / PILOT_STEPS,
            "numeric_failures": 0,
            "first_data_hash": first_data,
            "final_model_hash": model_hash(model),
        },
    )
    _gpu_state(directory / "gpu_end.txt")
    del loader, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


def preflight(output, device):
    """真实训练数据、最长padding形状反传；不执行任何优化器更新。"""
    import torch.nn.functional as F

    torch.set_num_threads(2)
    plan = safe_load(output / "common_plan.pt")
    model, optimizer, scheduler, config, audit = restore(BASE_CHECKPOINT, device)
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


# ====================
# 6. Public API
# ====================


def run(output, stage, device, arm=None, workers=8):
    """调度双臂 pilot 的准备、预检或指定训练臂。"""
    if stage == "prepare":
        prepare(output)
    elif stage == "preflight":
        preflight(output, device)
    elif stage == "train":
        if arm == "both":
            for name in ["control", "intervention"]:
                train(output, name, device, workers)
        else:
            train(output, arm, device, workers)
