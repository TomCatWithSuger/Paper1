"""模块说明：

- 对冻结 B2 统计固定区间的原始误差，并用真实批次核验 JVP。
- 通过临时只读采集分解 B3 原生损失，不改变计算目标与权重。
"""

# ====================
# 1. Imports
# ====================

import json
from itertools import zip_longest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.p1.evaluation import CACHE, _csv, _json
from src.analysis.p2.common import (
    COMMON_FRAMES,
    GRID,
    MICRO_BATCH,
    REPEATS,
    batch_for,
    load_checkpoint,
    load_manifest,
    load_samples,
    prepared_items,
    seed_for,
)
from src.analysis.unified.data import _collator, _dataset

# ====================
# 3. Utilities
# ====================


def masked_values(values, mask, common=False):
    """每个样本按实际有效元素归一化；共同窗口固定128帧，不截断模型上下文。"""
    results = []
    for i in range(len(values)):
        frames = int(mask[i].sum())
        assert frames >= COMMON_FRAMES
        start = (frames - COMMON_FRAMES) // 2 if common else 0
        end = start + COMMON_FRAMES if common else frames
        results.append(values[i, :, start:end].mean())
    return torch.stack(results)


def fd_statistics(automatic, finite, mask):
    """在有效帧上比较自动 JVP 与有限差分的相对误差和余弦。"""
    rows = []
    for i in range(len(automatic)):
        a = automatic[i, :, mask[i]].double().flatten()
        b = finite[i, :, mask[i]].double().flatten()
        rows.append(
            {
                "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-12)),
                "absolute_rmse": float((a - b).square().mean().sqrt()),
                "jvp_rms": float(a.square().mean().sqrt()),
                "cosine": float(F.cosine_similarity(a, b, dim=0)),
            }
        )
    return rows


def _append(path, rows):
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# ====================
# 4. Internal Functions
# ====================


def b2_intervals(output, device, limit=None, grid=None, directory_name=None, check_fd=True):
    """冻结 B2 按时间网格评估残差，并按需进行有限差分核查。"""
    manifest = load_manifest(output)
    if limit:
        manifest = manifest[:limit]
    config, _, by_id = load_samples()
    items = prepared_items(manifest, config, by_id)
    collator = _collator(config)
    model, _, identity = load_checkpoint("B2", "epoch_382.ckpt", device)
    directory = output / (directory_name or ("smoke_intervals" if limit else "d1"))
    directory.mkdir(exist_ok=False)
    _json(directory / "model.json", identity)
    records = []
    for i, row in enumerate(manifest):
        batch = batch_for([row], items, collator, device)
        x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
        with torch.no_grad():
            condition = model.method.encode_condition(
                batch["content_features"],
                batch["content_feature_lengths"],
                batch["speaker_features"],
                mask,
            )
            for repeat in range(REPEATS):
                generator = torch.Generator(device=device).manual_seed(
                    seed_for(row["index"], repeat)
                )
                noise = torch.randn(x.shape, generator=generator, device=device)
                velocity = noise - x
                for name, r, t in (grid if grid is not None else GRID):
                    rs = x.new_full((1,), r)
                    ts = x.new_full((1,), t)
                    z = model.method.interpolate(x, noise, ts)

                    def function(states, times, starts):
                        return model.method(states, starts, times, condition, mask)

                    prediction, derivative = torch.func.jvp(
                        function,
                        (z, ts, rs),
                        (velocity, torch.ones_like(ts), torch.zeros_like(rs)),
                    )
                    correction = (t - r) * derivative
                    target = velocity - correction
                    error = (prediction - target).square()
                    velocity_error = (prediction - velocity).square()
                    record = {
                        **row,
                        "repeat": repeat,
                        "seed": seed_for(row["index"], repeat),
                        "interval_group": name,
                        "r": r,
                        "t": t,
                        "delta": t - r,
                        "valid_frames": int(mask.sum()),
                        "common_frames": COMMON_FRAMES,
                    }
                    for key, values in [
                        ("raw_mse", error),
                        ("velocity_mse", velocity_error),
                        ("correction_squared", correction.square()),
                        ("target_energy", target.square()),
                        ("velocity_energy", velocity.square()),
                        ("cross_term", 2 * (prediction - velocity) * correction),
                    ]:
                        record[key] = float(masked_values(values, mask)[0])
                        record[f"{key}_common"] = float(masked_values(values, mask, True)[0])
                    record["raw_over_target_energy"] = record["raw_mse"] / max(
                        record["target_energy"], 1e-12
                    )
                    record["jvp_correction_rms"] = record["correction_squared"] ** 0.5
                    record["terminal_reconstruction_l1"] = (
                        float(masked_values((noise - prediction - x).abs(), mask)[0])
                        if name == "full"
                        else float("nan")
                    )
                    torch.testing.assert_close(
                        error.mean(),
                        (
                            velocity_error
                            + correction.square()
                            + 2 * (prediction - velocity) * correction
                        ).mean(),
                    )
                    records.append(record)
                    _append(directory / "records.jsonl", [record])
        if (i + 1) % 8 == 0 or i + 1 == len(manifest):
            print(f"D1区间 {i+1}/{len(manifest)}", flush=True)
    _csv(directory / "per_sample_intervals.csv", records)
    if not check_fd:
        return
    # 固定16条、覆盖64条清单的不同位置；真实变长批次而非合成输入。
    chosen = [
        manifest[i] for i in np.linspace(0, len(manifest) - 1, min(16, len(manifest)), dtype=int)
    ]
    checks = []
    with torch.no_grad():
        for start in range(0, len(chosen), MICRO_BATCH):
            rows = chosen[start : start + MICRO_BATCH]
            batch = batch_for(rows, items, collator, device)
            x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
            condition = model.method.encode_condition(
                batch["content_features"],
                batch["content_feature_lengths"],
                batch["speaker_features"],
                mask,
            )
            generator = torch.Generator(device=device).manual_seed(8192 + start)
            noise = torch.randn(x.shape, generator=generator, device=device)
            velocity = noise - x
            rs, ts = x.new_full((len(rows),), 0.2), x.new_full((len(rows),), 0.7)
            z = model.method.interpolate(x, noise, ts)

            def function(states, times, starts):
                return model.method(states, starts, times, condition, mask)

            _, derivative = torch.func.jvp(
                function, (z, ts, rs), (velocity, torch.ones_like(ts), torch.zeros_like(rs))
            )
            for epsilon in (0.01, 0.003, 0.001, 0.0003):
                finite = (
                    function(z + epsilon * velocity, ts + epsilon, rs)
                    - function(z - epsilon * velocity, ts - epsilon, rs)
                ) / (2 * epsilon)
                for row, stats in zip(rows, fd_statistics(derivative, finite, mask)):
                    checks.append(
                        {
                            "source_id": row["source_id"],
                            "source_speaker_id": row["source_speaker_id"],
                            "length_bin": row["length_bin"],
                            "epsilon": epsilon,
                            "r": 0.2,
                            "t": 0.7,
                            "batch_size": len(rows),
                            **stats,
                        }
                    )
    _csv(directory / "jvp_finite_difference.csv", checks)
    primary = [r for r in checks if r["epsilon"] == 0.001]
    _json(
        directory / "jvp_check.json",
        {
            "samples": len(primary),
            "primary_epsilon": 0.001,
            "max_relative_l2": max(r["relative_l2"] for r in primary),
            "min_cosine": min(r["cosine"] for r in primary),
            "passed": all(r["relative_l2"] < 0.05 and r["cosine"] > 0.99 for r in primary),
        },
    )
    print("D1区间与真实批JVP校验完成", flush=True)


def decompose_native(method, batch, generator):
    """调用原生 compute_loss，临时捕获中间量后恢复全部方法。"""
    captured = {}
    prior_fn, times_fn, ssim_fn = (
        method._build_training_prior,
        method._sample_time_intervals,
        method._masked_ssim_loss,
    )

    def prior(*args, **kwargs):
        result = prior_fn(*args, **kwargs)
        captured["prior"], captured["source_times"], captured["pseudo_mask"] = result
        return result

    def times(*args, **kwargs):
        result = times_fn(*args, **kwargs)
        captured["r"], captured["t"] = result
        return result

    def ssim(*args, **kwargs):
        captured["zero_prediction"] = args[0]
        return ssim_fn(*args, **kwargs)

    with (
        patch.object(method, "_build_training_prior", side_effect=prior),
        patch.object(method, "_sample_time_intervals", side_effect=times),
        patch.object(method, "_masked_ssim_loss", side_effect=ssim),
    ):
        total, prediction, target, _ = method.compute_loss(
            batch["mel_spectrograms"],
            batch["mel_attention_mask"],
            batch["content_features"],
            batch["content_feature_lengths"],
            batch["speaker_features"],
            generator=generator,
            speaker_ids=batch["speaker_ids"],
        )
    x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
    valid = mask[:, None].to(x.dtype)
    error = (prediction - target).square()
    energy = (error * valid).sum((1, 2))
    counts = mask.sum(1) * x.shape[1]
    adaptive = energy / (energy + method.adaptive_loss_epsilon)
    velocity = captured["prior"] - x
    correction = velocity - target
    ssim = torch.stack(
        [
            ssim_fn(captured["zero_prediction"][i : i + 1], x[i : i + 1], mask[i : i + 1])
            for i in range(len(x))
        ]
    )
    torch.testing.assert_close(
        total, adaptive.mean() + method.zero_reconstruction_weight * ssim.mean()
    )
    rows = []
    for i in range(len(x)):
        r, t = float(captured["r"][i]), float(captured["t"][i])
        rows.append(
            {
                "raw_mse": float(energy[i] / counts[i]),
                "raw_mse_common": float(masked_values(error, mask, True)[i]),
                "error_energy": float(energy[i]),
                "adaptive_loss": float(adaptive[i]),
                "adaptive_gradient_weight": float(1 / (energy[i] + method.adaptive_loss_epsilon)),
                "ssim_loss": float(ssim[i]),
                "margin_hit": float(ssim[i] <= method.zero_reconstruction_margin + 1e-6),
                "total_loss": float(adaptive[i] + method.zero_reconstruction_weight * ssim[i]),
                "native_batch_total": float(total),
                "jvp_correction_rms": float(masked_values(correction.square(), mask)[i].sqrt()),
                "velocity_mse": float(masked_values((prediction - velocity).square(), mask)[i]),
                "zero_reconstruction_l1": float(
                    masked_values((captured["zero_prediction"] - x).abs(), mask)[i]
                ),
                "pseudo": bool(captured["pseudo_mask"][i]),
                "source_time": float(captured["source_times"][i]),
                "r": r,
                "t": t,
                "delta": t - r,
                "valid_frames": int(mask[i].sum()),
                "batch_size": len(x),
            }
        )
    return rows


def b3_losses(output, device, limit=None):
    """审计 B3 原生损失分量、伪源覆盖及 best/last 权重关系。"""
    manifest = load_manifest(output)
    if limit:
        manifest = manifest[:limit]
    config, split, by_id = load_samples()
    items = prepared_items(manifest, config, by_id)
    collator = _collator(config)
    directory = output / ("smoke_losses" if limit else "d2")
    directory.mkdir(exist_ok=False)
    records = []
    for name in ("epoch_040.ckpt", "last.ckpt"):
        model, _, identity = load_checkpoint("B3", name, device)
        label = "best" if name.startswith("epoch") else "last"
        _json(directory / f"{label}_model.json", identity)
        with torch.no_grad():
            for start in range(0, len(manifest), MICRO_BATCH):
                rows = manifest[start : start + MICRO_BATCH]
                batch = batch_for(rows, items, collator, device)
                for repeat in range(REPEATS):
                    generator = torch.Generator(device=device).manual_seed(seed_for(start, repeat))
                    decomposed = decompose_native(model.method, batch, generator)
                    for row, value in zip(rows, decomposed):
                        records.append(
                            {
                                **row,
                                "checkpoint": label,
                                "repeat": repeat,
                                "batch_index": start // MICRO_BATCH,
                                "seed": seed_for(start, repeat),
                                **value,
                            }
                        )
                if (start + len(rows)) % 16 == 0 or start + len(rows) == len(manifest):
                    print(f"D2损失 {label} {start+len(rows)}/{len(manifest)}", flush=True)
        # 复现原验证loader的说话人交错顺序；只读元数据，不产生训练步骤。
        grouped = {}
        for sample in split.val:
            grouped.setdefault(sample.speaker_id, []).append(sample)
        ordered = [
            s
            for group in zip_longest(*(grouped[k] for k in sorted(grouped)))
            for s in group
            if s is not None
        ]
        pair_rows = []
        for start in range(0, len(ordered), 32):
            ids = [s.speaker_id for s in ordered[start : start + 32]]
            features = torch.zeros(len(ids), 256, device=device)
            _, valid_pairs = model.method._shuffle_speaker_features(
                features, ids, torch.Generator(device=device).manual_seed(2000 + start)
            )
            requested = int(len(ids) * model.method.conditional_input_probability)
            selected = min(requested, int(valid_pairs.sum()))
            pair_rows.append(
                {
                    "checkpoint": label,
                    "batch_index": start // 32,
                    "batch_size": len(ids),
                    "distinct_speakers": len(set(ids)),
                    "largest_speaker_group": max(ids.count(k) for k in set(ids)),
                    "valid_pair_candidates": int(valid_pairs.sum()),
                    "requested_pseudo": requested,
                    "effective_pseudo": selected,
                    "fallback_pure_noise": requested - selected,
                    "intentional_pure_noise": len(ids) - requested,
                    "speaker_ids": "|".join(ids),
                }
            )
        _csv(directory / f"{label}_original_val_pairing_audit.csv", pair_rows)
        # 对原loader首/中/末批真正运行先验构造，核对元数据统计。
        actual = []
        full_ds = _dataset(split.val, config, CACHE)
        val_index = {sample.utterance_id: i for i, sample in enumerate(split.val)}
        for batch_index in sorted({0, len(pair_rows) // 2, len(pair_rows) - 1}):
            samples = ordered[batch_index * 32 : (batch_index + 1) * 32]
            batch = collator([full_ds[val_index[s.utterance_id]] for s in samples])
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            x = batch["mel_spectrograms"]
            generator = torch.Generator(device=device).manual_seed(9999 + batch_index)
            noise = torch.randn(x.shape, device=device, generator=generator)
            with torch.no_grad():
                _, times, effective = model.method._build_training_prior(
                    noise,
                    batch["content_features"],
                    batch["content_feature_lengths"],
                    batch["speaker_features"],
                    batch["mel_attention_mask"],
                    generator,
                    batch["speaker_ids"],
                )
            assert int(effective.sum()) == pair_rows[batch_index]["effective_pseudo"]
            actual.append(
                {
                    "checkpoint": label,
                    "batch_index": batch_index,
                    "batch_size": len(samples),
                    "effective_pseudo": int(effective.sum()),
                    "source_times": times.cpu().tolist(),
                }
            )
        _json(directory / f"{label}_actual_prior_checks.json", actual)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _csv(directory / "loss_components.csv", records)
    print("D2原生损失分解与实际伪源核验完成", flush=True)


# ====================
# 6. Public API
# ====================


def run(output, stage, device, limit=None):
    """分派普通区间、匹配中点、边界或 B3 损失分解诊断。"""
    torch.set_num_threads(2)
    if stage == "d1":
        b2_intervals(output, device, limit)
    elif stage == "d1_center":
        grid = [
            ("zero", 0.5, 0.5),
            ("short", 0.45, 0.55),
            ("medium", 0.3, 0.7),
            ("long", 0.1, 0.9),
            ("full", 0.0, 1.0),
        ]
        b2_intervals(output, device, grid=grid, directory_name="d1_center_matched", check_fd=False)
    elif stage == "d1_boundary":
        grid = [
            ("zero", 1.0, 1.0),
            ("short", 0.9, 1.0),
            ("medium", 0.6, 1.0),
            ("long", 0.2, 1.0),
            ("full", 0.0, 1.0),
            ("zero", 0.0, 0.0),
            ("short", 0.0, 0.1),
            ("medium", 0.0, 0.4),
            ("long", 0.0, 0.8),
        ]
        b2_intervals(output, device, grid=grid, directory_name="d1_boundary", check_fd=False)
    elif stage == "d2":
        b3_losses(output, device, limit)
