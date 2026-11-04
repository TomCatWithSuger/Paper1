"""模块说明：以同一真实批次比较完整 FP32 与 TF32 的 JVP/有限差分。"""

# ====================
# 1. Imports
# ====================

import numpy as np
import torch

from src.analysis.p1.evaluation import _csv, _json
from src.analysis.p2.common import (
    batch_for,
    load_checkpoint,
    load_manifest,
    load_samples,
    prepared_items,
)
from src.analysis.p2.intervals import fd_statistics
from src.analysis.unified.data import _collator
from src.models.algorithms.autodiff import tensor_jvp

# ====================
# 6. Public API
# ====================


def run(output, device):
    torch.set_num_threads(2)
    rows = load_manifest(output)[:4]
    config, _, by_id = load_samples()
    items = prepared_items(rows, config, by_id)
    batch = batch_for(rows, items, _collator(config), device)
    model, _, _ = load_checkpoint("B2", "epoch_382.ckpt", device)
    x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
    records = []
    derivatives = {}
    with torch.no_grad():
        for enabled in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
            condition = model.method.encode_condition(
                batch["content_features"],
                batch["content_feature_lengths"],
                batch["speaker_features"],
                mask,
            )
            noise = torch.randn(
                x.shape,
                generator=torch.Generator(device=device).manual_seed(8192),
                device=device,
            )
            velocity = noise - x
            r, t = x.new_full((4,), 0.2), x.new_full((4,), 0.7)
            z = model.method.interpolate(x, noise, t)

            def function(states, ends, starts):
                return model.method(states, starts, ends, condition, mask)

            _, derivative = tensor_jvp(
                function, (z, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r))
            )
            derivatives[enabled] = derivative
            for epsilon in (0.01, 0.003, 0.001, 0.0003):
                finite = (
                    function(z + epsilon * velocity, t + epsilon, r)
                    - function(z - epsilon * velocity, t - epsilon, r)
                ) / (2 * epsilon)
                for row, stats in zip(rows, fd_statistics(derivative, finite, mask)):
                    records.append(
                        {
                            "source_id": row["source_id"],
                            "tf32": enabled,
                            "epsilon": epsilon,
                            **stats,
                        }
                    )
    _csv(output / "d1/precision_fd_comparison.csv", records)
    _csv(
        output / "d1/precision_jvp_drift.csv",
        [
            {"source_id": row["source_id"], **stats}
            for row, stats in zip(rows, fd_statistics(derivatives[False], derivatives[True], mask))
        ],
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_endpoint(output, device):
    """在t=1使用域内二阶后向差分，核验巨大端点JVP并非数值假象。"""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = load_manifest(output)
    rows = [manifest[i] for i in np.linspace(0, len(manifest) - 1, 16, dtype=int)]
    config, _, by_id = load_samples()
    items = prepared_items(rows, config, by_id)
    model, _, _ = load_checkpoint("B2", "epoch_382.ckpt", device)
    checks = []
    with torch.no_grad():
        for start in range(0, len(rows), 4):
            selected = rows[start : start + 4]
            batch = batch_for(selected, items, _collator(config), device)
            x, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
            c = model.method.encode_condition(
                batch["content_features"],
                batch["content_feature_lengths"],
                batch["speaker_features"],
                mask,
            )
            z = torch.randn(
                x.shape,
                generator=torch.Generator(device=device).manual_seed(8192 + start),
                device=device,
            )
            v = z - x
            r, t = x.new_zeros(len(selected)), x.new_ones(len(selected))

            def f(states, ends, starts):
                return model.method(states, starts, ends, c, mask)

            primal, derivative = tensor_jvp(
                f, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r))
            )
            for epsilon in (0.01, 0.003, 0.001, 0.0003):
                finite = (
                    3 * primal
                    - 4 * f(z - epsilon * v, t - epsilon, r)
                    + f(z - 2 * epsilon * v, t - 2 * epsilon, r)
                ) / (2 * epsilon)
                for row, stats in zip(selected, fd_statistics(derivative, finite, mask)):
                    checks.append(
                        {
                            "source_id": row["source_id"],
                            "r": 0.0,
                            "t": 1.0,
                            "epsilon": epsilon,
                            "scheme": "second_order_backward_inside_domain",
                            **stats,
                        }
                    )
    _csv(output / "d1/endpoint_finite_difference.csv", checks)
    primary = [r for r in checks if r["epsilon"] == 0.001]
    _json(
        output / "d1/endpoint_fd_check.json",
        {
            "n_sources": len(primary),
            "epsilon": 0.001,
            "max_relative_l2": max(r["relative_l2"] for r in primary),
            "min_cosine": min(r["cosine"] for r in primary),
            "passed": all(r["relative_l2"] < 0.05 and r["cosine"] > 0.99 for r in primary),
        },
    )
