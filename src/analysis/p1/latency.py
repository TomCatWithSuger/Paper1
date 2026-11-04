"""模块说明：

- 在 GPU0 共享负载下交错评估六种配置，减少固定执行顺序的影响。
- 保存每条输入三次延迟、固定随机执行顺序与聚合分位数。
"""

# ====================
# 1. Imports
# ====================

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.analysis.p1.evaluation import CACHE, ROOT, RUNS, SEED, VCTK, _csv, _gpu_state
from src.analysis.unified.checkpoints import _load_model
from src.analysis.unified.data import _collator, _load_manifest_samples, _prepare_pair
from src.analysis.unified_vc_evaluation import _generate, _seed, _to_device

# ====================
# 5. Core Logic
# ====================


def _execute(output: Path):
    assert (output / "full/PASSED.json").is_file()
    target = output / "latency_interleaved"
    target.mkdir(exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    models = {
        name: _load_model(ROOT / "logs/train/runs" / run, device) for name, run in RUNS.items()
    }
    config = models["B1"][1]
    manifest, by_id = _load_manifest_samples(output / "test_manifest.csv", VCTK, "mic1")
    specs = [("B1", n) for n in [1, 5, 10, 50]] + [("B2", 1), ("B3", 1)]
    rng = np.random.default_rng(20261024)
    collator = _collator(config)
    warm, _, _ = _prepare_pair(manifest[0], by_id, config, CACHE, collator)
    warm = _to_device(warm, device)
    with torch.inference_mode():
        for name, nfe in specs:
            for _ in range(3):
                _generate(models[name][0], warm, nfe)
    torch.cuda.synchronize(device)
    _gpu_state(target / "gpu_before.txt")
    records = []
    for index, row in enumerate(manifest):
        batch, _, samples = _prepare_pair(row, by_id, config, CACHE, collator)
        batch = _to_device(batch, device)
        for repeat in range(3):
            for order, spec_index in enumerate(rng.permutation(len(specs))):
                name, nfe = specs[spec_index]
                _seed(SEED + int(row["index"]), device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = _generate(models[name][0], batch, nfe)
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - started) * 1000
                records.append(
                    {
                        "experiment_id": f"nfe_{name.lower()}_{nfe:02d}",
                        "source_id": row["source_id"],
                        "reference_id": row["reference_id"],
                        "repeat": repeat,
                        "order": order,
                        "latency_ms": elapsed,
                        "audio_seconds": samples / 22050,
                        "rtf": elapsed / 1000 / (samples / 22050),
                    }
                )
                del generated
        if (index + 1) % 10 == 0:
            print(f"交错延迟 {index + 1}/{len(manifest)}", flush=True)
    aggregates = []
    for experiment in sorted({r["experiment_id"] for r in records}):
        selected = [r for r in records if r["experiment_id"] == experiment]
        values = np.array([r["latency_ms"] for r in selected])
        aggregates.append(
            {
                "experiment_id": experiment,
                "calls": len(selected),
                "mean_ms": values.mean(),
                "p50_ms": np.percentile(values, 50),
                "p95_ms": np.percentile(values, 95),
                "rtf_mean": np.mean([r["rtf"] for r in selected]),
            }
        )
    _csv(target / "per_call.csv", records)
    _csv(target / "aggregate.csv", aggregates)
    _gpu_state(target / "gpu_after.txt")
    (target / "protocol.json").write_text(
        json.dumps(
            {
                "device": "cuda:0",
                "shared_gpu": True,
                "repeats": 3,
                "order_seed": 20261024,
                "scope": "缓存条件到Mel；含条件投影和生成；不含CPU->GPU传输、离线编码器、声码器；3次首样本预热",
                "timing_source_sha256": __import__("hashlib")
                .sha256(Path(__file__).read_bytes())
                .hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


# ====================
# 6. Public API
# ====================


def run(output: Path):
    """对固定输入交错执行六种配置，记录共享 GPU 上的生成延迟。"""
    return _execute(output.resolve())
