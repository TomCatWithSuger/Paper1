"""模块说明：

- 准备与训练/测试互斥的原始验证样本、固定转换配对及诊断协议。
- 严格读取指定权重，保存数据、模型、配置与代码指纹。
"""

# ====================
# 1. Imports
# ====================

import json
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from src.analysis.p1.evaluation import CACHE, ROOT, RUNS, VCTK, _csv, _gpu_state, _json
from src.analysis.unified.checkpoints import CHECKPOINT_SAFE_GLOBALS, _model_config
from src.analysis.unified.data import _collator, _dataset
from src.analysis.unified.identity import _sha256_file
from src.data.components.vctk_splits import discover_vctk_samples

# ====================
# 2. Definitions
# ====================

BASE_SEED = 20261024
REPEATS = 2
COMMON_FRAMES = 128
MICRO_BATCH = 4
GRID = (
    [("zero", t, t) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    + [("short", r, r + 0.1) for r in (0.0, 0.2, 0.4, 0.6, 0.8)]
    + [("medium", r, r + 0.4) for r in (0.0, 0.2, 0.4, 0.6)]
    + [("long", r, r + 0.8) for r in (0.0, 0.1, 0.2)]
    + [("full", 0.0, 1.0)]
)
CONFIGS = [
    ("b1_anchor", "B1", "epoch_382.ckpt", "conversion", None),
    ("b2_base", "B2", "epoch_382.ckpt", "conversion", None),
    ("b2_content_donor", "B2", "epoch_382.ckpt", "content_donor", None),
    ("b2_speaker_source", "B2", "epoch_382.ckpt", "speaker_source", None),
    ("b3_best_s080", "B3", "epoch_040.ckpt", "conversion", 0.8),
    ("b3_best_s090", "B3", "epoch_040.ckpt", "conversion", 0.9),
    ("b3_best_s095", "B3", "epoch_040.ckpt", "conversion", 0.95),
    ("b3_best_s100", "B3", "epoch_040.ckpt", "conversion", 1.0),
    ("b3_last_s095", "B3", "last.ckpt", "conversion", 0.95),
]


# ====================
# 3. Utilities
# ====================


def seed_for(index, repeat):
    """按源序号和噪声重复编号生成跨配置共享的固定种子。"""
    return BASE_SEED + int(index) + int(repeat) * 100_000


def length_bin(seconds):
    """将语音时长映射为预定诊断长度分层。"""
    return "<3s" if seconds < 3 else "3-4s" if seconds < 4 else "4-6s" if seconds < 6 else ">=6s"


def load_checkpoint(label, name, device):
    """不依赖最佳权重查找规则，显式严格加载 best 或 last。"""
    run = ROOT / "logs/train/runs" / RUNS[label]
    config = _model_config(run)
    model = hydra.utils.instantiate(OmegaConf.create(config["model"]))
    path = run / "checkpoints" / name
    with torch.serialization.safe_globals(list(CHECKPOINT_SAFE_GLOBALS)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return (
        model,
        config,
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
            "strict_load": True,
        },
    )


def load_samples():
    """读取 B2 的实际运行配置和数据划分，建立语句索引。"""
    config = _model_config(ROOT / "logs/train/runs" / RUNS["B2"])
    samples = discover_vctk_samples(VCTK, "mic1")
    strategy = hydra.utils.instantiate(OmegaConf.create(config["data"]["split_strategy"]))
    return config, strategy.split(samples), {s.utterance_id: s for s in samples}


def load_manifest(output):
    """读取已经固定的验证配对清单，不重新抽样。"""
    return json.loads((output / "validation_manifest.json").read_text())


def prepared_items(manifest, config, by_id):
    """只缓存当前验证清单中的波形/特征，不读取测试波形。"""
    ids = sorted(
        {row[key] for row in manifest for key in ("source_id", "reference_id", "donor_id")}
    )
    ds = _dataset([by_id[uid] for uid in ids], config, CACHE)
    return {uid: ds[index] for index, uid in enumerate(ids)}


def batch_for(rows, items, collator, device):
    """从缓存样本构造源语音批次并移到指定设备。"""
    batch = collator([items[row["source_id"]] for row in rows])
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


# ====================
# 4. Internal Functions
# ====================


def prepare(output: Path):
    """固定互斥验证样本、转换配对、模型和资源指纹。"""
    output.mkdir(parents=True, exist_ok=False)
    config, split, by_id = load_samples()
    grouped = defaultdict(list)
    durations = {}
    for sample in split.val:
        grouped[sample.speaker_id].append(sample)
    speakers = sorted(grouped)
    chosen = [speakers[i] for i in np.linspace(0, len(speakers) - 1, 16, dtype=int)]
    # 在读取任何模型质量指标前，按说话人与时长分位固定 64 条源语音。
    selected = {}
    for speaker in chosen:
        candidates = []
        for sample in grouped[speaker]:
            seconds = sf.info(str(sample.audio_path)).duration
            durations[sample.utterance_id] = seconds
            if 2 <= seconds <= 10:
                candidates.append(sample)
        candidates.sort(key=lambda s: (durations[s.utterance_id], s.utterance_id))
        assert len(candidates) >= 4
        selected[speaker] = [
            candidates[int(q * (len(candidates) - 1))] for q in (0.125, 0.375, 0.625, 0.875)
        ]
        assert len({s.utterance_id for s in selected[speaker]}) == 4
    manifest = []
    all_val = sorted(split.val, key=lambda s: s.utterance_id)
    for rank in range(4):
        for speaker in chosen:
            source = selected[speaker][rank]
            donor = selected[speaker][rank ^ 1]
            assert donor.text != source.text
            candidates = [s for s in all_val if s.speaker_id != speaker and s.text == source.text]
            same_text = bool(candidates)
            if not candidates:
                candidates = [selected[s][rank] for s in chosen if s != speaker]
            reference = candidates[(chosen.index(speaker) + rank) % len(candidates)]
            ref_seconds = sf.info(str(reference.audio_path)).duration
            # 参考若过长，改用已固定、无截断的其他说话人语句。
            if ref_seconds > 10:
                reference = selected[chosen[(chosen.index(speaker) + 1) % len(chosen)]][rank]
                same_text = reference.text == source.text
            manifest.append(
                {
                    "index": len(manifest),
                    "source_id": source.utterance_id,
                    "source_speaker_id": speaker,
                    "sentence_id": source.sentence_id,
                    "reference_id": reference.utterance_id,
                    "reference_speaker_id": reference.speaker_id,
                    "donor_id": donor.utterance_id,
                    "source_text": source.text,
                    "donor_text": donor.text,
                    "reference_same_text": same_text,
                    "audio_seconds": durations[source.utterance_id],
                    "length_bin": length_bin(durations[source.utterance_id]),
                    "partition": "diagnostic_A" if rank < 2 else "confirmation_B",
                }
            )
    val_ids = {s.utterance_id for s in split.val}
    used = {r[k] for r in manifest for k in ("source_id", "reference_id", "donor_id")}
    assert used <= val_ids
    assert not used & {s.utterance_id for s in split.train + split.test}
    _json(output / "validation_manifest.json", manifest)
    _csv(output / "validation_manifest.csv", manifest)
    _csv(
        output / "original_validation_order.csv",
        [{"utterance_id": s.utterance_id, "speaker_id": s.speaker_id} for s in split.val],
    )
    protocol = {
        "seed": BASE_SEED,
        "repeats": REPEATS,
        "pairs": 64,
        "speakers": chosen,
        "split_counts": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "no_train_or_test_utterance_used": True,
        "validation_was_used_for_training_checkpoint_monitor": True,
        "sampling": "16等距说话人，每人时长12.5/37.5/62.5/87.5分位4条，2-10秒；前两条A、后两条B；全部参照/内容替换来自原始验证划分",
        "common_frames": COMMON_FRAMES,
        "interval_grid": GRID,
        "r_bins": [0, 0.25, 0.5, 0.75, 1.000001],
        "t_bins": [0, 0.25, 0.5, 0.75, 1.000001],
        "length_bins": ["<3s", "3-4s", "4-6s", ">=6s"],
        "fd_epsilons": [0.01, 0.003, 0.001, 0.0003],
        "fd_primary_epsilon": 0.001,
        "fd_gate": "相对L2<0.05且cosine>0.99；FP32，真实验证批，r=.2 t=.7",
        "configs": CONFIGS,
        "P1_chain": "analysis_results/p1_latest_20261024/full/runtime_config.json",
        "inference_noise": "同一配对两个固定噪声；所有配置共用；source_time先构造(1-s)*源Mel+s*noise，再传同一s",
        "causality": "区间误差为自重建路径诊断；与转换CER相关不等于因果；不改训练目标",
    }
    _json(output / "protocol.json", protocol)
    prior = json.loads((ROOT / "analysis_results/p1_latest_20261024/preflight.json").read_text())
    identity = {
        "resources": {},
        "caches": {},
        "audio": {},
        "training_configs": {},
        "checkpoint_files": {},
        "source": {},
    }
    for path, digest in prior["resource_files"].items():
        assert _sha256_file(Path(path)) == digest
        identity["resources"][path] = digest
    for uid in sorted(used):
        for kind, ndim in (("content", 2), ("speaker", 1)):
            path = CACHE / kind / f"{uid}.pt"
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            assert tensor.ndim == ndim and tensor.shape[0] == 256 and torch.isfinite(tensor).all()
            identity["caches"][str(path)] = _sha256_file(path)
        identity["audio"][str(by_id[uid].audio_path)] = _sha256_file(by_id[uid].audio_path)
    for label in RUNS:
        path = ROOT / "logs/train/runs" / RUNS[label] / ".hydra/config.yaml"
        identity["training_configs"][str(path)] = _sha256_file(path)
        shutil.copy2(path, output / f"{label}_training_config.yaml")
    for label, name in [
        ("B1", "epoch_382.ckpt"),
        ("B2", "epoch_382.ckpt"),
        ("B3", "epoch_040.ckpt"),
        ("B3", "last.ckpt"),
    ]:
        path = ROOT / "logs/train/runs" / RUNS[label] / "checkpoints" / name
        identity["checkpoint_files"][str(path)] = _sha256_file(path)
    snapshot(output, identity)
    _json(output / "identity.json", identity)
    _gpu_state(output / "gpu_prepare.txt")
    print(
        f"验证清单已冻结：64对、16源说话人；源/参考/替换共{len(used)}条，全部在原始3912条验证集，未读取测试波形。",
        flush=True,
    )


def snapshot(output, identity=None):
    """复制当前运行源码并记录指纹，保留阶段实现证据。"""
    identity = identity if identity is not None else {}
    identity["source"] = {}
    for path in [*sorted((ROOT / "src").rglob("*.py")), ROOT / "scripts/analysis/evaluate_p2.py"]:
        relative = path.relative_to(ROOT)
        identity["source"][str(relative)] = _sha256_file(path)
        dest = output / "code_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return identity["source"]


# ====================
# 6. Public API
# ====================


def run(output):
    """准备独立输出目录中的固定 P2 验证协议和清单。"""
    prepare(output.resolve())
