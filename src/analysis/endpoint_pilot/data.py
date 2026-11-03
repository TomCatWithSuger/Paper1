"""模块说明：

- 为两个端点覆盖训练臂生成共同的数据、裁剪、噪声和区间随机计划。
- 复用原生训练数据处理，保留整批padding后进行微批梯度累计。
"""

# ====================
# 1. Imports
# ====================

import hashlib

import torch
from torch.utils.data import Dataset

from src.analysis.p1.evaluation import CACHE
from src.analysis.unified.data import _collator
from src.data.vctk_datamodule import VCTKDataset

# ====================
# 2. Definitions
# ====================

PILOT_STEPS = 5000
BATCH_SIZE = 32
MICRO_BATCH = 32
PLAN_SEED = 20261024


# ====================
# 3. Utilities
# ====================


def tensor_hash(tensor):
    """对张量形状、精度类型及连续 CPU 内容计算 SHA-256。"""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_hash(batch):
    """合并训练批次中各条件张量的指纹，用于核验双臂输入一致性。"""
    digest = hashlib.sha256()
    for key in [
        "mel_spectrograms",
        "mel_attention_mask",
        "content_features",
        "content_feature_lengths",
        "speaker_features",
    ]:
        digest.update(key.encode())
        digest.update(tensor_hash(batch[key]).encode())
    return digest.hexdigest()


def planned_times(plan, step, arm, start=0, stop=None):
    """读取共同时间计划，仅在干预臂将指定样本替换为完整端点。"""
    stop = int(plan["batch_sizes"][step]) if stop is None else stop
    r = plan["r"][step, start:stop].clone()
    t = plan["t"][step, start:stop].clone()
    if arm == "intervention":
        full = plan["endpoint_mask"][step, start:stop]
        r[full], t[full] = 0.0, 1.0
    elif arm != "control":
        raise ValueError(arm)
    return r, t


# ====================
# 4. Internal Functions
# ====================


def make_plan(train_size, base_epoch, updates=PILOT_STEPS, batch_size=BATCH_SIZE):
    """从下一数据epoch起预生成共同流；不声称复原未保存的原始RNG。"""
    indices = torch.full((updates, batch_size), -1, dtype=torch.long)
    sizes = torch.zeros(updates, dtype=torch.long)
    epochs, batch_indices = [], []
    step, epoch = 0, base_epoch + 1
    while step < updates:
        generator = torch.Generator().manual_seed(PLAN_SEED + epoch)
        order = torch.randperm(train_size, generator=generator)
        for batch_index, start in enumerate(range(0, train_size, batch_size)):
            if step == updates:
                break
            selected = order[start : start + batch_size]
            indices[step, : len(selected)] = selected
            sizes[step] = len(selected)
            epochs.append(epoch)
            batch_indices.append(batch_index)
            step += 1
        epoch += 1
    generator = torch.Generator().manual_seed(PLAN_SEED + 7_000_000)
    first, second = torch.rand((updates, batch_size), generator=generator), torch.rand(
        (updates, batch_size), generator=generator
    )
    equal = torch.rand((updates, batch_size), generator=generator) < 0.75
    r, t = torch.minimum(first, second), torch.maximum(first, second)
    r = torch.where(equal, t, r)
    endpoint = (torch.rand((updates, batch_size), generator=generator) < 0.08) & ~equal
    # 三种流独立记录，额外端点抽样不会改变数据、裁剪或噪声。
    crop_seeds = torch.randint(1, 2**31 - 1, (updates, batch_size), generator=generator)
    noise_seeds = torch.randint(
        1,
        2**31 - 1,
        (updates, (batch_size + MICRO_BATCH - 1) // MICRO_BATCH),
        generator=generator,
    )
    return {
        "indices": indices,
        "batch_sizes": sizes,
        "epochs": torch.tensor(epochs),
        "batch_indices": torch.tensor(batch_indices),
        "r": r,
        "t": t,
        "endpoint_mask": endpoint,
        "crop_seeds": crop_seeds,
        "noise_seeds": noise_seeds,
        "micro_batch": MICRO_BATCH,
    }


class AuditedDataset(VCTKDataset):
    """只记录原生随机裁剪结果，不改变波形和缓存处理。"""

    def _load_audio(self, path):
        result = super()._load_audio(path)
        self.last_crop_start = result[1]
        return result


class PlannedDataset(Dataset):
    """用计划中的样本索引和CPU随机种子重放原生训练样本。"""

    def __init__(self, samples, config, plan):
        data = config["data"]
        self.dataset = AuditedDataset(
            samples=samples,
            sample_rate=int(data["sample_rate"]),
            max_duration_seconds=float(data["max_duration_seconds"]),
            randomize=True,
            reference_mode="same_speaker",
            feature_cache_dir=CACHE,
        )
        self.plan = plan

    def __len__(self):
        return self.plan["indices"].numel()

    def __getitem__(self, position):
        step, slot = divmod(position, self.plan["indices"].shape[1])
        index = int(self.plan["indices"][step, slot])
        assert index >= 0
        state = torch.random.get_rng_state()
        try:
            torch.default_generator.manual_seed(int(self.plan["crop_seeds"][step, slot]))
            item = self.dataset[index]
        finally:
            torch.random.set_rng_state(state)
        item["pilot_position"] = position
        item["pilot_crop_start"] = self.dataset.last_crop_start
        return item


class PlannedBatches:
    """保留原训练最后不足32条的批次，并支持从已完成更新处恢复。"""

    def __init__(self, plan, start=0, stop=None):
        self.plan, self.start = plan, start
        self.stop = len(plan["batch_sizes"]) if stop is None else stop

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        width = self.plan["indices"].shape[1]
        for step in range(self.start, self.stop):
            yield list(range(step * width, step * width + int(self.plan["batch_sizes"][step])))


class PilotCollator:
    """先按完整有效批次执行原生collate，微批累计不改变padding和mask。"""

    def __init__(self, config):
        self.collator = _collator(config)

    def __call__(self, items):
        batch = self.collator(items)
        batch["pilot_positions"] = [item["pilot_position"] for item in items]
        batch["pilot_crop_starts"] = [item["pilot_crop_start"] for item in items]
        return batch


# ====================
# 6. Public API
# ====================


def plan_counts(plan):
    """汇总共同计划的实际样本数、批次及端点覆盖次数。"""
    active = torch.arange(plan["indices"].shape[1])[None] < plan["batch_sizes"][:, None]
    zero = (plan["r"] == plan["t"]) & active
    endpoint = plan["endpoint_mask"] & active
    total = int(active.sum())
    return {
        "sample_draws": total,
        "zero_draws": int(zero.sum()),
        "control_nonzero_draws": total - int(zero.sum()),
        "intervention_original_nonzero_draws": total - int(zero.sum()) - int(endpoint.sum()),
        "intervention_endpoint_draws": int(endpoint.sum()),
        "zero_fraction": float(zero.sum() / active.sum()),
        "endpoint_fraction": float(endpoint.sum() / active.sum()),
    }
