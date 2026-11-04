"""模块说明：核验端点干预隔离与完整有效帧加权梯度累计，不加载预训练模型。"""

# ====================
# 1. Imports
# ====================

from copy import deepcopy

import torch

from src.analysis.endpoint_pilot.data import make_plan, planned_times
from src.analysis.endpoint_pilot.training import (
    microbatch,
    native_planned_loss,
    safe_load,
    save_checkpoint,
)
from src.models.components.conditioning.meanvc import CachedConditionEncoder
from src.models.methods.generative.flow.mean_flow.conditional_mean_flow import (
    ConditionalMeanFlow,
)
from src.models.networks.unet.conditional import UNet
from src.optim.warmup_cosine import WarmupCosineLR

# ====================
# 6. Public API
# ====================


def test_intervention_only_replaces_nonzero_time_pairs():
    plan = make_plan(35234, 382)
    changed = total = zero = 0
    for step in range(len(plan["batch_sizes"])):
        r0, t0 = planned_times(plan, step, "control")
        r1, t1 = planned_times(plan, step, "intervention")
        gate = plan["endpoint_mask"][step, : len(r0)]
        assert torch.equal(r0 == t0, r1 == t1)
        assert torch.equal(r0[~gate], r1[~gate]) and torch.equal(t0[~gate], t1[~gate])
        assert (r1[gate] == 0).all() and (t1[gate] == 1).all()
        assert (r0[gate] < t0[gate]).all()
        zero += int((r0 == t0).sum())
        changed += int(gate.sum())
        total += len(r0)
    assert abs(zero / total - 0.75) < 0.005
    assert abs(changed / total - 0.02) < 0.003
    assert total == 159880
    assert int(plan["epochs"][0]) == 383 and int(plan["epochs"][-1]) == 387


def test_microbatch_accumulation_matches_full_native_masked_gradient():
    torch.manual_seed(23)
    model = ConditionalMeanFlow(
        UNet(
            n_mels=2,
            condition_dim=4,
            hidden_channels=8,
            time_embedding_dim=8,
            time_fields=["time", "interval"],
        ),
        CachedConditionEncoder(content_dim=3, speaker_dim=2, output_dim=4),
    )
    other = deepcopy(model)
    batch = {
        "mel_spectrograms": torch.randn(4, 2, 160),
        "mel_attention_mask": torch.arange(160)[None]
        < torch.tensor([160, 133, 129, 153])[:, None],
        "content_features": torch.randn(4, 3, 80),
        "content_feature_lengths": torch.tensor([80, 70, 67, 76]),
        "speaker_features": torch.randn(4, 2),
    }
    r, t = torch.tensor([0.3, 0.2, 0.0, 0.7]), torch.tensor([0.3, 0.8, 1.0, 0.7])
    full = native_planned_loss(model, batch, r, t, torch.Generator().manual_seed(112))[0]
    full.backward()
    count = batch["mel_attention_mask"].sum() * 2
    generator = torch.Generator().manual_seed(112)
    losses = []
    for start in [0, 2]:
        small = microbatch(batch, start, start + 2)
        value = native_planned_loss(
            other, small, r[start : start + 2], t[start : start + 2], generator
        )[0]
        weighted = value * (small["mel_attention_mask"].sum() * 2 / count)
        weighted.backward()
        losses.append(weighted.detach())
    torch.testing.assert_close(sum(losses), full.detach(), atol=1e-5, rtol=1e-5)
    for left, right in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=2e-4)


def test_last_checkpoint_is_real_latest_update_and_preserves_optimizer(tmp_path):
    model = torch.nn.Linear(1, 1)
    setattr(model, "hparams", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupCosineLR(optimizer, total_steps=10, warmup_steps=0)
    plan = {
        "epochs": torch.tensor([3, 3]),
        "batch_indices": torch.tensor([0, 1]),
        "base_global_step": 0,
    }
    torch.save(plan, tmp_path / "common_plan.pt")
    paths = []
    for update in [1, 2]:
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
        paths.append(
            save_checkpoint(
                tmp_path,
                "control",
                model,
                optimizer,
                scheduler,
                plan,
                update,
                {"file_sha256": "0" * 64},
            )
        )
    first, final = safe_load(paths[0]), safe_load(tmp_path / "control/checkpoints/last.ckpt")
    assert first["global_step"] == 1 and final["global_step"] == 2
    assert final["epoch"] == 3 and final["pilot_batch_in_epoch"] == 1
    assert not torch.equal(first["state_dict"]["weight"], final["state_dict"]["weight"])
    assert all(int(s["step"]) == 2 for s in final["optimizer_states"][0]["state"].values())
    assert final["lr_schedulers"][0]["last_epoch"] == 2
