"""模块说明：

- 固定5000步末期权重，在P2固定与保留验证队列比较两个训练臂。
- 使用同一冻结链路评分，不选择中间权重、不读取测试语音。
"""

# ====================
# 1. Imports
# ====================

import json
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.version import cuda as torch_cuda_version

from src.analysis.endpoint_pilot.training import BASE_CHECKPOINT, safe_load
from src.analysis.p1.evaluation import (
    ROOT,
    _aggregate,
    _csv,
    _json,
    _save_audio,
    _score_audio,
)
from src.analysis.p2.common import batch_for, load_samples, prepared_items, seed_for
from src.analysis.p2.intervals import masked_values
from src.analysis.power_mel_adapter import from_vocos_weights
from src.analysis.source_prosody import compare_tracks, pitch_track
from src.analysis.unified.checkpoints import _model_config
from src.analysis.unified.data import _collator
from src.analysis.unified.identity import _sha256_file
from src.analysis.unified_vc_evaluation import _seed
from src.analysis.vocoder.models import _AsrScorer, _load_vocos, _SpeakerScorer
from src.models.algorithms.autodiff import tensor_jvp

# ====================
# 3. Utilities
# ====================


def _rows(path):
    return (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if path.exists()
        else []
    )


def _append(path, value):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load(path, device):
    config = _model_config(BASE_CHECKPOINT.parent.parent)
    model = hydra.utils.instantiate(OmegaConf.create(config["model"]))
    checkpoint = safe_load(path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return (
        model,
        config,
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "global_step": checkpoint["global_step"],
            "epoch": checkpoint["epoch"],
            "pilot_updates": checkpoint.get("pilot_updates", 0),
        },
    )


# ====================
# 4. Internal Functions
# ====================


def generate(output, device):
    """为初始及双臂权重生成固定与保留队列的 Mel，同时记录端点诊断。"""
    directory = output / "evaluation"
    directory.mkdir(exist_ok=True)
    identity = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    for filename, digest in identity["resources"].items():
        assert _sha256_file(Path(filename)) == digest, filename
    from src.analysis.endpoint_pilot.data import tensor_hash

    sources = {}
    for path in [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "scripts/analysis/run_endpoint_pilot.py",
    ]:
        relative = path.relative_to(ROOT)
        sources[str(relative)] = _sha256_file(path)
        destination = directory / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    _json(
        directory / "runtime.json",
        {
            "torch": torch.__version__,
            "cuda": torch_cuda_version,
            "device": str(device),
            "tf32": False,
            "source_sha256": sources,
            "fixed_final_update": 5000,
        },
    )
    for arm in ["control", "intervention"]:
        assert (
            json.loads((output / arm / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))[
                "updates"
            ]
            == 5000
        )
    configs = [
        ("initial", BASE_CHECKPOINT),
        ("control", output / "control/checkpoints/last.ckpt"),
        ("intervention", output / "intervention/checkpoints/last.ckpt"),
    ]
    config, _, by_id = load_samples()
    collator = _collator(config)
    completed = {
        (r["cohort"], r["experiment_id"], r["source_id"], r["repeat"])
        for r in _rows(directory / "generation.jsonl")
    }
    diagnostic_completed = {
        (r["cohort"], r["experiment_id"], r["source_id"], r["repeat"], r["interval"])
        for r in _rows(directory / "endpoint_diagnostics.jsonl")
    }
    identities = {}
    for cohort, filename in [
        ("p2_fixed", "p2_validation_manifest.json"),
        ("reserved", "reserved_validation_manifest.json"),
    ]:
        manifest = json.loads((output / filename).read_text(encoding="utf-8"))
        items = prepared_items(manifest, config, by_id)
        item_hashes = {
            uid: {
                key: tensor_hash(item[key])
                for key in ["waveform", "content_features", "speaker_features"]
            }
            for uid, item in items.items()
        }
        _json(directory / f"{cohort}_actual_inputs.json", item_hashes)
        for arm, checkpoint in configs:
            model, _, identity = _load(checkpoint, device)
            identities[arm] = identity
            if arm != "initial":
                assert identity["pilot_updates"] == 5000 and identity["global_step"] == 427066
            for index, row in enumerate(manifest):
                batch = batch_for([row], items, collator, device)
                source, mask = batch["mel_spectrograms"], batch["mel_attention_mask"]
                with torch.no_grad():
                    conversion_condition = model.method.encode_condition(
                        batch["content_features"],
                        batch["content_feature_lengths"],
                        items[row["reference_id"]]["speaker_features"][None].to(device),
                        mask,
                    )
                    self_condition = model.method.encode_condition(
                        batch["content_features"],
                        batch["content_feature_lengths"],
                        batch["speaker_features"],
                        mask,
                    )
                for repeat in range(2):
                    seed = seed_for(row["index"], repeat)
                    key = (cohort, arm, row["source_id"], repeat)
                    if key not in completed:
                        _seed(seed, device)
                        with torch.inference_mode():
                            generated = model.method.sample(
                                conversion_condition, mask, integration_steps=1
                            )[0].cpu()
                        assert torch.isfinite(generated).all()
                        path = (
                            directory
                            / "generated_mels"
                            / cohort
                            / arm
                            / f"{row['source_id']}__seed{seed}.pt"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "mel": generated.half(),
                                "seed": seed,
                                "source_id": row["source_id"],
                                "reference_id": row["reference_id"],
                            },
                            path,
                        )
                        _append(
                            directory / "generation.jsonl",
                            {
                                **row,
                                "cohort": cohort,
                                "experiment_id": arm,
                                "repeat": repeat,
                                "seed": seed,
                                "generated_path": str(path),
                            },
                        )
                    with torch.no_grad():
                        noise = torch.randn(
                            source.shape,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(seed),
                        )
                        velocity = noise - source
                        for interval, r, t in [("zero", 0.5, 0.5), ("full", 0.0, 1.0)]:
                            if (*key, interval) in diagnostic_completed:
                                continue
                            rs, ts = source.new_full((1,), r), source.new_full((1,), t)
                            z = model.method.interpolate(source, noise, ts)

                            def f(states, ends, starts):
                                assert model is not None
                                return model.method(states, starts, ends, self_condition, mask)

                            prediction, derivative = tensor_jvp(
                                f,
                                (z, ts, rs),
                                (velocity, torch.ones_like(ts), torch.zeros_like(rs)),
                            )
                            correction = (t - r) * derivative
                            target = velocity - correction
                            error = (prediction - target).square()
                            record = {
                                **row,
                                "cohort": cohort,
                                "experiment_id": arm,
                                "repeat": repeat,
                                "seed": seed,
                                "interval": interval,
                                "valid_frames": int(mask.sum()),
                                "raw_mse": float(masked_values(error, mask)[0]),
                                "raw_mse_common": float(masked_values(error, mask, True)[0]),
                                "jvp_rms": float(
                                    masked_values(correction.square(), mask)[0].sqrt()
                                ),
                                "velocity_mse": float(
                                    masked_values((prediction - velocity).square(), mask)[0]
                                ),
                                "target_energy": float(masked_values(target.square(), mask)[0]),
                                "reconstruction_l1": (
                                    float(
                                        masked_values((noise - prediction - source).abs(), mask)[0]
                                    )
                                    if interval == "full"
                                    else None
                                ),
                            }
                            assert np.isfinite(record["raw_mse"])
                            _append(directory / "endpoint_diagnostics.jsonl", record)
                if (index + 1) % 16 == 0:
                    print(
                        f"pilot生成/端点 {cohort} {arm} {index+1}/{len(manifest)}",
                        flush=True,
                    )
            model = None
            torch.cuda.empty_cache()
    _json(directory / "models.json", identities)
    generated = _rows(directory / "generation.jsonl")
    diagnostics = _rows(directory / "endpoint_diagnostics.jsonl")
    assert len(generated) == 768 and len(diagnostics) == 1536
    _csv(directory / "generation.csv", generated)
    _csv(directory / "endpoint_diagnostics.csv", diagnostics)


def score(output, device):
    """用冻结声码器和本地评分器评估已保存的双队列生成结果。"""
    directory = output / "evaluation"
    config, _, by_id = load_samples()
    manifests = {
        name: json.loads((output / filename).read_text(encoding="utf-8"))
        for name, filename in [
            ("p2_fixed", "p2_validation_manifest.json"),
            ("reserved", "reserved_validation_manifest.json"),
        ]
    }
    manifest_rows = [r for rows in manifests.values() for r in rows]
    items = prepared_items(manifest_rows, config, by_id)
    asr = _AsrScorer(ROOT / "pretrained/whisper-small", device)
    assert asr.available
    asr.model.requires_grad_(False)
    speaker = _SpeakerScorer(
        ROOT / "pretrained/meanvc/ecapa_tdnn.py",
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
        ROOT / "pretrained/meanvc/wavlm_large.pt",
        device,
    )
    vocos = _load_vocos(
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
    ).to(device)
    adapter = from_vocos_weights(ROOT / "pretrained/vocos/pytorch_model.bin")
    embeddings, tracks = {}, {}
    for uid, item in items.items():
        embeddings[uid] = speaker.embedding(item["waveform"], uid)
        tracks[uid] = pitch_track(item["waveform"])
        path = directory / "prosody_tracks/real"
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / f"{uid}.npz", **tracks[uid])
    ledger = directory / "scores.jsonl"
    results = _rows(ledger)
    completed = {(r["cohort"], r["experiment_id"], r["source_id"], r["repeat"]) for r in results}
    generation = _rows(directory / "generation.jsonl")
    assert len(generation) == 768
    for index, row in enumerate(generation):
        key = (row["cohort"], row["experiment_id"], row["source_id"], row["repeat"])
        if key in completed:
            continue
        payload = safe_load(row["generated_path"])
        with torch.inference_mode():
            waveform = vocos.decode(adapter(payload["mel"]).unsqueeze(0).to(device))[0].cpu()
        assert torch.isfinite(waveform).all() and waveform.square().mean() > 1e-12
        source_id, reference_id = row["source_id"], row["reference_id"]
        path = (
            directory
            / "generated_wavs"
            / row["cohort"]
            / row["experiment_id"]
            / f"{source_id}__seed{row['seed']}.wav"
        )
        _save_audio(path, waveform)
        result = {
            **row,
            **_score_audio(
                waveform,
                row["source_text"],
                asr,
                speaker,
                embeddings[source_id],
                embeddings[reference_id],
            ),
            "wav_path": str(path),
        }
        result["target_preference_gap"] = (
            result["target_speaker_similarity"] - result["source_speaker_similarity"]
        )
        result["target_preferred"] = float(result["target_preference_gap"] > 0)
        track = pitch_track(waveform)
        for prefix, uid in [("source", source_id), ("reference", reference_id)]:
            result.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in compare_tracks(
                        track, tracks[uid], normalized_time=prefix == "reference"
                    ).items()
                }
            )
        track_dir = directory / "prosody_tracks" / row["cohort"] / row["experiment_id"]
        track_dir.mkdir(parents=True, exist_ok=True)
        np.savez(track_dir / f"{source_id}__seed{row['seed']}.npz", **track)
        _append(ledger, result)
        results.append(result)
        if (index + 1) % 20 == 0 or index + 1 == len(generation):
            print(f"pilot波形评分 {index+1}/{len(generation)}", flush=True)
    assert len(results) == 768
    _csv(directory / "per_sample_metrics.csv", results)
    _json(
        directory / "EVALUATION_COMPLETE.json",
        {
            "records": len(results),
            "cohorts": ["p2_fixed", "reserved"],
            "pairs_per_cohort": 64,
            "noise_repeats": 2,
            "trained_arms": 2,
            "initial_is_inference_baseline_only": True,
        },
    )


# ====================
# 6. Public API
# ====================


def run(output, stage, device):
    """分派冻结模型的生成、评分或双臂报告阶段。"""
    torch.set_num_threads(2)
    if stage == "evaluate":
        generate(output, device)
        score(output, device)
    elif stage == "report":
        from src.analysis.endpoint_pilot.reporting import run as report

        report(output)
