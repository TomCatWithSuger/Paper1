"""模块说明：

- 固定验证配对和噪声，执行 B2 条件替换及 B3 checkpoint/源强度比较。
- 复用 P1 的冻结声码器、评分器和源/参考韵律定义，保存全部波形与逐样本结果。
"""

# ====================
# 1. Imports
# ====================

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.p1.evaluation import (
    ROOT,
    _aggregate,
    _csv,
    _gpu_state,
    _json,
    _save_audio,
    _score_audio,
)
from src.analysis.p2.common import (
    CONFIGS,
    REPEATS,
    batch_for,
    load_checkpoint,
    load_manifest,
    load_samples,
    prepared_items,
    seed_for,
)
from src.analysis.power_mel_adapter import from_vocos_weights
from src.analysis.source_prosody import compare_tracks, pitch_track
from src.analysis.unified.data import _collator
from src.analysis.unified_vc_evaluation import _seed
from src.analysis.vocoder.audio import _ours_mel_filterbank, _project_log_mel
from src.analysis.vocoder.metrics import _edit_distance, _normalize_text
from src.analysis.vocoder.models import _AsrScorer, _load_vocos, _SpeakerScorer

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


def _append(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


# ====================
# 4. Internal Functions
# ====================


def generate(output, device, smoke=False):
    """按固定验证配对及噪声生成条件替换和源强度扫描的 Mel。"""
    manifest = load_manifest(output)
    if smoke:
        manifest = manifest[:2]
    elif not (output / "smoke_conversion/SCORE_PASSED.json").is_file():
        raise RuntimeError("必须先通过转换/评分冒烟")
    directory = output / ("smoke_conversion" if smoke else "conversion")
    directory.mkdir(exist_ok=True)
    config, _, by_id = load_samples()
    items = prepared_items(manifest, config, by_id)
    collator = _collator(config)
    ledger = directory / "generation.jsonl"
    records = _rows(ledger)
    completed = {(r["experiment_id"], r["source_id"], r["repeat"]) for r in records}
    identities = {}
    configs = CONFIGS
    initial = json.loads((output / "identity.json").read_text(encoding="utf-8"))
    fingerprints = initial["checkpoint_files"]
    best_hash = next(
        v
        for k, v in fingerprints.items()
        if "2026-10-21_14-29-28" in k and k.endswith("epoch_040.ckpt")
    )
    last_hash = next(
        v
        for k, v in fingerprints.items()
        if "2026-10-21_14-29-28" in k and k.endswith("last.ckpt")
    )
    duplicate_last = best_hash == last_hash
    if duplicate_last and not smoke:
        configs = [c for c in CONFIGS if c[0] != "b3_last_s095"]
    _json(
        directory / "active_experiments.json",
        {
            "configs": configs,
            "b3_last_is_best_alias": duplicate_last,
            "skipped_duplicate_last_in_full": duplicate_last and not smoke,
        },
    )
    for experiment, label, checkpoint, variant, source_time in configs:
        model, _, identity = load_checkpoint(label, checkpoint, device)
        identities[experiment] = identity
        for index, row in enumerate(manifest):
            batch = batch_for([row], items, collator, device)
            # 统一连续布局，使各方法的同种子噪声顺序一致。
            source = batch["mel_spectrograms"].contiguous()
            mask = batch["mel_attention_mask"]
            features, lengths = (
                batch["content_features"],
                batch["content_feature_lengths"],
            )
            speaker_features = items[row["reference_id"]]["speaker_features"][None].to(device)
            applied_target = row["reference_id"]
            applied_text = row["source_text"]
            if variant == "content_donor":
                donor = collator([items[row["donor_id"]]])
                features, lengths = donor["content_features"].to(device), donor[
                    "content_feature_lengths"
                ].to(device)
                applied_text = row["donor_text"]
            elif variant == "speaker_source":
                speaker_features = batch["speaker_features"]
                applied_target = row["source_id"]
            with torch.inference_mode():
                condition = model.method.encode_condition(
                    features, lengths, speaker_features, mask
                )
                for repeat in range(REPEATS):
                    if (experiment, row["source_id"], repeat) in completed:
                        continue
                    seed = seed_for(row["index"], repeat)
                    if label == "B3":
                        with torch.random.fork_rng(devices=[device.index or 0]):
                            _seed(seed, device)
                            fixed_noise = torch.randn_like(source)
                        expected = model.method.interpolate(
                            source, fixed_noise, source.new_full((1,), source_time)
                        )
                        captured = []

                        def verify(module, args):
                            torch.testing.assert_close(args[0], expected, atol=1e-6, rtol=1e-5)
                            torch.testing.assert_close(
                                args[1]["source_time"],
                                source.new_full((1,), source_time),
                            )
                            captured.append(True)

                        handle = model.method.network.register_forward_pre_hook(verify)
                    _seed(seed, device)
                    if label == "B3":
                        generated = model.method.sample(
                            condition,
                            mask,
                            integration_steps=1,
                            source_mels=source,
                            source_diffusion_time=source_time,
                        )
                    else:
                        generated = model.method.sample(condition, mask, integration_steps=1)
                    if label == "B3":
                        handle.remove()
                        assert captured == [True]
                    assert torch.isfinite(generated).all()
                    mel = generated[0, :, : int(mask.sum())].cpu()
                    path = (
                        directory
                        / "generated_mels"
                        / experiment
                        / f"{row['source_id']}__seed{seed}.pt"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "mel": mel.half(),
                            "seed": seed,
                            "source_time": source_time,
                            "source_id": row["source_id"],
                            "reference_id": row["reference_id"],
                            "applied_target_id": applied_target,
                            "applied_content_text": applied_text,
                        },
                        path,
                    )
                    record = {
                        **row,
                        "experiment_id": experiment,
                        "repeat": repeat,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "source_time": source_time,
                        "applied_target_id": applied_target,
                        "applied_content_text": applied_text,
                        "generated_path": str(path),
                        "self_mel_l1": float((generated - source).abs().mean()),
                        "source_time_input_verified": label == "B3",
                    }
                    _append(ledger, record)
                    records.append(record)
            if (index + 1) % 16 == 0 or index + 1 == len(manifest):
                print(f"转换 {experiment} {index+1}/{len(manifest)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _json(directory / "models.json", identities)
    _csv(directory / "generation.csv", records)
    assert len(records) == len(manifest) * REPEATS * len(configs)
    _json(
        directory / "GENERATION_PASSED.json",
        {
            "records": len(records),
            "source_pairs": len(manifest),
            "repeats": REPEATS,
            "all_b3_inputs_verified": True,
        },
    )


def score(output, device, smoke=False):
    """用冻结声码器及本地识别和身份评分器评估 P2 生成结果。"""
    directory = output / ("smoke_conversion" if smoke else "conversion")
    assert (directory / "GENERATION_PASSED.json").is_file()
    manifest = load_manifest(output)
    if smoke:
        manifest = manifest[:2]
    config, _, by_id = load_samples()
    items = prepared_items(manifest, config, by_id)
    records = _rows(directory / "generation.jsonl")
    vocos = _load_vocos(
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
    ).to(device)
    adapter = from_vocos_weights(ROOT / "pretrained/vocos/pytorch_model.bin")
    asr = _AsrScorer(ROOT / "pretrained/whisper-small", device)
    assert asr.available
    asr.model.requires_grad_(False)
    speaker = _SpeakerScorer(
        ROOT / "pretrained/meanvc/ecapa_tdnn.py",
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
        ROOT / "pretrained/meanvc/wavlm_large.pt",
        device,
    )
    waves = {uid: item["waveform"] for uid, item in items.items()}
    embeddings, tracks = {}, {}
    for uid, waveform in waves.items():
        embeddings[uid] = speaker.embedding(waveform, uid)
        tracks[uid] = pitch_track(waveform)
        dest = directory / "prosody_tracks/real"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / f"{uid}.npz", **tracks[uid])
    ledger = directory / "scores.jsonl"
    results = _rows(ledger)
    completed = {(r["experiment_id"], r["source_id"], r["repeat"]) for r in results}
    _gpu_state(directory / "gpu_scoring.txt")
    for index, row in enumerate(records):
        if (row["experiment_id"], row["source_id"], row["repeat"]) in completed:
            continue
        source_id, reference_id, donor_id = (
            row["source_id"],
            row["reference_id"],
            row["donor_id"],
        )
        payload = torch.load(row["generated_path"], map_location="cpu", weights_only=True)
        with torch.inference_mode():
            waveform = vocos.decode(adapter(payload["mel"]).unsqueeze(0).to(device))[0].cpu()
        assert torch.isfinite(waveform).all() and waveform.square().mean() > 1e-12
        wav_path = (
            directory
            / "generated_wavs"
            / row["experiment_id"]
            / f"{source_id}__seed{row['seed']}.wav"
        )
        _save_audio(wav_path, waveform)
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
            "wav_path": str(wav_path),
        }
        intended = _normalize_text(row["applied_content_text"], asr._normalizer)
        normalized_hyp = result["normalized_hypothesis"]
        result.update(
            {
                "applied_content_cer": _edit_distance(
                    intended.replace(" ", ""), normalized_hyp.replace(" ", "")
                )
                / len(intended.replace(" ", "")),
                "applied_content_wer": _edit_distance(intended.split(), normalized_hyp.split())
                / len(intended.split()),
                "applied_target_similarity": (
                    result["source_speaker_similarity"]
                    if row["applied_target_id"] == source_id
                    else result["target_speaker_similarity"]
                ),
            }
        )
        track = pitch_track(waveform)
        for prefix, uid in [
            ("source", source_id),
            ("reference", reference_id),
            ("donor", donor_id),
        ]:
            result.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in compare_tracks(
                        track, tracks[uid], normalized_time=prefix != "source"
                    ).items()
                }
            )
        dest = directory / "prosody_tracks" / row["experiment_id"]
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / f"{source_id}__seed{row['seed']}.npz", **track)
        _append(ledger, result)
        results.append(result)
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f"评分 {index+1}/{len(records)}", flush=True)
    assert len(results) == len(records)
    _csv(directory / "per_sample_metrics.csv", results)
    _csv(directory / "aggregate_metrics.csv", _aggregate(results))
    _json(
        directory / "SCORE_PASSED.json",
        {
            "records": len(results),
            "source_pairs": len(manifest),
            "repeats": REPEATS,
            "all_required_finite": all(
                np.isfinite(row["cer"]) and np.isfinite(row["target_speaker_similarity"])
                for row in results
            ),
        },
    )
    print(f"评分完成：{len(results)}条", flush=True)


def controls(output, device):
    """校准当前验证文本的原音ASR和既有Mel/声码器链路损耗。"""
    directory = output / "validation_controls"
    directory.mkdir(exist_ok=False)
    manifest = load_manifest(output)
    config, _, by_id = load_samples()
    items = prepared_items(manifest, config, by_id)
    vocos = _load_vocos(
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
    ).to(device)
    adapter = from_vocos_weights(ROOT / "pretrained/vocos/pytorch_model.bin")
    asr = _AsrScorer(ROOT / "pretrained/whisper-small", device)
    assert asr.available
    asr.model.requires_grad_(False)
    speaker = _SpeakerScorer(
        ROOT / "pretrained/meanvc/ecapa_tdnn.py",
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
        ROOT / "pretrained/meanvc/wavlm_large.pt",
        device,
    )
    results = []
    for index, row in enumerate(manifest):
        uid = row["source_id"]
        source = items[uid]["waveform"]
        embedding = speaker.embedding(source, uid)
        source_track = pitch_track(source)
        for kind in ["original", "reconstruction"]:
            waveform = source
            if kind == "reconstruction":
                with torch.inference_mode():
                    waveform = vocos.decode(
                        adapter(_project_log_mel(source, _ours_mel_filterbank()))
                        .unsqueeze(0)
                        .to(device)
                    )[0].cpu()
            path = directory / "wavs" / kind / f"{uid}.wav"
            _save_audio(path, waveform)
            result = {
                **row,
                "experiment_id": kind,
                **_score_audio(waveform, row["source_text"], asr, speaker, embedding, embedding),
                "wav_path": str(path),
            }
            result.update(
                {
                    f"source_{key}": value
                    for key, value in compare_tracks(pitch_track(waveform), source_track).items()
                }
            )
            results.append(result)
            _append(directory / "controls.jsonl", result)
        if (index + 1) % 16 == 0:
            print(f"验证原音/重建 {index+1}/{len(manifest)}", flush=True)
    _csv(directory / "per_sample.csv", results)
    _csv(directory / "aggregate.csv", _aggregate(results))


# ====================
# 6. Public API
# ====================


def run(output, stage, device, smoke=False):
    """按阶段调度 P2 的生成、评分、对照或报告。"""
    torch.set_num_threads(2)
    if stage == "conversion":
        generate(output, device, smoke)
    elif stage == "score":
        score(output, device, smoke)
    elif stage == "controls":
        controls(output, device)
    elif stage == "report":
        from src.analysis.p2.reporting import run as report

        report(output)
