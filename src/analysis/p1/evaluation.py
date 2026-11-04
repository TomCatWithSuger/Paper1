"""模块说明：

- 绑定最新三组最佳权重，执行离线 P1 冒烟和完整转换评估。
- 保存固定配对、配置、指纹、波形、逐样本指标及聚合结果。
"""

# ====================
# 1. Imports
# ====================

import json
import math
import shutil
import subprocess  # nosec B404 - 仅用固定参数数组调用已定位的本地工具。
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.version import cuda as torch_cuda_version

from src.analysis.p1.metrics import _aggregate as _aggregate
from src.analysis.p1.metrics import _score_audio as _score_audio
from src.analysis.power_mel_adapter import from_vocos_weights
from src.analysis.source_prosody import compare_tracks, pitch_track
from src.analysis.unified.checkpoints import (
    _best_checkpoint,
    _load_model,
    _model_config,
)
from src.analysis.unified.data import (
    _collator,
    _dataset,
    _load_manifest_samples,
    _prepare_pair,
    _validate_feature_cache,
)
from src.analysis.unified.identity import _sha256_file
from src.analysis.unified.records import _load_ledger, _read_csv
from src.analysis.unified.records import _write_table as _csv
from src.analysis.unified_vc_evaluation import (
    _generate,
    _seed,
    _to_device,
    evaluate_unified,
)
from src.analysis.vocoder.audio import (
    _adapt_log_mel,
    _build_mel_adapter,
    _ours_mel_filterbank,
    _project_log_mel,
)
from src.analysis.vocoder.metrics import _edit_distance, _normalize_text
from src.analysis.vocoder.models import _AsrScorer, _load_vocos, _SpeakerScorer

# ====================
# 2. Definitions
# ====================

ROOT = Path(__file__).resolve().parents[3]
VCTK = Path("/mnt/data2/home/tsx/project/data/VCTK")
CACHE = ROOT / "data/meanvc_features/vctk"
RUNS = {
    "B1": "2026-10-18_23-25-52",
    "B2": "2026-10-19_13-54-44",
    "B3": "2026-10-21_14-29-28",
}
BEST = {"B1": "epoch_382.ckpt", "B2": "epoch_382.ckpt", "B3": "epoch_040.ckpt"}
SR = 22050
SEED = 12345


# ====================
# 3. Utilities
# ====================


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_state(path):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise FileNotFoundError("未找到本地 nvidia-smi")
    result = subprocess.run(  # nosec B603 - 本地可执行文件与固定参数，不经过命令解释器。
        [executable], capture_output=True, text=True, check=True
    )
    path.write_text(result.stdout, encoding="utf-8")


def _save_audio(path, waveform):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 浮点 WAV 保留全部幅度，避免 PCM 截幅改变被评估的输出。
    sf.write(path, waveform.detach().cpu().numpy(), SR, subtype="FLOAT")


# ====================
# 4. Internal Functions
# ====================


def preflight(output: Path, device: torch.device):
    """核验完整缓存、配置、严格加载与 B3 实际前向路径。"""
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "test_manifest.csv"
    shutil.copy2(ROOT / "analysis_results/baseline_validation/test_manifest.csv", manifest_path)
    rows, by_id = _load_manifest_samples(manifest_path, VCTK, "mic1")
    assert len(rows) == 100 and len({r["source_id"] for r in rows}) == 100
    _validate_feature_cache(rows, CACHE)
    plans, audit = [], {
        "models": {},
        "cache_files": {},
        "resource_files": {},
        "source_files": {},
    }
    first_data = None
    for label, run in RUNS.items():
        run_dir = ROOT / "logs/train/runs" / run
        assert _best_checkpoint(run_dir).name == BEST[label]
        model, config = _load_model(run_dir, device)
        if first_data is None:
            first_data = config["data"]
        else:
            assert first_data == config["data"], "三组数据配置不一致"
        shutil.copy2(run_dir / ".hydra/config.yaml", output / f"{label}_training_config.yaml")
        item = {
            "checkpoint": str(_best_checkpoint(run_dir)),
            "sha256": _sha256_file(_best_checkpoint(run_dir)),
            "strict_load": True,
            "method": type(model.method).__name__,
        }
        batch, _, _ = _prepare_pair(rows[0], by_id, config, CACHE, _collator(config))
        batch = _to_device(batch, device)
        calls = []

        def capture(module, args):
            calls.append(
                (
                    args[0].detach().clone(),
                    {k: v.detach().clone() for k, v in args[1].items()},
                )
            )

        handle = model.method.network.register_forward_pre_hook(capture)
        _seed(SEED, device)
        with torch.inference_mode():
            generated = _generate(model, batch, 1)
        handle.remove()
        assert torch.isfinite(generated).all() and len(calls) == 1
        if label == "B3":
            assert model.method.requires_source_mels
            assert model.method.inference_source_diffusion_time == 0.95
            states, times = calls[0]
            assert torch.all(times["source_time"] == 0.95)
            _seed(SEED, device)
            noise = torch.randn_like(batch["mel_spectrograms"])
            expected = model.method.interpolate(
                batch["mel_spectrograms"], noise, torch.full((1,), 0.95, device=device)
            )
            torch.testing.assert_close(states, expected)
            item["source_time"] = 0.95
            item["diffused_real_source_verified"] = True
        _seed(SEED, device)
        with torch.inference_mode():
            repeated = _generate(model, batch, 1)
        torch.testing.assert_close(generated, repeated, atol=0, rtol=0)
        item["same_seed_repeat_identical"] = True
        audit["models"][label] = item
        for nfe in ([1, 5, 10, 50] if label == "B1" else [1]):
            plans.append(
                {
                    "experiment_id": f"nfe_{label.lower()}_{nfe:02d}",
                    "model": label,
                    "nfe": nfe,
                    "ablation": "最新完整训练最佳权重",
                    "run_dir": str(run_dir),
                    "status": "待运行统一转换评估",
                }
            )
        calls.clear()
        del model, generated, repeated, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for uid in sorted({r[k] for r in rows for k in ["source_id", "reference_id"]}):
        for kind, shape in [("content", 2), ("speaker", 1)]:
            path = CACHE / kind / f"{uid}.pt"
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            assert tensor.ndim == shape and tensor.shape[0] == 256 and torch.isfinite(tensor).all()
            audit["cache_files"][str(path)] = _sha256_file(path)
    paths = [
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
        ROOT / "pretrained/meanvc/ecapa_tdnn.py",
        ROOT / "pretrained/meanvc/wavlm_large.pt",
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
    ]
    paths += [
        p
        for p in sorted((ROOT / "pretrained/whisper-small").iterdir())
        if p.suffix in {".json", ".txt", ".safetensors"}
    ]
    for path in paths:
        assert path.is_file()
        audit["resource_files"][str(path)] = _sha256_file(path)
    for path in sorted((ROOT / "src").rglob("*.py")):
        audit["source_files"][str(path.relative_to(ROOT))] = _sha256_file(path)
    # decode 所需权重必须全部存在；仅允许多余的声学特征提取器权重。
    vocos = _load_vocos(
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
    )
    state = torch.load(
        ROOT / "pretrained/vocos/pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    missing = sorted(set(vocos.state_dict()) - set(state))
    extra = sorted(set(state) - set(vocos.state_dict()))
    assert not missing and all(k.startswith("feature_extractor.") for k in extra)
    audit["vocos"] = {
        "missing_keys": missing,
        "ignored_keys": extra,
        "all_parameters_frozen": all(not p.requires_grad for p in vocos.parameters()),
    }
    # 实际数据与项目 Mel 独立入口必须完全一致。
    cfg = _model_config(ROOT / "logs/train/runs" / RUNS["B1"])
    ds = _dataset((by_id[rows[0]["source_id"]], by_id[rows[0]["reference_id"]]), cfg, CACHE)
    waveform = ds[0]["waveform"]
    torch.testing.assert_close(
        _collator(cfg)._log_mel(waveform),
        _project_log_mel(waveform, _ours_mel_filterbank()),
    )
    _csv(output / "experiment_plan.csv", plans)
    audit["protocol"] = {
        "seed": "12345 + manifest.index",
        "source_time": 0.95,
        "pairs": 100,
        "generation_draws_per_pair": 1,
        "sample_rate": SR,
        "waveform_resampling": "训练数据集线性重采样；前10秒，与生成输入一致",
        "timing": "一次预热后每个样本计时一次；缓存条件到Mel，不含前端；GPU共享负载",
        "mel_storage": "统一评估入口保存float16 Mel，声码器输入转float32",
        "prosody": "NCCF 16kHz/40ms窗/10ms帧移/60-500Hz/置信度0.6/RMS>=0.001；源按真实时间，参考按归一化时间；非DTW，诊断指标",
    }
    _json(
        output / "preflight.json",
        {**audit, "manifest_sha256": _sha256_file(manifest_path)},
    )
    _gpu_state(output / "gpu_preflight.txt")
    print(
        "预检通过：三组 strict 加载、固定种子、B3 真实源路径、200个缓存与声码器权重匹配。",
        flush=True,
    )


def evaluate(output: Path, stage: str, device: torch.device):
    """先生成 Mel，再用统一冻结模型评分；各阶段写入不同目录。"""
    assert stage in {"smoke", "full"}
    if stage == "full":
        assert (output / "smoke_corrected/PASSED.json").is_file(), "必须先通过修正链路冒烟"
    stage_dir = output / ("smoke_corrected" if stage == "smoke" else stage)
    stage_dir.mkdir(exist_ok=True)
    snapshot_dir = stage_dir / "code_snapshot"
    current_sources = {}
    for path in [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "scripts/analysis/evaluate_p1_latest.py",
    ]:
        relative = path.relative_to(ROOT)
        current_sources[str(relative)] = _sha256_file(path)
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    _json(
        stage_dir / "runtime_config.json",
        {
            "stage": stage,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch_cuda_version,
            "base_seed": SEED,
            "source_sha256": current_sources,
            "offline": True,
            "mel_adapter": "带内平均功率除源滤波器质量->幅度谱包络插值->权重内Slaney滤波器积分；近似非可逆",
            "voicing_threshold": 0.6,
            "f0_range_hz": [60, 500],
        },
    )
    limit = 2 if stage == "smoke" else None
    manifest_path = output / "test_manifest.csv"
    _gpu_state(stage_dir / "gpu_before.txt")
    evaluate_unified(
        output / "experiment_plan.csv",
        manifest_path,
        VCTK,
        CACHE,
        stage_dir / "mel",
        str(device),
        SEED,
        limit,
    )
    records = _load_ledger(stage_dir / "mel/evaluation_records.jsonl")
    manifest, by_id = _load_manifest_samples(manifest_path, VCTK, "mic1")
    if limit:
        manifest = manifest[:limit]
    config = _model_config(ROOT / "logs/train/runs" / RUNS["B1"])
    unique = sorted({r[k] for r in manifest for k in ["source_id", "reference_id"]})
    ds = _dataset([by_id[uid] for uid in unique], config, CACHE)
    waves = {uid: ds[i]["waveform"] for i, uid in enumerate(unique)}
    asr = _AsrScorer(ROOT / "pretrained/whisper-small", device)
    assert asr.available, "本地ASR未成功加载，不能跳过CER/WER"
    asr.model.requires_grad_(False)
    speaker = _SpeakerScorer(
        ROOT / "pretrained/meanvc/ecapa_tdnn.py",
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
        ROOT / "pretrained/meanvc/wavlm_large.pt",
        device,
    )
    speaker_state = torch.load(
        ROOT / "pretrained/meanvc/wavlm_large_finetune.pth",
        map_location="cpu",
        weights_only=True,
    )["model"]
    missing = sorted(set(speaker.encoder.model.state_dict()) - set(speaker_state))
    extra = sorted(set(speaker_state) - set(speaker.encoder.model.state_dict()))
    # 原始训练分类头不参与说话人向量提取；除此之外不允许任何不匹配。
    assert not missing and set(extra) <= {"loss_calculator.projection.weight"}
    _json(
        stage_dir / "speaker_load_audit.json",
        {"missing_keys": missing, "ignored_training_classifier_keys": extra},
    )
    for key, tensor in speaker.encoder.model.state_dict().items():
        assert tensor.shape == speaker_state[key].shape
    del speaker_state
    vocos = _load_vocos(
        ROOT / "pretrained/vocos/config.yaml",
        ROOT / "pretrained/vocos/pytorch_model.bin",
    ).to(device)
    adapter = from_vocos_weights(ROOT / "pretrained/vocos/pytorch_model.bin")
    with torch.inference_mode():
        for _ in range(3):
            vocos.decode(torch.zeros(1, 80, 64, device=device))
    embeddings, tracks = {}, {}
    for uid, waveform in waves.items():
        embeddings[uid] = speaker.embedding(waveform, uid)
        tracks[uid] = pitch_track(waveform)
        track_dir = stage_dir / "prosody_tracks/real"
        track_dir.mkdir(parents=True, exist_ok=True)
        np.savez(track_dir / f"{uid}.npz", **tracks[uid])
    manifests = {r["source_id"]: r for r in manifest}
    ledger = stage_dir / "waveform_records.jsonl"
    results = _load_ledger(ledger)
    completed = {(r["experiment_id"], r["source_id"]) for r in results}
    for i, record in enumerate(records, 1):
        if (record["experiment_id"], record["source_id"]) in completed:
            continue
        payload = torch.load(record["generated_path"], map_location="cpu", weights_only=True)
        mel = payload["mel"].float()
        adapted = adapter(mel).unsqueeze(0).to(device)
        _sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = vocos.decode(adapted)[0]
        _sync(device)
        vocoder_ms = (time.perf_counter() - started) * 1000
        generated = generated.detach().cpu()
        assert torch.isfinite(generated).all() and generated.square().mean() > 1e-12
        path = (
            stage_dir
            / "generated_wavs"
            / record["experiment_id"]
            / f"{record['source_id']}__{record['reference_id']}.wav"
        )
        _save_audio(path, generated)
        source_id, reference_id = record["source_id"], record["reference_id"]
        result = dict(record)
        result.update(
            _score_audio(
                generated,
                manifests[source_id]["text"],
                asr,
                speaker,
                embeddings[source_id],
                embeddings[reference_id],
            )
        )
        track = pitch_track(generated)
        track_dir = stage_dir / "prosody_tracks" / record["experiment_id"]
        track_dir.mkdir(parents=True, exist_ok=True)
        np.savez(track_dir / f"{source_id}__{reference_id}.npz", **track)
        for prefix, uid in [("source", source_id), ("reference", reference_id)]:
            result.update(
                {
                    f"{prefix}_{k}": v
                    for k, v in compare_tracks(
                        track, tracks[uid], normalized_time=prefix == "reference"
                    ).items()
                }
            )
        result.update(
            {
                "wav_path": str(path),
                "vocoder_latency_ms": vocoder_ms,
                "cached_to_wave_latency_ms": record["latency_ms"] + vocoder_ms,
                "cached_to_wave_rtf": (record["latency_ms"] + vocoder_ms)
                / 1000
                / record["audio_seconds"],
                "output_duration_seconds": len(generated) / SR,
            }
        )
        with ledger.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        if i % 10 == 0 or i == len(records):
            print(f"{stage} 波形评分 {i}/{len(records)}", flush=True)
    controls = []
    # 全部独立真实语句各做一次原音与同Mel声码器重建对照。
    for i, uid in enumerate(unique, 1):
        waveform = waves[uid]
        text = by_id[uid].text
        for kind in ["original", "reconstruction"]:
            if kind == "original":
                signal = waveform
            else:
                adapted = (
                    adapter(_project_log_mel(waveform, _ours_mel_filterbank()))
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.inference_mode():
                    signal = vocos.decode(adapted)[0].detach().cpu()
            path = stage_dir / "control_wavs" / kind / f"{uid}.wav"
            _save_audio(path, signal)
            score = _score_audio(signal, text, asr, speaker, embeddings[uid], embeddings[uid])
            score.update({"experiment_id": kind, "source_id": uid, "wav_path": str(path)})
            score.update(
                {
                    f"self_{k}": v
                    for k, v in compare_tracks(pitch_track(signal), tracks[uid]).items()
                }
            )
            controls.append(score)
        if i % 10 == 0 or i == len(unique):
            print(f"{stage} 原音/重建对照 {i}/{len(unique)}", flush=True)
    assert len(results) == len(manifest) * 6
    for row in results:
        for metric in [
            "cer",
            "wer",
            "target_speaker_similarity",
            "mel_l1",
            "latency_ms",
        ]:
            assert math.isfinite(row[metric])
    _csv(stage_dir / "per_sample_metrics.csv", results)
    _csv(stage_dir / "aggregate_metrics.csv", _aggregate(results))
    _csv(stage_dir / "control_per_sample.csv", controls)
    _csv(stage_dir / "control_aggregate.csv", _aggregate(controls))
    _gpu_state(stage_dir / "gpu_after.txt")
    _json(
        stage_dir / "PASSED.json",
        {
            "samples": len(manifest),
            "generated": len(results),
            "controls": len(controls),
            "all_required_metrics_finite": True,
            "asr_available": True,
            "speaker_weights_complete": True,
            "device": str(device),
        },
    )
    print(f"{stage} 完成：{len(results)} 条生成、{len(controls)} 条对照。", flush=True)


# ====================
# 6. Public API
# ====================


def run(output: Path, stage: str, device="cuda:0"):
    """调用指定阶段；新目录预检成功后才能执行生成。"""
    output = output.expanduser().resolve()
    target = torch.device(device)
    torch.set_num_threads(2)
    if stage == "preflight":
        preflight(output, target)
    else:
        assert (output / "preflight.json").is_file()
        evaluate(output, stage, target)
