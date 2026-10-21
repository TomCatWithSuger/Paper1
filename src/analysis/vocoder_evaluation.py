"""模块说明：

- 将统一评估生成的 Mel 经 Vocos 声码器合成为波形
- 计算 WER/CER（Whisper）、说话人相似度（WavLM+ECAPA）、F0、能量与 RTF 等波形域指标
- 对外提供 evaluate_vocoder_metrics
"""

# ====================
# 1. Imports
# ====================


import math
import time
from pathlib import Path
from typing import Any

import torch

from src.analysis.vocoder.audio import _adapt_log_mel as _adapt_log_mel
from src.analysis.vocoder.audio import _build_mel_adapter as _build_mel_adapter
from src.analysis.vocoder.audio import _load_audio as _load_audio
from src.analysis.vocoder.audio import _ours_mel_centers as _ours_mel_centers
from src.analysis.vocoder.audio import _ours_mel_filterbank as _ours_mel_filterbank
from src.analysis.vocoder.audio import _project_log_mel as _project_log_mel
from src.analysis.vocoder.audio import _save_waveform as _save_waveform
from src.analysis.vocoder.audio import _slaney_mel_centers as _slaney_mel_centers
from src.analysis.vocoder.definitions import AGGREGATE_FIELDS as AGGREGATE_FIELDS
from src.analysis.vocoder.definitions import ASR_SAMPLE_RATE as ASR_SAMPLE_RATE
from src.analysis.vocoder.definitions import CONTOUR_POINTS as CONTOUR_POINTS
from src.analysis.vocoder.definitions import (
    CONTROL_EXPERIMENT_ID as CONTROL_EXPERIMENT_ID,
)
from src.analysis.vocoder.definitions import CONTROL_FIELDS as CONTROL_FIELDS
from src.analysis.vocoder.definitions import ENERGY_HOP as ENERGY_HOP
from src.analysis.vocoder.definitions import ENERGY_WIN as ENERGY_WIN
from src.analysis.vocoder.definitions import F0_FRAME_TIME as F0_FRAME_TIME
from src.analysis.vocoder.definitions import F0_FREQ_HIGH as F0_FREQ_HIGH
from src.analysis.vocoder.definitions import F0_FREQ_LOW as F0_FREQ_LOW
from src.analysis.vocoder.definitions import F_MAX as F_MAX
from src.analysis.vocoder.definitions import F_MIN as F_MIN
from src.analysis.vocoder.definitions import HOP_LENGTH as HOP_LENGTH
from src.analysis.vocoder.definitions import LOG_MEL_FLOOR as LOG_MEL_FLOOR
from src.analysis.vocoder.definitions import N_FFT as N_FFT
from src.analysis.vocoder.definitions import N_MELS as N_MELS
from src.analysis.vocoder.definitions import PER_SAMPLE_FIELDS as PER_SAMPLE_FIELDS
from src.analysis.vocoder.definitions import SAMPLE_RATE as SAMPLE_RATE
from src.analysis.vocoder.definitions import VOCOS_LOG_FLOOR as VOCOS_LOG_FLOOR
from src.analysis.vocoder.definitions import WIN_LENGTH as WIN_LENGTH
from src.analysis.vocoder.metrics import _char_error_rate as _char_error_rate
from src.analysis.vocoder.metrics import _contour_metrics as _contour_metrics
from src.analysis.vocoder.metrics import _edit_distance as _edit_distance
from src.analysis.vocoder.metrics import _f0_contour as _f0_contour
from src.analysis.vocoder.metrics import _frame_log_energy as _frame_log_energy
from src.analysis.vocoder.metrics import _normalize_text as _normalize_text
from src.analysis.vocoder.metrics import _pearson as _pearson
from src.analysis.vocoder.metrics import _resample_contour as _resample_contour
from src.analysis.vocoder.metrics import _word_error_rate as _word_error_rate
from src.analysis.vocoder.models import _AsrScorer as _AsrScorer
from src.analysis.vocoder.models import _load_vocos as _load_vocos
from src.analysis.vocoder.models import _SpeakerScorer as _SpeakerScorer
from src.analysis.vocoder.reporting import _aggregate as _aggregate
from src.analysis.vocoder.reporting import _append_ledger as _append_ledger
from src.analysis.vocoder.reporting import _load_ledger as _load_ledger
from src.analysis.vocoder.reporting import _mean_or_nan as _mean_or_nan
from src.analysis.vocoder.reporting import _read_csv as _read_csv
from src.analysis.vocoder.reporting import _report as _report
from src.analysis.vocoder.reporting import _write_csv as _write_csv
from src.data.components.vctk_splits import discover_vctk_samples

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _evaluate_generated_sample(
    *,
    record: dict[str, Any],
    vocos_model: Any,
    mel_matrix: torch.Tensor,
    asr: _AsrScorer,
    speaker: _SpeakerScorer,
    reference_waveform: torch.Tensor,
    reference_embedding: torch.Tensor,
    text: str,
    wav_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """合成单条生成 Mel 并计算全部波形指标。"""
    payload = torch.load(record["generated_path"], map_location="cpu", weights_only=True)
    generated_mel = payload["mel"] if isinstance(payload, dict) else payload
    adapted = _adapt_log_mel(generated_mel, mel_matrix).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    waveform = vocos_model.decode(adapted)[0].detach().cpu()
    if device.type == "cuda":
        torch.cuda.synchronize()
    vocoder_seconds = time.perf_counter() - started

    _save_waveform(wav_path, waveform)

    asr_scores = asr.score(waveform, text)
    spk_sim = speaker.similarity(waveform, reference_embedding)
    contour = _contour_metrics(waveform, reference_waveform)

    audio_seconds = float(record["audio_seconds"])
    mel_seconds = float(record["latency_ms"]) / 1000.0
    e2e_seconds = mel_seconds + vocoder_seconds

    return {
        "experiment_id": record["experiment_id"],
        "model": record["model"],
        "nfe": record["nfe"],
        "sample_index": record["sample_index"],
        "source_id": record["source_id"],
        "reference_id": record["reference_id"],
        "audio_seconds": audio_seconds,
        "mel_latency_ms": mel_seconds * 1000.0,
        "vocoder_latency_ms": vocoder_seconds * 1000.0,
        "e2e_latency_ms": e2e_seconds * 1000.0,
        "vocoder_rtf": vocoder_seconds / audio_seconds,
        "e2e_rtf": e2e_seconds / audio_seconds,
        "wer": asr_scores["wer"],
        "cer": asr_scores["cer"],
        "spk_sim": spk_sim,
        "f0_corr": contour["f0_corr"],
        "f0_log_mae": contour["f0_log_mae"],
        "energy_corr": contour["energy_corr"],
        "energy_log_mae": contour["energy_log_mae"],
        "asr_hyp": asr_scores["asr_hyp"],
        "wav_path": str(wav_path),
    }


def _evaluate_control_sample(
    *,
    row: dict[str, str],
    vocos_model: Any,
    mel_matrix: torch.Tensor,
    filterbank: torch.Tensor,
    asr: _AsrScorer,
    speaker: _SpeakerScorer,
    reference_waveform: torch.Tensor,
    device: torch.device,
    wav_path: Path,
) -> dict[str, Any]:
    """对真实参考 Mel 做重建对照（声码器链路上限）。"""
    # 与训练/评估同口径的项目 Mel（log 功率）
    log_mel = _project_log_mel(reference_waveform, filterbank)
    adapted = _adapt_log_mel(log_mel, mel_matrix).unsqueeze(0).to(device)

    with torch.inference_mode():
        waveform = vocos_model.decode(adapted)[0].detach().cpu()
    _save_waveform(wav_path, waveform)

    text = row["text"]
    original_scores = asr.score(reference_waveform, text)
    recon_scores = asr.score(waveform, text)

    original_embedding = speaker.embedding(reference_waveform, cache_key=row["reference_id"])
    spk_sim = speaker.similarity(waveform, original_embedding)
    contour = _contour_metrics(waveform, reference_waveform)

    return {
        "reference_id": row["reference_id"],
        "text": text,
        "wer_original": original_scores["wer"],
        "wer_recon": recon_scores["wer"],
        "cer_original": original_scores["cer"],
        "cer_recon": recon_scores["cer"],
        "spk_sim": spk_sim,
        "f0_corr": contour["f0_corr"],
        "f0_log_mae": contour["f0_log_mae"],
        "energy_corr": contour["energy_corr"],
        "energy_log_mae": contour["energy_log_mae"],
        "hyp_original": original_scores["asr_hyp"],
        "hyp_recon": recon_scores["asr_hyp"],
    }


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def evaluate_vocoder_metrics(
    *,
    unified_dir: Path,
    manifest_path: Path,
    vctk_root: Path,
    vocos_config: Path,
    vocos_weights: Path,
    asr_model_dir: Path,
    speaker_source: Path,
    speaker_checkpoint: Path,
    wavlm_checkpoint: Path,
    output_dir: Path,
    device: str = "cuda",
    limit: int | None = None,
    run_control: bool = True,
) -> dict[str, Any]:
    """执行波形域评估主流程（可断点续跑）。"""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ledger_path = output_dir / "evaluation_records.jsonl"
    per_sample_path = output_dir / "per_sample_metrics.csv"
    control_path = output_dir / "reconstruction_control.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    report_path = output_dir / "vocoder_evaluation_report.md"

    manifest = {row["source_id"]: row for row in _read_csv(manifest_path)}
    unified_records = [
        record
        for record in _load_ledger(Path(unified_dir) / "evaluation_records.jsonl")
        if record["source_id"] in manifest
    ]
    unified_records.sort(key=lambda row: (str(row["experiment_id"]), int(row["sample_index"])))
    if limit is not None:
        selected_sources = list(manifest)[:limit]
        unified_records = [
            record for record in unified_records if record["source_id"] in set(selected_sources)
        ]

    print(f"加载 Vocos 声码器: {vocos_config}")
    vocos_model = _load_vocos(Path(vocos_config), Path(vocos_weights)).to(resolved_device)
    # 预热声码器，避免首次解码开销计入延迟
    with torch.inference_mode():
        for _ in range(3):
            vocos_model.decode(torch.zeros(1, N_MELS, 64, device=resolved_device))
    mel_matrix = _build_mel_adapter()
    print(f"检查本地 Whisper ASR: {asr_model_dir}")
    asr = _AsrScorer(Path(asr_model_dir), resolved_device)
    print(f"加载 MeanVC 说话人编码器: {speaker_checkpoint}")
    speaker = _SpeakerScorer(
        Path(speaker_source), Path(speaker_checkpoint), Path(wavlm_checkpoint), resolved_device
    )
    filterbank = _ours_mel_filterbank()

    # 建立参考语句音频路径映射（按 utterance_id 复用）
    audio_paths = {
        sample.utterance_id: Path(sample.audio_path)
        for sample in discover_vctk_samples(Path(vctk_root), "mic1")
    }
    reference_audio_cache: dict[str, torch.Tensor] = {}

    def reference_waveform_for(reference_id: str) -> torch.Tensor:
        if reference_id not in reference_audio_cache:
            if reference_id not in audio_paths:
                raise FileNotFoundError(f"参考语句不存在: {reference_id}")
            reference_audio_cache[reference_id] = _load_audio(audio_paths[reference_id])
        return reference_audio_cache[reference_id]

    completed = _load_ledger(ledger_path)
    results_by_key = {
        (
            str(record["experiment_id"]),
            str(record["source_id"]),
            str(record["reference_id"]),
        ): record
        for record in completed
    }

    total = len(unified_records)
    for index, record in enumerate(unified_records, start=1):
        key = (str(record["experiment_id"]), str(record["source_id"]), str(record["reference_id"]))
        previous = results_by_key.get(key)
        if previous is not None and (
            not asr.available
            or all(
                math.isfinite(float(previous.get(metric, math.nan))) for metric in ("wer", "cer")
            )
        ):
            continue
        reference_id = str(record["reference_id"])
        row = manifest[str(record["source_id"])]
        reference_waveform = reference_waveform_for(reference_id)
        reference_embedding = speaker.embedding(reference_waveform, cache_key=reference_id)
        wav_path = (
            output_dir
            / "generated_wavs"
            / str(record["experiment_id"])
            / (f"{record['source_id']}__{reference_id}.wav")
        )
        result = _evaluate_generated_sample(
            record=record,
            vocos_model=vocos_model,
            mel_matrix=mel_matrix,
            asr=asr,
            speaker=speaker,
            reference_waveform=reference_waveform,
            reference_embedding=reference_embedding,
            text=row["text"],
            wav_path=wav_path,
            device=resolved_device,
        )
        _append_ledger(ledger_path, result)
        results_by_key[key] = result
        if index % 20 == 0 or index == total:
            print(f"生成样本进度: {index}/{total}")

    control_rows: list[dict[str, Any]] = []
    if run_control:
        control_sources = list(manifest)
        if limit is not None:
            control_sources = control_sources[:limit]
        for index, source_id in enumerate(control_sources, start=1):
            row = manifest[source_id]
            reference_id = row["reference_id"]
            reference_waveform = reference_waveform_for(reference_id)
            wav_path = output_dir / "control_wavs" / f"{reference_id}.wav"
            control_rows.append(
                _evaluate_control_sample(
                    row=row,
                    vocos_model=vocos_model,
                    mel_matrix=mel_matrix,
                    filterbank=filterbank,
                    asr=asr,
                    speaker=speaker,
                    reference_waveform=reference_waveform,
                    device=resolved_device,
                    wav_path=wav_path,
                )
            )
            if index % 20 == 0 or index == len(control_sources):
                print(f"重建对照进度: {index}/{len(control_sources)}")

    results = list(results_by_key.values())
    aggregates = _aggregate(results)
    _write_csv(per_sample_path, results, PER_SAMPLE_FIELDS)
    _write_csv(control_path, control_rows, CONTROL_FIELDS)
    _write_csv(aggregate_path, aggregates, AGGREGATE_FIELDS)
    report_path.write_text(
        _report(aggregates, control_rows, str(resolved_device), asr_available=asr.available),
        encoding="utf-8",
    )

    return {
        "per_sample": per_sample_path,
        "control": control_path,
        "aggregate": aggregate_path,
        "report": report_path,
        "aggregates": aggregates,
        "asr_available": asr.available,
    }


# ====================
# 7. Entry Point
# 直接运行提示
# ====================


if __name__ == "__main__":
    raise SystemExit("请使用 scripts/analysis/evaluate_vocoder_metrics.py 运行评估")
