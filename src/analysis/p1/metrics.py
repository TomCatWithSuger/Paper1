"""模块说明：负责 P1 音频评分和逐实验指标聚合，不加载模型或启动评估。"""

# ====================
# 1. Imports
# ====================

import math

import numpy as np

from src.analysis.vocoder.metrics import _edit_distance, _normalize_text

# ====================
# 4. Internal Functions
# ====================


def _aggregate(rows):
    aggregates = []
    for experiment in sorted({r["experiment_id"] for r in rows}):
        group = [r for r in rows if r["experiment_id"] == experiment]
        row = {"experiment_id": experiment, "sample_count": len(group)}
        for key, value in group[0].items():
            if isinstance(value, (int, float)) and key not in {
                "seed",
                "sample_index",
                "nfe",
            }:
                values = np.array([r[key] for r in group], dtype=float)
                values = values[np.isfinite(values)]
                row[f"{key}_valid_n"] = len(values)
                row[f"{key}_mean"] = float(values.mean()) if len(values) else math.nan
                if "latency_ms" in key:
                    for q in [50, 95]:
                        row[f"{key}_p{q}"] = float(np.percentile(values, q))
        row["cer_corpus"] = sum(r["char_edits"] for r in group) / sum(
            r["reference_chars"] for r in group
        )
        row["wer_corpus"] = sum(r["word_edits"] for r in group) / sum(
            r["reference_words"] for r in group
        )
        aggregates.append(row)
    return aggregates


def _score_audio(waveform, text, asr, speaker, source_embedding, reference_embedding):
    scores = asr.score(waveform, text)
    ref = _normalize_text(text, asr._normalizer)
    hyp = _normalize_text(scores["asr_hyp"], asr._normalizer)
    vector = speaker.embedding(waveform)
    return {
        **scores,
        "normalized_reference": ref,
        "normalized_hypothesis": hyp,
        "char_edits": _edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")),
        "reference_chars": len(ref.replace(" ", "")),
        "word_edits": _edit_distance(ref.split(), hyp.split()),
        "reference_words": len(ref.split()),
        "target_speaker_similarity": float((vector * reference_embedding).sum()),
        "source_speaker_similarity": float((vector * source_embedding).sum()),
        "waveform_rms": float(waveform.square().mean().sqrt()),
        "waveform_peak": float(waveform.abs().max()),
        "overrange_fraction": float((waveform.abs() > 1).float().mean()),
    }
