"""定义波形评估参数与输出字段。"""

# ====================
# 1. Imports
# ====================


# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================


SAMPLE_RATE = 22050


N_FFT = 1024


WIN_LENGTH = 1024


HOP_LENGTH = 256


N_MELS = 80


F_MIN = 0.0


F_MAX = 8000.0


LOG_MEL_FLOOR = 1e-5


VOCOS_LOG_FLOOR = 1e-7


ASR_SAMPLE_RATE = 16000


CONTOUR_POINTS = 200


F0_FRAME_TIME = 0.01


F0_FREQ_LOW = 60


F0_FREQ_HIGH = 500


ENERGY_WIN = 1024


ENERGY_HOP = 256


CONTROL_EXPERIMENT_ID = "control_recon"


PER_SAMPLE_FIELDS = (
    "experiment_id",
    "model",
    "nfe",
    "sample_index",
    "source_id",
    "reference_id",
    "audio_seconds",
    "mel_latency_ms",
    "vocoder_latency_ms",
    "e2e_latency_ms",
    "vocoder_rtf",
    "e2e_rtf",
    "wer",
    "cer",
    "spk_sim",
    "f0_corr",
    "f0_log_mae",
    "energy_corr",
    "energy_log_mae",
    "asr_hyp",
    "wav_path",
)


CONTROL_FIELDS = (
    "reference_id",
    "text",
    "wer_original",
    "wer_recon",
    "cer_original",
    "cer_recon",
    "spk_sim",
    "f0_corr",
    "f0_log_mae",
    "energy_corr",
    "energy_log_mae",
    "hyp_original",
    "hyp_recon",
)


AGGREGATE_FIELDS = (
    "experiment_id",
    "model",
    "nfe",
    "sample_count",
    "wer_mean",
    "cer_mean",
    "spk_sim_mean",
    "f0_corr_mean",
    "f0_log_mae_mean",
    "energy_corr_mean",
    "energy_log_mae_mean",
    "vocoder_latency_ms_mean",
    "vocoder_rtf_mean",
    "e2e_latency_ms_mean",
    "e2e_rtf_mean",
)
