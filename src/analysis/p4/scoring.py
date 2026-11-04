"""模块说明：

- 使用本地 CAMPPlus ONNX 对已有输出及真实源/参考音频独立评分。
- 复用 CosyVoice 的 16kHz、80 维 fbank 和逐维均值归一化，不加载 VC 模型。
"""

# ====================
# 1. Imports
# ====================

import json
from pathlib import Path

import numpy as np

from src.analysis.p4.identity import (
    original_path,
    read_csv,
    sha256,
    speaker_genders,
    statistics,
    validate_rows,
    write_csv,
)

# ====================
# 3. Utilities
# 通用工具函数
# ====================


def unit_embedding(value):
    """拒绝非有限或零范数向量，返回单位长度的说话人表示。"""
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("独立评分器输出无效或零范数")
    return value / norm


# ====================
# 4. Internal Functions
# 内部业务 / 功能实现
# ====================


class CampPlus:
    """只从显式本地路径创建 CPU 推理会话。"""

    def __init__(self, model):
        import onnxruntime as ort
        import torch

        torch.set_num_threads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.runtime_version = ort.__version__
        self.torch_version = torch.__version__

    def embedding(self, path):
        """以 16kHz、80维 fbank 和去均值预处理本地音频并提取单位向量。"""
        import soundfile as sf
        import torch
        import torchaudio
        from torchaudio.compliance.kaldi import fbank

        wave, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
        if (
            wave.shape[1] < rate * 0.1
            or not torch.isfinite(wave).all()
            or wave.square().mean() < 1e-12
        ):
            raise ValueError(f"音频过短、静音或非有限：{path}")
        with torch.inference_mode():
            if rate != 16000:
                wave = torchaudio.functional.resample(wave, rate, 16000)
            features = fbank(wave, num_mel_bins=80, dither=0, sample_frequency=16000)
            features -= features.mean(dim=0, keepdim=True)
            result = self.session.run(
                None, {self.session.get_inputs()[0].name: features.unsqueeze(0).numpy()}
            )[0]
        return unit_embedding(result)


# ====================
# 5. Core Logic
# 核心流程
# ====================


def _execute(output, model):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if sha256(protocol["input_csv"]) != protocol["input_sha256"]:
        raise ValueError("原始评分表已变化，拒绝混合协议")
    if sha256(protocol["speaker_info"]) != protocol["speaker_info_sha256"]:
        raise ValueError("说话人元数据已变化")
    for path, fingerprint in protocol["audio_sha256"].items():
        if sha256(path) != fingerprint:
            raise ValueError(f"输入音频指纹已变化：{path}")
    rows = validate_rows(
        read_csv(protocol["input_csv"]), speaker_genders(protocol["speaker_info"])
    )
    model = model.resolve(strict=True)
    directory = output / "campplus"
    if directory.exists():
        raise FileExistsError("独立评分输出已存在；请核验后使用新的分析目录")
    backend = CampPlus(model)
    directory.mkdir()
    model_hash = sha256(model)
    cache = {}

    def embed(path):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = backend.embedding(key)
        return cache[key]

    dataset = Path(protocol["dataset"])
    result, controls = [], []
    seen = set()
    for index, row in enumerate(rows, 1):
        source = embed(original_path(dataset, row["source_id"]))
        target = embed(original_path(dataset, row["reference_id"]))
        generated = embed(row["wav_path"])
        item = dict(row)
        item["target_speaker_similarity"] = float(generated @ target)
        item["source_speaker_similarity"] = float(generated @ source)
        item["target_preference_gap"] = (
            item["target_speaker_similarity"] - item["source_speaker_similarity"]
        )
        item["target_preferred"] = float(item["target_preference_gap"] > 0)
        result.append(item)
        if row["source_id"] not in seen:
            donor = embed(original_path(dataset, row["donor_id"]))
            same, cross = float(source @ donor), float(source @ target)
            controls.append(
                {
                    "source_id": row["source_id"],
                    "source_speaker_id": row["source_speaker_id"],
                    "reference_speaker_id": row["reference_speaker_id"],
                    "gender_pair": row["gender_pair"],
                    "same_speaker_distinct_utterance": same,
                    "different_speaker": cross,
                    "same_minus_different": same - cross,
                }
            )
            seen.add(row["source_id"])
        if index % 32 == 0:
            print(f"CAMPPlus {index}/{len(rows)}", flush=True)
    if sha256(model) != model_hash:
        raise ValueError("评分过程中模型文件发生变化")
    write_csv(directory / "per_sample.csv", result)
    write_csv(directory / "real_speaker_controls.csv", controls)
    summaries, pairs = statistics(result)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "paired.csv", pairs)
    np.savez(
        directory / "embeddings.npz",
        paths=np.asarray(list(cache)),
        embeddings=np.stack(list(cache.values())),
    )
    audit = {
        "model": str(model),
        "model_sha256": model_hash,
        "architecture": "CAMPPlus；不同于条件前端 WavLM+ECAPA；训练数据来源的独立性未确认",
        "preprocessing": "原始源/目标mic1音频及已存生成WAV均单声道重采样16k；80维Kaldi fbank、dither=0、逐维去时间均值；全句，无随机裁剪，无峰值归一化",
        "provider": "CPUExecutionProvider",
        "onnxruntime": backend.runtime_version,
        "torch": backend.torch_version,
        "n_outputs": len(result),
        "n_embeddings": len(cache),
        "n_control_pairs": len(controls),
        "same_speaker_preferred_fraction": float(
            np.mean([r["same_minus_different"] > 0 for r in controls])
        ),
        "source_sha256": {
            str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name("identity.py")]
        },
        "limitations": "CER沿用Whisper；相似度绝对值不可跨编码器比较；真实源/目标采用原始48k输入重采样而非旧22050中间链路；未执行听测",
    }
    (directory / "COMPLETE.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol["independent_scoring"] = "completed_campplus"
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def run(output, model):
    """核验输入指纹后执行本地 CAMPPlus 交叉评分及配对统计。"""
    return _execute(Path(output), Path(model))
