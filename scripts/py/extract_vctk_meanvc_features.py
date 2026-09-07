"""离线提取 VCTK 的 MeanVC 内容特征和说话人向量。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.components.vctk_splits import discover_vctk_samples  # noqa: E402
from src.models.components.meanvc_conditioning import (  # noqa: E402
    ContentEncoder,
    SpeakerEncoder,
)


def _dotenv_value(name: str) -> str | None:
    """读取环境变量，未设置时再读取项目根目录的 .env。"""
    value = os.getenv(name)
    if value:
        return value
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", maxsplit=1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return None


def _configured_path(name: str, fallback: Path) -> Path:
    """读取路径配置，并在未配置时使用备用路径。"""
    value = _dotenv_value(name)
    return Path(value).expanduser().resolve() if value else fallback.resolve()


def _load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    """读取音频并将多声道波形转换为单声道张量。"""
    result: tuple[Any, int] = sf.read(path, dtype="float32", always_2d=True)
    audio, sample_rate = result
    return torch.from_numpy(audio).mean(dim=1), int(sample_rate)


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """以 float16 和原子替换方式保存特征张量。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu().to(torch.float16).contiguous(), temporary_path)
    temporary_path.replace(path)


def _resolve_device(value: str) -> torch.device:
    """解析计算设备并检查 CUDA 是否可用。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def extract_features(args: argparse.Namespace) -> None:
    """遍历 VCTK 并以 utterance ID 保存完整语句特征。"""
    device = _resolve_device(args.device)
    samples = discover_vctk_samples(args.data_dir, args.microphone)
    if args.limit is not None:
        samples = samples[: args.limit]

    content_dir = args.output_dir / "content"
    speaker_dir = args.output_dir / "speaker"
    content_dir.mkdir(parents=True, exist_ok=True)
    speaker_dir.mkdir(parents=True, exist_ok=True)

    need_any_content = args.overwrite or any(
        not (content_dir / f"{sample.utterance_id}.pt").is_file() for sample in samples
    )
    need_any_speaker = args.overwrite or any(
        not (speaker_dir / f"{sample.utterance_id}.pt").is_file() for sample in samples
    )
    content_encoder = None
    speaker_encoder = None
    if need_any_content:
        content_encoder = ContentEncoder(checkpoint_path=str(args.content_checkpoint)).to(device)
        content_encoder.eval()
    if need_any_speaker:
        speaker_encoder = SpeakerEncoder(
            source_path=str(args.speaker_source),
            checkpoint_path=str(args.speaker_checkpoint),
            wavlm_checkpoint_path=str(args.wavlm_checkpoint),
        ).to(device)
        speaker_encoder.eval()

    completed = 0
    skipped = 0
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            content_path = content_dir / f"{sample.utterance_id}.pt"
            speaker_path = speaker_dir / f"{sample.utterance_id}.pt"
            need_content = args.overwrite or not content_path.is_file()
            need_speaker = args.overwrite or not speaker_path.is_file()
            if not need_content and not need_speaker:
                skipped += 1
                continue

            waveform, sample_rate = _load_waveform(sample.audio_path)
            waveform = waveform.to(device)
            if need_content:
                if content_encoder is None:
                    raise RuntimeError("content encoder was not initialized")
                content = content_encoder.extract_features(waveform, sample_rate)
                _save_tensor(content_path, content)
            if need_speaker:
                if speaker_encoder is None:
                    raise RuntimeError("speaker encoder was not initialized")
                speaker = speaker_encoder.extract_embedding(waveform, sample_rate)
                _save_tensor(speaker_path, speaker)
            completed += 1

            if index % 100 == 0 or index == len(samples):
                print(f"进度: {index}/{len(samples)}，新增 {completed}，跳过 {skipped}")

    print(f"特征缓存完成: {args.output_dir}，新增 {completed}，跳过 {skipped}")


def main() -> None:
    """解析参数并启动离线特征提取。"""
    pretrained_dir = _configured_path(
        "MEANVC_PRETRAINED_DIR", PROJECT_ROOT / "pretrained" / "meanvc"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_configured_path("VCTK_DATA_DIR", PROJECT_ROOT / "data" / "VCTK"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_configured_path(
            "MEANVC_FEATURE_CACHE_DIR", PROJECT_ROOT / "data" / "meanvc_features" / "vctk"
        ),
    )
    parser.add_argument(
        "--content-checkpoint",
        type=Path,
        default=_configured_path("MEANVC_CONTENT_ENCODER_CKPT", pretrained_dir / "fastu2++.pt"),
    )
    parser.add_argument(
        "--speaker-source",
        type=Path,
        default=_configured_path(
            "MEANVC_SPEAKER_ENCODER_SOURCE", pretrained_dir / "ecapa_tdnn.py"
        ),
    )
    parser.add_argument(
        "--speaker-checkpoint",
        type=Path,
        default=_configured_path(
            "MEANVC_SPEAKER_ENCODER_CKPT", pretrained_dir / "wavlm_large_finetune.pth"
        ),
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=Path,
        default=_configured_path("MEANVC_WAVLM_CKPT", pretrained_dir / "wavlm_large.pt"),
    )
    parser.add_argument("--microphone", choices=("mic1", "mic2"), default="mic1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    extract_features(args)


if __name__ == "__main__":
    main()
