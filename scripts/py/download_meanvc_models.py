"""下载并校验条件 Flow Matching 使用的 MeanVC 预训练文件。"""

import argparse
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import gdown
import requests

MEANVC_REVISION = "fe5286ae205a26ad4eba64395513130bd5974b46"
GOOGLE_DRIVE_SPEAKER_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_HASHES = {
    "fastu2++.pt": "4d6bc4290c4d489ed50b6ffbbcda33bd3ba9551506852c7f2fa683f9fe9512a1",
    "wavlm_large.pt": "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f",
    "wavlm_large_finetune.pth": "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b",
    "ecapa_tdnn.py": "c4c7363128564365b61e1c3a6c589e2e2406db370121a021dba8779c2245d28f",
}


def _dotenv_value(name: str) -> str | None:
    """优先读取环境变量，其次读取项目根目录的 .env。"""
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


def _default_destination() -> Path:
    """解析 MeanVC 预训练文件的默认保存目录。"""
    project_default = PROJECT_ROOT / "pretrained" / "meanvc"
    configured_dir = _dotenv_value("MEANVC_PRETRAINED_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()
    content_checkpoint = _dotenv_value("MEANVC_CONTENT_ENCODER_CKPT")
    if content_checkpoint:
        return Path(content_checkpoint).expanduser().resolve().parent
    return project_default


def _sha256(path: Path) -> str:
    """计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checked(
    destination: Path,
    downloader: Callable[[Path], None],
) -> None:
    """下载文件并在原子替换前校验 SHA-256。"""
    expected_hash = FILE_HASHES[destination.name]
    if destination.is_file() and _sha256(destination) == expected_hash:
        print(f"已存在: {destination}")
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    downloader(temporary_path)
    actual_hash = _sha256(temporary_path)
    if actual_hash != expected_hash:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{destination.name} 校验失败: expected {expected_hash}, got {actual_hash}"
        )
    temporary_path.replace(destination)
    print(f"下载完成: {destination}")


def _download_https_file(url: str, destination: Path) -> None:
    """以流式方式将 HTTPS 资源写入指定文件。"""
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def download_meanvc_models(destination: Path) -> None:
    """将 MeanVC 条件编码器依赖下载到指定目录。"""
    destination.mkdir(parents=True, exist_ok=True)
    _download_checked(
        destination / "fastu2++.pt",
        lambda path: _download_https_file(
            "https://huggingface.co/ASLP-lab/MeanVC/resolve/main/fastu2%2B%2B.pt",
            path,
        ),
    )
    _download_checked(
        destination / "wavlm_large.pt",
        lambda path: _download_https_file(
            "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt",
            path,
        ),
    )
    _download_checked(
        destination / "wavlm_large_finetune.pth",
        lambda path: gdown.download(
            id=GOOGLE_DRIVE_SPEAKER_ID,
            output=str(path),
            quiet=False,
        ),
    )
    source_url = (
        "https://raw.githubusercontent.com/ASLP-lab/MeanVC/"
        f"{MEANVC_REVISION}/src/runtime/speaker_verification/ecapa_tdnn.py"
    )
    _download_checked(
        destination / "ecapa_tdnn.py",
        lambda path: _download_https_file(source_url, path),
    )


def main() -> None:
    """解析下载目录并执行下载。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=_default_destination())
    args = parser.parse_args()
    download_meanvc_models(args.destination.expanduser().resolve())


if __name__ == "__main__":
    main()
