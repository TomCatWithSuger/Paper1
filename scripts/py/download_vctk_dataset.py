"""下载、校验并整理 VCTK 0.92 数据集。"""

import argparse
import hashlib
import shutil
import stat
import tempfile
from pathlib import Path
from zipfile import ZipFile

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VCTK_URL = "https://datashare.ed.ac.uk/bitstreams/60ba27b5-0330-4cdf-87bf-201e7ba30ce0/download"
VCTK_ARCHIVE_NAME = "VCTK-Corpus-0.92.zip"
VCTK_MD5 = "e04e2a665ac7db1d6e8ec76ba9d5a8c5"


def _md5(path: Path) -> str:
    """计算文件的 MD5 摘要用于数据完整性校验。"""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_vctk_root(path: Path) -> bool:
    """判断目录是否包含完整的 VCTK 核心结构。"""
    return (path / "txt").is_dir() and (path / "wav48_silence_trimmed").is_dir()


def _download_archive(url: str, destination: Path, expected_md5: str) -> None:
    """支持断点续传地下载并校验 VCTK 压缩包。"""
    if destination.is_file() and _md5(destination) == expected_md5:
        print(f"已存在并通过校验: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    for _ in range(2):
        resume_size = partial_path.stat().st_size if partial_path.is_file() else 0
        headers = {"Range": f"bytes={resume_size}-"} if resume_size else {}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            if response.status_code == 416 and resume_size:
                if _md5(partial_path) == expected_md5:
                    partial_path.replace(destination)
                    return
                partial_path.unlink()
                continue
            response.raise_for_status()
            append = resume_size > 0 and response.status_code == 206
            if resume_size and not append:
                resume_size = 0
            mode = "ab" if append else "wb"
            downloaded = resume_size
            next_report = ((downloaded // (512 * 1024 * 1024)) + 1) * 512 * 1024 * 1024
            with partial_path.open(mode) as file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"已下载: {downloaded / (1024**3):.1f} GiB")
                        next_report += 512 * 1024 * 1024
            break
    else:
        raise RuntimeError("VCTK 下载断点文件无法恢复")

    actual_md5 = _md5(partial_path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"VCTK 校验失败: expected {expected_md5}, got {actual_md5}")
    partial_path.replace(destination)
    print(f"下载完成: {destination}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """拒绝符号链接和路径穿越后安全解压 ZIP 文件。"""
    destination_root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"VCTK 压缩包包含符号链接: {member.filename}")
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"VCTK 压缩包包含非法路径: {member.filename}")
        archive.extractall(destination)


def _find_vctk_root(extracted_dir: Path) -> Path:
    """在解压目录中定位实际的 VCTK 数据根目录。"""
    if _is_vctk_root(extracted_dir):
        return extracted_dir
    for transcript_dir in extracted_dir.rglob("txt"):
        candidate = transcript_dir.parent
        if _is_vctk_root(candidate):
            return candidate
    raise RuntimeError("压缩包中未找到 VCTK 0.92 目录结构")


def download_vctk_dataset(
    destination: Path,
    archive_path: Path,
    keep_archive: bool = False,
) -> None:
    """下载 VCTK，并将实际数据根目录整理到指定位置。"""
    destination = destination.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    if _is_vctk_root(destination):
        print(f"VCTK 已存在: {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"目标目录非空且不是有效 VCTK 数据集: {destination}")

    _download_archive(VCTK_URL, archive_path, VCTK_MD5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vctk-extract-", dir=destination.parent) as temp_dir:
        extracted_dir = Path(temp_dir)
        print("正在解压 VCTK，可能需要数分钟……")
        _safe_extract(archive_path, extracted_dir)
        dataset_root = _find_vctk_root(extracted_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for child in dataset_root.iterdir():
            shutil.move(str(child), destination / child.name)

    if not _is_vctk_root(destination):
        raise RuntimeError(f"VCTK 安装后目录校验失败: {destination}")
    if not keep_archive:
        archive_path.unlink()
    print(f"VCTK 安装完成: {destination}")


def main() -> None:
    """解析下载和安装目录。"""
    default_destination = PROJECT_ROOT / "data" / "VCTK"
    default_archive = PROJECT_ROOT / "data" / "downloads" / VCTK_ARCHIVE_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=default_destination)
    parser.add_argument("--archive", type=Path, default=default_archive)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    download_vctk_dataset(args.destination, args.archive, args.keep_archive)


if __name__ == "__main__":
    main()
