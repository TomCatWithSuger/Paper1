import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.py import download_meanvc_models as meanvc_download
from scripts.py import download_vctk_dataset as vctk_download
from scripts.py import extract_vctk_meanvc_features as feature_extraction

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_vctk_archive(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("VCTK-Corpus-0.92/txt/p225/p225_001.txt", "test")
        archive.writestr(
            "VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_001_mic1.flac",
            b"audio",
        )


def test_download_vctk_dataset_installs_normalized_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "VCTK"
    archive_path = tmp_path / "downloads" / "VCTK.zip"

    def fake_download(url: str, destination_path: Path, expected_md5: str) -> None:
        destination_path.parent.mkdir(parents=True)
        _write_vctk_archive(destination_path)

    monkeypatch.setattr(vctk_download, "_download_archive", fake_download)
    vctk_download.download_vctk_dataset(destination, archive_path)

    assert (destination / "txt" / "p225" / "p225_001.txt").is_file()
    assert (destination / "wav48_silence_trimmed" / "p225" / "p225_001_mic1.flac").is_file()
    assert not archive_path.exists()


def test_vctk_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid.zip"
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "invalid")

    with pytest.raises(RuntimeError, match="非法路径"):
        vctk_download._safe_extract(archive_path, extract_dir)

    assert not (tmp_path / "escaped.txt").exists()


def test_meanvc_download_uses_project_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_destination = None

    def fake_download(destination: Path) -> None:
        nonlocal captured_destination
        captured_destination = destination

    for name in ("MEANVC_PRETRAINED_DIR", "MEANVC_CONTENT_ENCODER_CKPT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(meanvc_download, "download_meanvc_models", fake_download)
    monkeypatch.setattr(sys, "argv", ["download_meanvc_models.py"])

    meanvc_download.main()

    assert captured_destination == PROJECT_ROOT / "pretrained" / "meanvc"


def test_vctk_download_uses_project_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_paths = None

    def fake_download(
        destination: Path,
        archive_path: Path,
        keep_archive: bool = False,
    ) -> None:
        nonlocal captured_paths
        captured_paths = destination, archive_path, keep_archive

    monkeypatch.setattr(vctk_download, "download_vctk_dataset", fake_download)
    monkeypatch.setattr(sys, "argv", ["download_vctk_dataset.py"])

    vctk_download.main()

    assert captured_paths == (
        PROJECT_ROOT / "data" / "VCTK",
        PROJECT_ROOT / "data" / "downloads" / vctk_download.VCTK_ARCHIVE_NAME,
        False,
    )


def test_feature_extraction_uses_vctk_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_args = None

    def fake_extract(args: object) -> None:
        nonlocal captured_args
        captured_args = args

    monkeypatch.setattr(feature_extraction, "_dotenv_value", lambda name: None)
    monkeypatch.setattr(feature_extraction, "extract_features", fake_extract)
    monkeypatch.setattr(sys, "argv", ["extract_vctk_meanvc_features.py"])

    feature_extraction.main()

    assert captured_args is not None
    assert captured_args.data_dir == PROJECT_ROOT / "data" / "VCTK"
