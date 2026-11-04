"""模块说明：在分析流程回归测试中模拟 Windows 的非 UTF-8 默认文件编码。"""

from pathlib import Path

import pytest


@pytest.fixture()
def legacy_text_encoding(monkeypatch):
    """未声明编码的 Path 文本操作使用 CP1252，显式编码保持原行为。"""
    read_text = Path.read_text
    write_text = Path.write_text

    def read(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    def write(path, data, encoding=None, errors=None, newline=None):
        return write_text(
            path, data, encoding=encoding or "cp1252", errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)
