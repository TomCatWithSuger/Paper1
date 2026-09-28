"""读取 Mel 张量并输出热力图和数值统计。"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .svg_utils import write_heatmap


def _select_mapping_value(values: Mapping[str, Any], key: str | None) -> Any:
    """从映射中选择显式键或唯一张量值。"""
    if key is not None:
        if key not in values:
            raise KeyError(f"不存在键 {key!r}，可用键：{', '.join(values)}")
        return values[key]
    tensors = [value for value in values.values() if isinstance(value, (torch.Tensor, np.ndarray))]
    if len(tensors) != 1:
        raise ValueError("文件包含多个候选张量，请使用 --key 指定")
    return tensors[0]


def _load_array(path: Path, key: str | None) -> np.ndarray:
    """安全读取常用张量文件格式。"""
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, Mapping):
            value = _select_mapping_value(value, key)
        if not isinstance(value, torch.Tensor):
            raise TypeError("PyTorch 文件中未找到张量")
        return value.detach().float().cpu().numpy()
    if suffix == ".npy":
        return cast(np.ndarray, np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected_key = key or (archive.files[0] if len(archive.files) == 1 else None)
            if selected_key is None:
                raise ValueError("npz 包含多个数组，请使用 --key 指定")
            if selected_key not in archive:
                raise KeyError(f"不存在键 {selected_key!r}，可用键：{', '.join(archive.files)}")
            return np.asarray(archive[selected_key])
    raise ValueError(f"不支持的文件格式: {suffix}")


def _prepare_mel(
    array: np.ndarray,
    batch_index: int,
    transpose: bool,
    max_frames: int,
) -> np.ndarray:
    """将输入整理为 ``[mel, frames]`` 并限制显示帧数。"""
    array = np.squeeze(array)
    if array.ndim == 3:
        if not 0 <= batch_index < array.shape[0]:
            raise IndexError(f"batch-index 超出范围: {batch_index}")
        array = array[batch_index]
    if array.ndim != 2:
        raise ValueError(f"Mel 张量必须是二维或三维，当前形状为 {array.shape}")
    if transpose:
        array = array.T
    elif array.shape[0] > array.shape[1] and array.shape[1] <= 256:
        array = array.T
    if max_frames <= 0:
        raise ValueError("max-frames 必须为正数")
    if array.shape[1] > max_frames:
        indices = np.linspace(0, array.shape[1] - 1, max_frames).astype(int)
        array = array[:, indices]
    if not np.isfinite(array).all():
        raise ValueError("Mel 张量包含 NaN 或无穷值")
    return array.astype(np.float32, copy=False)


def visualize(
    input_path: Path,
    key: str | None,
    batch_index: int,
    transpose: bool,
    max_frames: int,
    output_dir: Path | None,
    title: str | None,
) -> tuple[Path, Path]:
    """生成 Mel 热力图和统计摘要。"""
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    mel = _prepare_mel(
        _load_array(input_path, key),
        batch_index=batch_index,
        transpose=transpose,
        max_frames=max_frames,
    )

    resolved_output = (output_dir or input_path.parent / "analysis").resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    chart_path = resolved_output / f"{input_path.stem}_mel.svg"
    summary_path = resolved_output / f"{input_path.stem}_mel_stats.json"
    write_heatmap(
        matrix=mel.tolist(),
        output=chart_path,
        title=title or f"Mel 频谱：{input_path.name}",
    )
    summary = {
        "source": str(input_path),
        "shape": list(mel.shape),
        "minimum": float(mel.min()),
        "maximum": float(mel.max()),
        "mean": float(mel.mean()),
        "standard_deviation": float(mel.std()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return chart_path, summary_path
