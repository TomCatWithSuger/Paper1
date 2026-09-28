"""分析模块与命令行入口的回归测试。"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import compare_experiments, plot_training_metrics, visualize_mel
from src.analysis import metrics_io
from src.analysis.experiment_comparison import compare
from src.analysis.mel import _prepare_mel, visualize
from src.analysis.metrics_io import available_metrics, metric_points, summarize_metric
from src.analysis.training_metrics import DEFAULT_METRICS, analyze

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def metrics_csv(tmp_path: Path) -> Path:
    path = tmp_path / "run" / "metrics.csv"
    path.parent.mkdir()
    path.write_text(
        "epoch,step,train/loss,val/loss,val/accuracy\n"
        "2,20,4,3,0.7\n"
        "0,0,8,6,0.4\n"
        "1,10,5,2,0.9\n"
        "1,11,4,1,0.8\n"
        "2,21,,NaN,\n",
        encoding="utf-8",
    )
    return path


def test_metric_points_deduplicate_and_summarize(metrics_csv: Path) -> None:
    rows = metrics_io.read_metrics(metrics_csv)
    points = metric_points(rows, "val/loss")

    assert available_metrics(rows) == ["train/loss", "val/loss", "val/accuracy"]
    assert points == [(0.0, 6.0), (1.0, 1.0), (2.0, 3.0)]
    assert summarize_metric(points) == {
        "count": 3,
        "minimum": 1.0,
        "maximum": 6.0,
        "final": 3.0,
        "best": 1.0,
        "best_position": 1.0,
    }
    assert summarize_metric(points, minimize=False)["best"] == 6.0
    assert summarize_metric(points, minimize=False)["best_position"] == 0.0


def test_metric_points_position_fallback() -> None:
    rows = [
        {"epoch": "", "step": "7", "loss": "4"},
        {"epoch": "NaN", "step": "7", "loss": "2"},
        {"epoch": "", "step": "", "loss": "3"},
        {"epoch": "", "step": "8", "loss": "inf"},
    ]
    assert metric_points(rows, "loss") == [(2.0, 3.0), (7.0, 2.0)]


@pytest.mark.parametrize("explicit_output", [False, True])
def test_analyze_outputs(metrics_csv: Path, tmp_path: Path, explicit_output: bool) -> None:
    output_dir = tmp_path / "training" if explicit_output else None
    chart_path, summary_path = analyze(
        input_path=metrics_csv.parent,
        metrics=[*DEFAULT_METRICS, "val/accuracy"],
        output_dir=output_dir,
        title="训练 <指标>",
    )
    expected_output = output_dir or metrics_csv.parent / "analysis"

    assert chart_path == expected_output / "training_curves.svg"
    assert summary_path == expected_output / "metrics_summary.json"
    assert "训练 &lt;指标&gt;" in chart_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source"] == str(metrics_csv)
    assert set(summary["metrics"]) == {"train/loss", "val/loss", "val/accuracy"}
    assert summary["metrics"]["val/loss"]["best"] == 1.0
    assert summary["metrics"]["val/loss"]["final"] == 3.0
    assert summary["metrics"]["val/accuracy"]["best"] == 0.8
    assert summary["metrics"]["val/accuracy"]["final"] == 0.7


@pytest.mark.parametrize(
    ("metric", "best", "final", "best_position"),
    [("val/loss", 1.0, 3.0, 1.0), ("val/accuracy", 0.8, 0.7, 1.0)],
)
def test_compare_outputs(
    metrics_csv: Path,
    tmp_path: Path,
    metric: str,
    best: float,
    final: float,
    best_position: float,
) -> None:
    other_csv = tmp_path / "other.csv"
    other_csv.write_text(metrics_csv.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "comparison"
    table_path, chart_path = compare(
        runs=[("B1", metrics_csv.parent), ("B2", other_csv)],
        metric=metric,
        output_dir=output_dir,
    )

    assert table_path == output_dir / "experiment_comparison.csv"
    assert chart_path == output_dir / "experiment_comparison.svg"
    assert "<svg" in chart_path.read_text(encoding="utf-8")
    with table_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    assert [record["experiment"] for record in records] == ["B1", "B2"]
    assert [record["source"] for record in records] == [str(metrics_csv), str(other_csv)]
    for record in records:
        assert record["metric"] == metric
        assert int(record["count"]) == 3
        assert float(record["best"]) == best
        assert float(record["final"]) == final
        assert float(record["best_position"]) == best_position


@pytest.mark.parametrize("explicit_output", [False, True])
def test_visualize_npy_uses_sampled_statistics(tmp_path: Path, explicit_output: bool) -> None:
    array = np.array([[0, 999, 8, -999, 4], [10, 999, 18, -999, 14]], dtype=np.float32)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, array)
    output_dir = tmp_path / "mel_output" if explicit_output else None
    chart_path, summary_path = visualize(
        input_path=input_path,
        key=None,
        batch_index=0,
        transpose=False,
        max_frames=3,
        output_dir=output_dir,
        title=None,
    )
    expected_output = output_dir or tmp_path / "analysis"
    sampled = array[:, [0, 2, 4]]

    assert chart_path == expected_output / "sample_mel.svg"
    assert summary_path == expected_output / "sample_mel_stats.json"
    assert "Mel 频谱：sample.npy" in chart_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "source": str(input_path),
        "shape": [2, 3],
        "minimum": float(sampled.min()),
        "maximum": float(sampled.max()),
        "mean": float(sampled.mean()),
        "standard_deviation": float(sampled.std()),
    }


@pytest.mark.parametrize(
    ("max_frames", "indices"),
    [(1, [0]), (3, [0, 2, 4]), (5, [0, 1, 2, 3, 4]), (10, [0, 1, 2, 3, 4])],
)
def test_prepare_mel_max_frames(max_frames: int, indices: list[int]) -> None:
    array = np.arange(10, dtype=np.float64).reshape(2, 5)
    mel = _prepare_mel(array, batch_index=0, transpose=False, max_frames=max_frames)
    np.testing.assert_array_equal(mel, array[:, indices].astype(np.float32))
    assert mel.dtype == np.float32


@pytest.mark.parametrize("max_frames", [0, -1])
def test_prepare_mel_rejects_nonpositive_max_frames(max_frames: int) -> None:
    with pytest.raises(ValueError, match="max-frames 必须为正数"):
        _prepare_mel(np.zeros((2, 5)), 0, False, max_frames)


def test_prepare_mel_checks_finiteness_after_sampling() -> None:
    array = np.array([[0, np.nan, 2, np.inf, 4], [5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(_prepare_mel(array, 0, False, 3), array[:, [0, 2, 4]])
    with pytest.raises(ValueError, match="Mel 张量包含 NaN 或无穷值"):
        _prepare_mel(array, 0, False, 5)


def test_cli_entries_use_analysis_api() -> None:
    assert plot_training_metrics.analyze is analyze
    assert plot_training_metrics.DEFAULT_METRICS is DEFAULT_METRICS
    assert compare_experiments.compare is compare
    assert visualize_mel.visualize is visualize
    assert visualize_mel._prepare_mel is _prepare_mel


@pytest.mark.parametrize(
    "script",
    ["plot_training_metrics.py", "compare_experiments.py", "visualize_mel.py"],
)
def test_cli_help_from_external_cwd(tmp_path: Path, script: str) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "analysis" / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout
    assert script in result.stdout
