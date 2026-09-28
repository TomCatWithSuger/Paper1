"""分析模块与命令行入口的回归测试。

使用临时 CSV 和人工构造的二维矩阵验证指标统计、文件输出及 Mel 抽样规则， 不依赖真实训练日志、音频或模型权重，也不会启动训练或下载资源。
图表检查针对输出路径与部分文本内容，不等同于完整的视觉效果验证。 命令行帮助通过当前进程中的 runpy 执行，不验证独立进程的导入隔离。
"""

import csv
import json
import runpy
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
    """提供包含重复位置及无效值的指标记录。

    epoch 故意乱序排列，且 epoch=1 出现两次，用于区分排序与覆盖行为。 最后一行含空值和 NaN，用于确认无效值不会覆盖已有的有效指标。 accuracy
    的数值设计同时覆盖最优值、最终值及最大化方向。
    """
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
    """验证去重后的最优值与最终值统计。

    同一 epoch 保留最后一个有效数值，最终序列按 epoch 升序排列。 最终值取最大训练位置上的记录，而非 CSV 中物理位置最后一行。
    同一序列分别按最小化和最大化统计，检查最优值及对应位置。
    """
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
    """验证缺少 epoch 时的横坐标回退规则。

    空值或 NaN 的 epoch 回退到 step；两者均无效时使用零起始记录序号。 重复的 step 同样遵循最后有效值覆盖规则，指标中的无穷值不参与统计。
    """
    rows = [
        {"epoch": "", "step": "7", "loss": "4"},
        {"epoch": "NaN", "step": "7", "loss": "2"},
        {"epoch": "", "step": "", "loss": "3"},
        {"epoch": "", "step": "8", "loss": "inf"},
    ]
    assert metric_points(rows, "loss") == [(2.0, 3.0), (7.0, 2.0)]


@pytest.mark.parametrize("explicit_output", [False, True])
def test_analyze_outputs(metrics_csv: Path, tmp_path: Path, explicit_output: bool) -> None:
    """验证训练曲线及摘要的内容和输出位置。

    分别覆盖显式输出目录和指标文件旁的默认 analysis 目录。 输入使用运行目录，验证模块能够定位其中唯一的 metrics.csv。 请求的 test/loss
    不存在时应跳过，标题中的尖括号应正确转义。
    """
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
    """验证不同指标的实验对比表和图表。

    两个实验使用相同数据，分别以运行目录和 CSV 路径作为输入。 loss 取最小值，accuracy 取最大值；CSV 中的实验顺序应与输入一致。 本测试检查 SVG
    已生成，但不比较柱形几何位置或实际渲染效果。
    """
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
    """验证 Mel 摘要使用抽样后的数据。

    人工矩阵并非真实声学特征，仅用于验证张量读取和数值处理。 将五帧压缩为三帧时，索引 1 和 3 的极端值被跳过， 因此可明确区分完整数据统计与当前实现的绘图矩阵统计。
    同时覆盖默认输出目录、显式目录和基于输入文件名生成的标题。
    """
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
    """验证显示帧数限制的均匀抽样规则。

    覆盖只保留一帧、减少帧数、等于原帧数和超过原帧数四种情况。 限制大于原帧数时不补帧，返回矩阵统一转换为 float32。
    """
    array = np.arange(10, dtype=np.float64).reshape(2, 5)
    mel = _prepare_mel(array, batch_index=0, transpose=False, max_frames=max_frames)
    np.testing.assert_array_equal(mel, array[:, indices].astype(np.float32))
    assert mel.dtype == np.float32


@pytest.mark.parametrize("max_frames", [0, -1])
def test_prepare_mel_rejects_nonpositive_max_frames(max_frames: int) -> None:
    """验证显示帧数必须为正数。

    零和负值均应抛出 ValueError，避免无效抽样数量进入绘图流程。
    """
    with pytest.raises(ValueError, match="max-frames 必须为正数"):
        _prepare_mel(np.zeros((2, 5)), 0, False, max_frames)


def test_prepare_mel_checks_finiteness_after_sampling() -> None:
    """验证非有限值检查发生在抽样之后。

    当前实现允许被抽样丢弃的 NaN 和无穷值存在于原始矩阵中， 但保留全部帧时必须拒绝这些值；此测试记录现有行为而非推荐规则。 若以后改为检查完整输入，应同步调整本测试与对应行为说明。
    """
    array = np.array([[0, np.nan, 2, np.inf, 4], [5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(_prepare_mel(array, 0, False, 3), array[:, [0, 2, 4]])
    with pytest.raises(ValueError, match="Mel 张量包含 NaN 或无穷值"):
        _prepare_mel(array, 0, False, 5)


def test_cli_entries_use_analysis_api() -> None:
    """验证脚本入口复用核心分析接口。

    使用对象身份比较确认入口直接引用核心函数与常量， 而不是在脚本层保留独立的统计或可视化实现。
    """
    assert plot_training_metrics.analyze is analyze
    assert plot_training_metrics.DEFAULT_METRICS is DEFAULT_METRICS
    assert compare_experiments.compare is compare
    assert visualize_mel.visualize is visualize
    assert visualize_mel._prepare_mel is _prepare_mel


@pytest.mark.parametrize(
    "script",
    ["plot_training_metrics.py", "compare_experiments.py", "visualize_mel.py"],
)
def test_cli_help_from_external_cwd(
    tmp_path: Path,
    script: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证脚本在外部工作目录下正常显示帮助。

    为三个固定入口设置 --help 参数，检查退出码、帮助文本和错误输出。 monkeypatch 恢复工作目录、环境变量、argv 和导入路径，避免测试间污染。
    已导入模块仍保留在当前进程中，因此本测试不证明全新进程的导入行为。
    """
    script_path = PROJECT_ROOT / "scripts" / "analysis" / script
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(sys, "argv", [str(script_path), "--help"])
    monkeypatch.setattr(sys, "path", sys.path.copy())

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script_path), run_name="__main__")

    assert exit_info.value.code == 0
    output = capsys.readouterr()
    assert "--help" in output.out
    assert script in output.out
    assert not output.err
