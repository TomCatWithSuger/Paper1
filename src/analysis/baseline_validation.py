"""审计 B1、B2、B3 基线实验并生成可复现实验记录。"""

# ====================
# 1. Imports
# ====================


import json
from pathlib import Path

from src.analysis.baseline.dataset import _build_test_manifest as _build_test_manifest
from src.analysis.baseline.dataset import _resolve_vctk_root as _resolve_vctk_root
from src.analysis.baseline.dataset import _shared_split_config as _shared_split_config
from src.analysis.baseline.reporting import _controlled_rows as _controlled_rows
from src.analysis.baseline.reporting import (
    _experiment_plan_rows as _experiment_plan_rows,
)
from src.analysis.baseline.reporting import _format_metric as _format_metric
from src.analysis.baseline.reporting import _format_value as _format_value
from src.analysis.baseline.reporting import _report_text as _report_text
from src.analysis.baseline.reporting import _write_csv as _write_csv
from src.analysis.baseline.runs import (
    CONTROLLED_CONFIG_PATHS as CONTROLLED_CONFIG_PATHS,
)
from src.analysis.baseline.runs import RUN_LABEL_TAGS as RUN_LABEL_TAGS
from src.analysis.baseline.runs import RunSummary as RunSummary
from src.analysis.baseline.runs import _checkpoint_name as _checkpoint_name
from src.analysis.baseline.runs import _discover_runs as _discover_runs
from src.analysis.baseline.runs import _metric as _metric
from src.analysis.baseline.runs import _nested_value as _nested_value
from src.analysis.baseline.runs import _read_config as _read_config
from src.analysis.baseline.runs import _read_tags as _read_tags
from src.analysis.baseline.runs import _scalar_metrics as _scalar_metrics

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _execute(runs_root: Path, data_root: Path, output_dir: Path) -> dict[str, Path]:
    runs_root = runs_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(runs_root)
    vctk_root = _resolve_vctk_root(data_root.expanduser().resolve())
    manifest_rows, dataset_summary = _build_test_manifest(vctk_root, runs)
    controlled_rows = _controlled_rows(runs)
    experiment_rows = _experiment_plan_rows(runs)

    metric_rows = [
        {"experiment": run.label, "run_dir": str(run.run_dir), **metric}
        for run in runs
        for metric in run.metrics
    ]
    paths = {
        "report": output_dir / "baseline_validation_report.md",
        "metrics": output_dir / "run_metrics.csv",
        "conditions": output_dir / "controlled_conditions.csv",
        "manifest": output_dir / "test_manifest.csv",
        "dataset": output_dir / "dataset_summary.json",
        "plan": output_dir / "experiment_plan.csv",
    }
    _write_csv(
        paths["metrics"],
        metric_rows,
        ("experiment", "run_dir", "metric", "count", "best", "best_step", "final", "final_step"),
    )
    _write_csv(
        paths["conditions"],
        controlled_rows,
        ("config_path", "B1", "B2", "B3", "consistent"),
    )
    _write_csv(
        paths["manifest"],
        manifest_rows,
        (
            "index",
            "source_id",
            "source_speaker_id",
            "sentence_id",
            "reference_id",
            "reference_speaker_id",
            "text",
            "audio_path",
        ),
    )
    _write_csv(
        paths["plan"],
        experiment_rows,
        ("experiment_id", "model", "nfe", "ablation", "run_dir", "status"),
    )
    paths["dataset"].write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        _report_text(runs, dataset_summary, controlled_rows),
        encoding="utf-8",
    )
    return paths


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def analyze_baselines(runs_root: Path, data_root: Path, output_dir: Path) -> dict[str, Path]:
    """生成 B1/B2/B3 审计报告、固定测试清单和后续实验记录。"""
    return _execute(runs_root, data_root, output_dir)
