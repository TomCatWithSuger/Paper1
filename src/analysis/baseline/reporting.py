"""生成基线审计表格、实验计划与报告。"""

# ====================
# 1. Imports
# ====================


import csv
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from src.analysis.baseline.runs import (
    CONTROLLED_CONFIG_PATHS,
    RunSummary,
    _metric,
    _nested_value,
)

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def _controlled_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    rows = []
    for config_path in CONTROLLED_CONFIG_PATHS:
        values = [_nested_value(run.config, config_path) for run in runs]
        rows.append(
            {
                "config_path": config_path,
                "B1": _format_value(values[0]),
                "B2": _format_value(values[1]),
                "B3": _format_value(values[2]),
                "consistent": values[0] == values[1] == values[2],
            }
        )
    return rows


def _experiment_plan_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    run_paths = {run.label: str(run.run_dir) for run in runs}
    return [
        {
            "experiment_id": "nfe_b1_01",
            "model": "B1",
            "nfe": 1,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_05",
            "model": "B1",
            "nfe": 5,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_10",
            "model": "B1",
            "nfe": 10,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_50",
            "model": "B1",
            "nfe": 50,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b2_01",
            "model": "B2",
            "nfe": 1,
            "ablation": "equal_time_probability=0.75",
            "run_dir": run_paths["B2"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "ablation_b2_equal_time_00",
            "model": "B2",
            "nfe": 1,
            "ablation": "equal_time_probability=0.0",
            "run_dir": "需单独训练",
            "status": "待训练",
        },
        {
            "experiment_id": "nfe_b3_01",
            "model": "B3",
            "nfe": 1,
            "ablation": "当前完整方法",
            "run_dir": run_paths["B3"],
            "status": "待运行统一转换评估",
        },
    ]


def _report_text(
    runs: Sequence[RunSummary],
    dataset_summary: dict[str, Any],
    controlled_rows: Sequence[dict[str, Any]],
) -> str:
    inconsistent = [row["config_path"] for row in controlled_rows if not row["consistent"]]
    summaries = {run.label: run for run in runs}
    lines = [
        "# B1/B2/B3 基线验证审计",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        "",
        "## 1. 当前结论",
        "",
        "现有记录只能证明三种训练目标都完成了训练，**尚不能证明 MeanFlow 更适合 VC**。",
        "`test/loss` 是各自训练目标的自重构损失，不是跨说话人转换质量；B3 的损失定义与 B1/B2 不同，三者数值不能直接排序。",
        "在完成固定测试清单上的统一生成、质量指标和效率计时前，不建议设计第四个模型。",
        "",
        "## 2. 已定位实验",
        "",
        "| 实验 | 运行目录 | 最优检查点 | 最优 val/loss | test/loss |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                (
                    run.label,
                    str(run.run_dir),
                    run.checkpoint,
                    _format_metric(_metric(run, "val/loss", "best")),
                    _format_metric(_metric(run, "test/loss", "final")),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### 训练记录解读",
            "",
            f"- B1 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B1'], 'val/loss', 'best'))}，最佳检查点为 `{summaries['B1'].checkpoint}`。",
            f"- B2 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B2'], 'val/loss', 'best'))}，最佳检查点测试损失为 {_format_metric(_metric(summaries['B2'], 'test/loss', 'final'))}；验证/测试尺度差异较大，统一转换评估前应先复核固定采样和数据分布。",
            f"- B3 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B3'], 'val/loss', 'best'))}；其自适应损失和零输入下界使该数值不能与 B1/B2 的 masked MSE 比较。",
            "",
            "## 3. 数据集适配与固定测试清单",
            "",
            f"- 输入数据根目录：`{dataset_summary['data_root_argument']}`",
            f"- 自动适配后的 VCTK 根目录：`{dataset_summary['resolved_vctk_root']}`",
            f"- 发现样本：{dataset_summary['discovered_samples']}",
            f"- 划分：训练 {dataset_summary['train_samples']} / 验证 {dataset_summary['validation_samples']} / 测试 {dataset_summary['test_samples']}",
            f"- 测试集合：{dataset_summary['test_speakers']} 个说话人 × {dataset_summary['test_sentences']} 个句子",
            f"- 固定测试清单 SHA256：`{dataset_summary['test_manifest_sha256']}`",
            "- 数据适配只解析现有目录结构并生成外部清单，未修改数据集元数据。",
            "",
            "## 4. 受控条件审计",
            "",
        ]
    )
    if inconsistent:
        lines.append("以下条件不一致：" + "、".join(f"`{item}`" for item in inconsistent) + "。")
    else:
        lines.append("数据划分、Mel 参数、条件维度和 U-Net 主干关键参数一致。")

    lines.extend(
        [
            "",
            "已确认的非受控差异包括：B3 使用不同 Adam betas、预热/余弦调度和不同损失；B1 未配置固定多次验证采样；B3 验证仍为随机单次估计。",
            "因此现有最佳 checkpoint 选择过程并不完全等价，后续统一评估必须直接加载固定 checkpoint，而不能比较训练曲线高低。",
            "",
            "## 5. 指标完成状态",
            "",
            "| 类别 | 指标 | 当前状态 | 下一步 |",
            "| --- | --- | --- | --- |",
            "| 内容保持 | CER / WER | 未完成 | 接入与统一声码器输出匹配的 ASR |",
            "| 内容保持 | 内容 embedding 距离 | 未完成 | 对生成音频重新提取内容特征 |",
            "| 说话人保持 | ECAPA/WavLM cosine | 未完成 | 对生成音频重新提取说话人特征 |",
            "| 音质 | Mel 距离 | 可先完成 | 固定随机种子后比较生成 Mel 与目标 Mel |",
            "| 音质 | MOS | 未完成 | 客观链稳定后再设计主观实验 |",
            "| 效率 | NFE | 已定义 | B1: 1/5/10/50；B2/B3: 1 |",
            "| 效率 | RTF / latency | 未完成 | 同设备、同 batch、预热后计时 |",
            "",
            "## 6. 下一轮实验顺序",
            "",
            "1. 使用 `test_manifest.csv` 固定全部 source/reference 组合。",
            "2. 对 B1 执行 1、5、10、50 NFE，对 B2/B3 执行 1 NFE，并保存逐样本生成结果和耗时。",
            "3. 首先完成 Mel 距离、NFE、RTF、latency；接入统一声码器后再完成 CER/WER 和生成音频 embedding 指标。",
            "4. 以逐样本结果计算 `ΔCER = CER_B1(50) - CER_B2(1)`，按说话人、句长和内容复杂度分组。",
            "5. 只有当统一结果显示 B2(1 NFE) 接近 B1(50 NFE)，才能主张 MeanFlow 成功压缩生成步数。",
            "6. 时间采样消融需要重新训练 `equal_time_probability=0.0` 的 B2，不能通过修改现有 checkpoint 推断。",
            "",
            "## 7. 输出记录",
            "",
            "- `run_metrics.csv`：训练日志标量摘要。",
            "- `controlled_conditions.csv`：三组配置的一致性审计。",
            "- `test_manifest.csv`：固定 source/reference 测试清单。",
            "- `dataset_summary.json`：数据适配结果和清单指纹。",
            "- `experiment_plan.csv`：下一轮 NFE 与消融实验记录表。",
            "",
        ]
    )
    return "\n".join(lines)
