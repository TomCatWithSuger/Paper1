# 分析模块

## 入口与职责

原有入口保持不变，调用脚本无需调整：

| 入口                       | 对外函数                   | 职责                                      |
| -------------------------- | -------------------------- | ----------------------------------------- |
| `baseline_validation.py`   | `analyze_baselines`        | 审计 B1/B2/B3，生成固定测试清单和实验计划 |
| `unified_vc_evaluation.py` | `evaluate_unified`         | 调度 Mel 生成、逐样本评估与断点续跑       |
| `vocoder_evaluation.py`    | `evaluate_vocoder_metrics` | 调度波形合成、波形指标与真实 Mel 重建对照 |

## 内部实现

```text
baseline/
  runs.py          实验发现、配置、检查点与训练指标
  dataset.py       数据划分核验与固定测试清单
  reporting.py     受控条件、实验计划、表格与报告
unified/
  checkpoints.py   评估任务定义与模型检查点加载
  data.py          转换配对、特征缓存与批次准备
  identity.py      评估身份、指纹与续跑一致性检查
  records.py       CSV 和逐样本账本读写
  reporting.py     指标聚合、样本差异与报告
vocoder/
  definitions.py   音频参数与输出字段
  audio.py         音频读写、Mel 提取与频带适配
  metrics.py       文本错误率、F0 与能量指标
  models.py        本地 Vocos、Whisper 与说话人评分模型
  reporting.py     账本、表格、指标聚合与报告
```

入口负责流程控制，子模块负责具体计算和资源处理。子模块不反向导入入口，避免循环依赖。
原模块继续导出已迁移的内部符号，以兼容现有调用和测试；新增实现应直接引用所属子模块。

训练指标、实验对比和 Mel 可视化等其他分析模块保持原结构。

## 验证

```bash
conda activate Lightning-Template
python -m pytest tests/analysis -q
```

现有分析测试使用临时数据和模型替身，不要求远程测试机器具备预训练权重，也不下载模型。

波形评估优先从 `--asr-model-dir` 指定的目录加载 Whisper，默认是项目内的
`pretrained/whisper-small`，始终只读取本地文件。模型目录、必要文件或依赖缺失时，
会提示原因并跳过 WER/CER，以 `NaN` 标记未计算结果；报告会注明，其他指标继续计算。
实际波形合成和说话人评分仍需各自的本地模型权重。

补齐本地 Whisper 后，重新运行相同评估命令会重算此前缺少 WER/CER 的样本。
账本保留历史记录，结果表和聚合指标只使用每个转换配对的最新记录，避免重复计数。
