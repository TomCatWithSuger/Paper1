# 分析产物导航

各实验目录保留原名和内部结构。文件可能被忽略规则排除，其他机器需具备对应本地产物。

| 目录                          | 阶段 / 用途                                  | 主要入口                                                                                          |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `baseline_validation/`        | 基线审计及固定测试清单                       | `baseline_validation_report.md`、`test_manifest.csv`                                              |
| `latest_training_20261023/`   | P0 最新三组训练统计                          | `summary.json`、`scalars.csv`                                                                     |
| `p1_latest_20261024/`         | P1 最新统一转换评估                          | `README.md`、`full/`；旧 `smoke/` 已拒绝                                                          |
| `p2_mechanisms_20261024/`     | P2 区间、JVP、条件及 source_time 诊断        | `README.md`、`d1/`、`d2/`、`conversion/`                                                          |
| `b2_endpoint_pilot_20261024/` | P3 两臂各 5000 步的受控实验                  | `README.md`、`evaluation/`、`training_statistics.csv`                                             |
| `p4_identity_20261026/`       | P4 B3 身份分层及盲听准备                     | `protocol.json`、`existing_paired.csv`、`listening/`；独立评分完成后另有 `campplus/COMPLETE.json` |
| `b3_paper_audit_20261026/`    | 原论文文本、配置/权重审计及 padding 修复验证 | `paper.txt`、`audit.json`、`source/`                                                              |
| `external_vc_20261026/`       | 接近B1规模的官方开源模型初测                 | `README.md`、`manifest.csv`、`protocol.json`                                                      |
| `unified_evaluation/`         | 较早的统一 Mel 评估                          | `unified_evaluation_report.md`                                                                    |
| `vocoder_evaluation/`         | 较早的波形评估                               | `vocoder_evaluation_report.md`                                                                    |
| `vocoder_evaluation_smoke/`   | 较早的声码器冒烟                             | `vocoder_evaluation_report.md`                                                                    |

旧波形评估与 P1 的权重、Mel 适配不同，绝对分数不能直接横比。当前研究判断见[阶段总结](../docs/research/%E7%A0%94%E7%A9%B6%E9%98%B6%E6%AE%B5%E6%80%BB%E7%BB%93%E4%B8%8E%E4%B8%8B%E4%B8%80%E6%AD%A5%E5%86%B3%E7%AD%96_20261026.md)。

新增实验使用独立、含日期的目录；不要覆盖已有训练、账本或指纹。历史 `runtime/` 中的源码是运行时快照，不随当前代码目录调整而修改。
