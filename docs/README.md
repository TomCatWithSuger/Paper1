# 项目文档导航

当前进展和下一步决策以[研究阶段总结](research/%E7%A0%94%E7%A9%B6%E9%98%B6%E6%AE%B5%E6%80%BB%E7%BB%93%E4%B8%8E%E4%B8%8B%E4%B8%80%E6%AD%A5%E5%86%B3%E7%AD%96_20261026.md)为准。历史报告保留当时的判断，不代表全部建议仍应执行。

最新优先级：先处理[B3 论文实现核对](diagnostics/B3%E8%AE%BA%E6%96%87%E5%AE%9E%E7%8E%B0%E6%A0%B8%E5%AF%B9_20261026.md)中的复现缺口。已修复共享条件归一化的 padding 问题，尚未验证质量收益；暂不将源身份现象定为论文主线。

外部能力对照见[近参数量开源 VC 筛选](research/%E5%BC%80%E6%BA%90VC%E8%BF%91%E5%8F%82%E6%95%B0%E6%A8%A1%E5%9E%8B%E7%AD%9B%E9%80%89_20261026.md)及[已完成的20对初测](evaluation/%E5%BC%80%E6%BA%90%E8%BF%91%E5%8F%82%E6%95%B0VC%E5%88%9D%E6%B5%8B_20261027.md)：MeanVC/MeanVC2两套官方系统、七配置、140条统一评分，附本地音频对照。

| 目录            | 内容                          | 推荐入口                                                                                                                                                                                                                                                                                      |
| --------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `baselines/`    | B1/B2/B3 实现、训练及问题排查 | [B1](baselines/B1%E5%AE%9E%E9%AA%8C%E5%AE%9E%E7%8E%B0%E8%AF%B4%E6%98%8E.md)、[B2](baselines/B2%E5%AE%9E%E9%AA%8C%E5%AE%9E%E7%8E%B0%E8%AF%B4%E6%98%8E.md)、[B3](baselines/B3%E5%AE%9E%E9%AA%8C%E5%AE%9E%E7%8E%B0%E8%AF%B4%E6%98%8E.md)                                                         |
| `evaluation/`   | 统一评估协议与质量结果        | [P1 最新三组评估](evaluation/P1%E6%9C%80%E6%96%B0%E4%B8%89%E7%BB%84%E7%BB%9F%E4%B8%80%E8%AF%AD%E9%9F%B3%E8%BD%AC%E6%8D%A2%E8%AF%84%E4%BC%B0.md)                                                                                                                                               |
| `diagnostics/`  | 冻结模型诊断及端点覆盖干预    | [P2 诊断](diagnostics/P2%E6%9C%BA%E5%88%B6%E8%AF%8A%E6%96%AD%E4%B8%8E%E4%B8%8B%E4%B8%80%E6%AD%A5%E6%9C%80%E5%B0%8F%E5%AE%9E%E9%AA%8C.md)、[端点 pilot](diagnostics/B2%E5%AE%8C%E6%95%B4%E7%AB%AF%E7%82%B9%E8%A6%86%E7%9B%96pilot_20261024.md)                                                 |
| `research/`     | 阶段总结、论文方向和相关工作  | [当前总结与决策](research/%E7%A0%94%E7%A9%B6%E9%98%B6%E6%AE%B5%E6%80%BB%E7%BB%93%E4%B8%8E%E4%B8%8B%E4%B8%80%E6%AD%A5%E5%86%B3%E7%AD%96_20261026.md)、[相关工作核查](research/%E4%B8%80%E6%AD%A5%E8%AF%AD%E9%9F%B3%E8%BD%AC%E6%8D%A2%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9C%E6%A0%B8%E6%9F%A5.md) |
| `architecture/` | 模型结构与代码职责            | [模型结构](architecture/src_models%E7%BB%93%E6%9E%84%E8%AF%B4%E6%98%8E.md)                                                                                                                                                                                                                    |

代码入口见 [scripts/analysis](../scripts/analysis/README.md)，实现职责见 [src/analysis](../src/analysis/README.md)，原始产物导航见 [analysis_results](../analysis_results/README.md)。

P4 当前状态见[B3 源身份验证](diagnostics/P4%E6%BA%90%E8%BA%AB%E4%BB%BD%E7%8B%AC%E7%AB%8B%E9%AA%8C%E8%AF%81_20261026.md)：已完成同源评分分层、CAMPPlus 交叉验证及盲听准备，人工听测待完成。

新增文档放入对应主题目录；实验报告注明日期、输入权重、评估协议、结论限制和产物位置。原始结果保留原目录，避免破坏清单中的绝对路径、代码快照和指纹。
