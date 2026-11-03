---
name: add-research-experiment
description: 用于新增或修改神经网络、学习方法、数据模块和实验配置，以及模型重构和实验接入。Use when adding a Network, Method, DataModule, Lightning Module, or research experiment; decide module placement, dependencies, ownership, and focused validation before implementation.
---

# 新增研究实验

## 规则来源

先阅读根 [AGENTS.md](../../../AGENTS.md) 和[研究模板开发规范](../../../docs/architecture/%E7%A0%94%E7%A9%B6%E6%A8%A1%E6%9D%BF%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83.md)。具体当前实现以[模型结构说明](../../../docs/architecture/src_models%E7%BB%93%E6%9E%84%E8%AF%B4%E6%98%8E.md)和相邻代码为准。

本 Skill 指导实现流程，不自动保证架构正确，也不授予下载、安装、正式训练、删除数据或扩大修改范围的权限。

## 1. 定义最小任务

明确输入、输出、目标、预测/生成语义和验收条件。只有缺失信息会改变实现位置或算法含义时才询问用户。

新增实验先用合成数据建立最小验证，不自动使用外部数据和预训练权重，不启动正式长时间训练。

## 2. 路由与复用

从一个相关实现、配置或测试开始检查。输出简短修改清单，说明每项所属层与复用对象：

| 内容                           | 位置                                  |
| ------------------------------ | ------------------------------------- |
| 读取、变换、划分、Batch 生产   | `src/data`                            |
| 可复用局部计算                 | `src/models/components`               |
| 稳定前向接口                   | `src/models/networks`                 |
| 目标、损失组合、采样、阶段语义 | `src/models/methods`                  |
| 训练生命周期和 Batch 适配      | `src/models/modules`                  |
| 独立数学原语                   | `src/models/algorithms`               |
| 实验组装                       | `configs/model`、`configs/experiment` |
| 契约与生命周期验证             | 复用相邻测试文件                      |

仅更换网络或目标不自动新增 Module；先确认 Batch、优化角色、指标和预测契约是否相同。不通过增加目录代替职责判断。

## 3. 明确契约与所有权

- 写明输入输出形状、轴含义、Mask 极性、时间/条件字段、目标语义。
- 明确 Parameter、Buffer、随机数与数据副作用的所有者。
- Method 接收依赖，不在核心流程中调用 Hydra 或通过类名分支创建具体网络。
- 在唯一组装点使用 Hydra；共享对象只构造一次，测试身份，不能把重复配置等同于共享。
- 多优化器声明阶段目标、顺序、角色和梯度边界；执行器不重新实现算法。
- 底层不得直接或间接依赖 Lightning、训练入口和任务数据加载器。

## 4. 实现与聚焦验证

按 AGENTS 的 Python 文件章节和中文说明约定实现，省略空章节。新增文件不超过 500 物理行，不能靠删注释或压缩代码绕过。

发现存量超长文件时报告位置和拆分建议；不得顺手进行未授权的大规模重构。公开 API、配置目标或状态键名变化应有兼容方案或明确迁移说明。

每完成一个行为切片，立即运行对应测试：先局部数值/梯度，再配置组合和生命周期。尽量复用测试文件及夹具，不创建只覆盖成功导入的空壳测试。

## 5. 验收

- 兼容网络可替换，错误 Shape/条件被拒绝。
- Method 可独立计算和反传；提供生成接口时验证生成语义。
- 多网络测试角色参数、实际更新范围、Buffer 和 RNG 行为。
- 组装验证参数无意外副本；测试训练、评估、预测和所需恢复路径。
- 按已安装并配置的工具执行格式、类型、依赖图、长度及测试检查。
- 在已有 Conda 环境运行 `python scripts/quality.py all`；可单独选择 `architecture`、`length`、`types`、`templates`、`selftest` 定位失败。格式检查沿用现有 pre-commit 工具链。
- Pyright 仅使用已有 Node，不在 PATH 时设置 `QUALITY_NODE`。门禁自测使用临时文件，不加载论文数据夹具。
- 存量问题导致全仓失败时，列出本次引入与已有问题；不得全局关闭检查或擅自展开历史文件重构。
- 不存在工具时明确报告“未执行”，不安装、不伪称通过，也不因文档出现命令就认为门禁已接入。

## 6. 交付

说明修改位置、复用关系、运行命令、实际测试结果及限制。同步实验说明与配置，不把仅在 CPU 合成数据上的结果写成 GPU、分布式或精确续跑保证。
