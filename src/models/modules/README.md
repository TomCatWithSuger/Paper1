# Modules 分类说明

`modules` 按训练生命周期、Batch 契约、指标体系和优化流程分类，不按 Network 架构或 Method 的算法谱系分类。

```text
modules/
├── generative.py               # 通用单优化器与标量损失生命周期
├── reconstruction.py           # 重构类任务的预测与附加指标
├── adversarial.py              # 双优化器与手动优化生命周期
├── classification.py           # 多分类 Batch 与准确率指标生命周期
└── voice_flow_module.py        # 条件语音 Batch、固定采样评估和检查点兼容
```

当前文件数量较少，保持扁平结构。只有出现多个稳定且相互独立的生命周期家族时，才考虑增加子目录。

## 核心原则

1. Module 持有 Method，并将数据批次适配为 Method 的明确输入。
2. Module 负责 Lightning 生命周期、指标、日志、优化器和调度器。
3. Method 负责训练目标、损失、扰动及生成语义，Network 负责前向参数化。
4. 仅更换 Network 或 Method 时，不应新增 Module。
5. 只有训练生命周期、Batch 契约、指标或优化流程发生实质变化时，才新增或拆分 Module。

Module 与 Method 不要求一一对应。多个 Method 可以共用同一个 Module；同一类 Method 在数据批次或训练流程不同时，也可以使用不同 Module。

## 层级边界

| 层级       | 职责                                           |
| ---------- | ---------------------------------------------- |
| Network    | 神经网络结构与确定性前向计算                   |
| Method     | 算法目标、损失、扰动、路径和生成逻辑           |
| Module     | Batch 适配、训练与评估生命周期、日志及优化配置 |
| DataModule | 数据准备、数据集划分、DataLoader 和批次生产    |

Module 可以：

- 从 Batch 中提取并整理 Method 所需张量。
- 实现 `training_step()`、`validation_step()`、`test_step()` 和 `predict_step()`。
- 管理 TorchMetrics、日志名称和跨 epoch 聚合。
- 配置一个或多个优化器及学习率调度器。
- 根据训练阶段调用 `torch.compile()`。
- 处理 Lightning 检查点兼容和训练状态恢复约束。
- 为可复现评估创建随机数生成器并组织固定次数采样。

Module 不应：

- 重新实现 Method 中的损失公式、概率路径或采样算法。
- 直接构造或替换 Network 的内部层。
- 将原始 Batch 字段传入只接受规范张量的 Network。
- 根据实验名称复制相同的训练生命周期。
- 保存独立于 Method 的重复 Network 引用。

## 当前生命周期类型

### `generative.py`

`GenerativeModule` 提供当前最通用的自动优化生命周期：

- 单优化器。
- 可选 epoch 级调度器。
- 以 `loss` 为主要指标。
- 默认从 `(inputs, targets)` 形式的 Batch 中取输入。
- 统一完成训练、验证、测试、预测和旧检查点键映射。

`DDPMLitModule`、`FlowMatchingLitModule` 和 `ScoreBasedLitModule` 当前共享完全相同的生命周期，保留独立类名用于配置可读性和兼容性。`DiTLitModule` 只覆盖波形 Batch 适配。

虽然文件名为 `generative.py`，其基类也被 AE 等具有相同生命周期的任务复用。分类依据是生命周期复用关系，而不是声称所有子类都属于同一种算法。

### `reconstruction.py`

在通用单优化器生命周期上增加重构任务差异：

- `AELitModule` 调整预测输出。
- `VAELitModule` 记录重构损失和 KL 损失。
- `UNetLitModule` 适配带波形与注意力掩码的 Batch。

损失公式仍由对应 Method 实现，Module 只负责拆解返回值并记录指标。

### `adversarial.py`

`GANLitModule` 使用与通用自动优化不同的生命周期：

- 生成器和判别器使用不同优化器。
- 关闭 Lightning 自动优化。
- 显式切换优化器并执行两阶段更新。
- 分别记录生成器和判别器损失。

双网络结构和对抗损失属于 Network 与 Method；手动优化顺序属于 Module。

### `classification.py`

`ClassificationLitModule` 负责通用多分类实验的生命周期：

- 适配 `(inputs, targets)` 分类 Batch。
- 通过 `num_classes` 配置类别数。
- 记录分类损失、准确率和最佳验证准确率。
- 使用单优化器及可选调度器。

数据集负责产生符合该契约的输入和整数类别标签；MNIST 或其他多分类数据集均可复用该 Module。分类损失和预测逻辑由 `Classification` Method 提供。

### `voice_flow_module.py`

`VoiceFlowModule` 负责条件语音流方法共用的生命周期：

- 将语音字典 Batch 转换为规范训练和推理输入。
- 统一组织内容、说话人、Mel 和掩码字段。
- 支持固定随机种子下的多次采样评估。
- 支持 warmup cosine 或配置注入的调度器。
- 处理旧检查点键映射及优化器恢复约束。

CFM、Conditional Mean Flow 和 MeanVoiceFlow 可以共用该 Module，因为它们具有相同的语音 Batch 与生命周期契约；算法目标仍分别位于各自 Method 中。

## 何时新增 Module

满足以下任一条件时，可以考虑新增 Module：

1. Batch 结构或输入适配方式明显不同。
2. 指标体系和验证逻辑无法由现有 Module 参数化表达。
3. 优化器数量、手动优化顺序或梯度更新策略不同。
4. 预测、测试或固定采样评估流程明显不同。
5. 检查点恢复需要独立且稳定的兼容策略。

以下情况通常不应新增 Module：

- 只更换 Dense、U-Net 或 Transformer。
- 只调整扩散步数、积分步数或损失权重。
- 只增加条件输入，而现有 Batch 适配已经能够提供该输入。
- 只为不同实验配置使用不同超参数。

## 新 Module 的实现要求

新增 Module 时应：

1. 通过构造函数接收 Method、优化器工厂和可选调度器工厂。
2. 使用 `self.method` 作为唯一算法入口，不额外保存 `self.net`。
3. 在 `model_step()` 中完成 Batch 到 Method 参数的适配。
4. 让 Method 返回损失和算法输出，Module 只记录和组织这些值。
5. 明确 `predict_step()` 返回训练输出、重构结果还是生成样本。
6. 为优化器分组、调度器间隔和监控指标提供稳定配置。
7. 需要旧检查点兼容时，严格检测缺失键、额外键和键映射冲突。
8. 添加训练、验证、测试、预测和 Hydra 实例化测试。

命名应优先描述稳定的训练生命周期或 Batch 契约。不要仅因新增一种 Method、Network、数据模态变体或实验名称而复制 Module。
