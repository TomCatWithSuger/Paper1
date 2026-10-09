# 新增模型组件注意事项

## 1. 当前分层

```text
Config
└── Module
    └── Method
        ├── 一个或多个 Network
        │   └── 网络组件
        └── 少量算法相关组件（可选）
```

各层职责：

| 层级      | 目录                     | 职责                                                |
| --------- | ------------------------ | --------------------------------------------------- |
| Component | `src/models/components/` | 可复用的小型计算单元                                |
| Network   | `src/models/networks/`   | 独立、完整的神经网络前向计算                        |
| Method    | `src/models/methods/`    | 训练目标、损失、扰动和生成算法                      |
| Module    | `src/models/modules/`    | Batch 适配、Lightning 生命周期、日志和优化器        |
| Config    | `configs/model/`         | 选择并组装以上对象                                  |

依赖必须保持从上到下：

- Module 只持有 Method，不保存独立的 Network。
- Method 组织一个或多个 Network，并可按需持有少量条件编码等 Component。
- Network 主要使用 `blocks`、`embeddings` 等网络组件。
- Component 不得依赖具体 Network、Method 或 Module。
- Network 不得依赖具体算法或 Lightning。

## 2. 新增 Component

### 放置位置

```text
src/models/components/
├── blocks/          # 特殊卷积、注意力、残差块
├── embeddings/      # 时间、位置等嵌入
└── conditioning/    # 条件编码、对齐和融合
```

### 步骤

1. 确认 PyTorch 没有直接提供同等功能，避免包装普通 `Linear`、`Conv`、`Norm`。
2. 根据职责放入对应子目录；只有形成新类别时才新增子目录。
3. 让输入、输出张量形状和异常条件明确。
4. 添加组件级测试，覆盖形状、掩码和边界参数。

### 注意事项

- Component 应小而独立，不读取 Batch、配置或 Trainer 状态。
- 只被一个 Network 使用的小型辅助类可以留在该 Network 文件中。
- `conditioning` 可由 Method 直接持有；其他网络结构组件通常只由 Network 使用。

## 3. 新增 Network

### 放置位置

按主要架构家族放置：

```text
src/models/networks/
├── dense/
├── unet/
└── transformer/
```

新架构不属于现有家族时，再建立新的架构子目录。

架构目录内按完整网络的结构变体或前向接口划分文件。详细分类和边界见 `src/models/networks/README.md`。

### 步骤

1. 实现继承 `torch.nn.Module` 的完整网络。
2. 只实现前向预测，不实现损失、加噪、概率路径或采样循环。
3. 从 `components` 组合必要的小组件。
4. 在对应 Method 的 Hydra 配置中，将 Network 放在 `method.network`。
5. 添加前向形状、掩码、梯度和非法参数测试。

配置示例：

```yaml
method:
  _target_: src.models.methods.generative.flow.example.ExampleMethod
  network:
    _target_: src.models.networks.transformer.example.ExampleNetwork
    input_dim: 128
    hidden_dim: 256
```

### 注意事项

- Network 不应根据自己用于 CFM、DDPM 或其他 Method 而实现训练目标或生成流程。
- 相同输入输出契约的 Network 才能通过配置互换。
- 网络结构参数属于 `method.network`，不能放在 Module 顶层。
- 修改已有参数名或注册顺序时，需要评估检查点兼容性。

## 4. 新增 Method

### 放置位置

```text
src/models/methods/
├── discriminative/
├── representation/
└── generative/
    ├── adversarial/
    ├── diffusion/
    ├── latent_variable/
    └── flow/
```

### 基本接口

```python
class ExampleMethod(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def compute_loss(self, inputs: torch.Tensor):
        prediction = self(inputs)
        ...
        return loss, prediction
```

### 步骤

1. 根据判别、表征或生成目标选择目录，再按生成机制选择子目录。
2. 让 Method 持有 Network，禁止每次调用时额外传入 Network。
3. 实现算法需要的目标构造、损失、扰动和采样。
4. 条件生成可额外持有 `conditioning` 组件。
5. 添加固定随机种子的损失、梯度和采样测试。

### 注意事项

- Method 不读取原始 Batch 字段，Batch 适配属于 Module。
- Method 不调用 `self.log()`，也不访问 Trainer。
- 通用损失优先放在 `src/losses/`；算法专用损失留在 Method。
- 算法超参数属于 Method，例如扩散步数、积分步数和损失权重。

## 5. 新增 Module

只有训练生命周期发生变化时才新增 Module。仅更换 Method 或 Network 不需要新 Module。

Module 按训练生命周期、Batch 契约、指标体系和优化流程分类，不按算法或网络架构分类。详细判断标准见 `src/models/modules/README.md`。

### 基本接口

```python
class ExampleLitModule(LightningModule):
    def __init__(self, method, optimizer, scheduler=None, compile=False):
        super().__init__()
        self.method = method

    def forward(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def model_step(self, batch):
        inputs = ...
        return self.method.compute_loss(inputs)
```

### 需要新增 Module 的情况

- 使用不同的 Batch 结构或指标。
- 需要多个优化器或手动优化，例如 GAN。
- 验证、测试或预测流程明显不同。
- 需要特殊的调度器间隔或固定多次评估。
- 需要独立且稳定的检查点恢复策略。

### 注意事项

- Module 不得保存 `self.net`，统一通过 `self.method.network` 访问。
- Module 不实现算法公式，只负责训练组织。
- `torch.compile()`只替换 `self.method.network`。
- 多个模型训练流程相同时，应复用已有 Module。

## 6. 添加 Config

完整配置示例：

```yaml
_target_: src.models.modules.generative.ExampleLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.0002

scheduler: null

method:
  _target_: src.models.methods.generative.flow.example.ExampleMethod
  integration_steps: 50
  network:
    _target_: src.models.networks.unet.example.ExampleUNet
    input_channels: 80
    hidden_channels: 256

compile: false
```

配置必须满足：

- Module 顶层包含 `method`，不包含 `net`。
- Network 必须嵌套在 `method.network`。
- 算法参数放在 Method，网络参数放在 Network。
- `_target_` 使用当前真实 Python 路径。

## 7. 测试与检查

至少覆盖：

1. Component 或 Network 的输入输出形状。
2. Method 的损失、梯度和生成结果。
3. Module 的 `model_step()` 与 Lightning 生命周期。
4. Hydra 配置可以正常实例化。
5. 旧检查点需要兼容时，严格检查缺失键、额外键和形状错误。

推荐运行：

```bash
pytest tests/models tests/test_configs.py
pre-commit run --all-files
```

提交前确认：

- 没有旧路径或重复实现。
- Module 没有 `self.net`。
- Method 持有 Network。
- Network 不包含损失和采样算法。
- Component 没有反向依赖上层。
- 配置、测试和文档中的路径已经同步更新。
