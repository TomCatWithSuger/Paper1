# Methods 分类说明

`methods` 按算法的主要学习职责和核心机制分类，不按网络结构、数据类型或 Lightning 训练流程分类。这是一套工程职责分类，不要求各类算法在数学上严格互斥。

```text
methods/
├── base.py
├── discriminative/             # 判别学习
│   └── classification.py
├── representation/             # 表征学习
│   ├── autoencoder.py
│   └── masked.py
└── generative/                 # 生成学习
    ├── adversarial/
    │   └── gan.py
    ├── latent_variable/
    │   └── vae.py
    ├── diffusion/
    │   ├── ddpm.py
    │   └── score_based.py
    └── flow/
        ├── flow_matching.py
        ├── cfm/
        └── mean_flow/
```

## 组件职责

| 组件       | 职责                                                       |
| ---------- | ---------------------------------------------------------- |
| Network    | 神经网络结构、参数化函数与前向计算                         |
| Method     | 组织一个或多个 Network，定义算法目标、损失、扰动及生成逻辑 |
| Module     | Lightning 训练生命周期、日志和优化器配置                   |
| DataModule | 数据加载、预处理和批次组织                                 |

Method 不得依赖 Lightning。复杂的 Sampler、Scheduler 或 Solver 可以拆成独立组件，由 Method 组织调用，不要求全部实现在 Method 内部。Method 与 Module 也不要求目录一一对应。

## 分类原则

Method 描述“学习什么、目标如何构造、如何生成”，因此遵循以下原则：

1. 顶层根据算法的主要学习职责确定唯一工程归属。
2. 生成方法继续根据核心概率建模、训练或生成机制分类。
3. 辅助损失、数据类型和使用的 Network 结构不决定类别。
4. 同时符合多个类别时，以论文和实现中的主要优化目标及算法职责为准。

例如，普通 AE 和 VAE 都包含重构目标，但普通 AE 以确定性编码和重构为核心，属于表征学习；VAE 以概率潜变量建模和变分推断为核心，属于生成学习。AE 可以作为生成系统的一部分，但这不会改变其主要算法职责。

## 各类别判断标准

### `discriminative/`

主要目标是根据输入预测标签、类别或连续值：

```text
x → y
```

典型算法包括分类、回归和序列标注。判断重点是算法是否主要优化对目标变量的预测。

### `representation/`

主要目标是学习可复用的数据表示，通常通过重构、掩码预测或对比目标训练。典型算法包括普通 Autoencoder、Masked Modeling 和对比学习。

是否能够被用于生成系统不是绝对判断标准，应根据算法自身的主要优化目标和职责归类。

### `generative/`

主要目标是建模数据分布、概率路径或生成过程，并从噪声、潜变量或历史状态生成样本。按核心机制细分：

| 子目录             | 判断依据                                             |
| ------------------ | ---------------------------------------------------- |
| `adversarial/`     | 以生成器和判别器之间的对抗目标为核心                 |
| `latent_variable/` | 以显式潜变量概率模型和变分推断等机制为核心，例如 VAE |
| `diffusion/`       | 以加噪、去噪或基于分数的随机生成过程为核心           |
| `flow/`            | 以连续向量场、概率路径或 ODE 生成过程为核心          |

#### Diffusion、Score-based 与 Flow 的边界

- DDPM 定义离散前向扩散与反向去噪过程，属于 `diffusion/`。
- Score Matching 本身是一类训练目标；当实现包含多噪声尺度扰动、分数网络和随机采样过程时，作为 Score-based Generative Method 放入 `diffusion/`。
- Flow Matching 学习连续向量场，并通常通过 ODE 积分生成样本，属于 `flow/`。
- Conditional Flow Matching 是 Flow Matching 的具体形式，放入 `flow/cfm/`。
- MeanFlow 学习平均速度场，与标准 CFM 不等同，但当前仍属于流方法家族，放入 `flow/mean_flow/`。

Diffusion 与 Flow 在连续时间建模上存在联系，但目录按实现的核心训练目标和生成机制确定唯一归属。

## 新算法如何归类

按以下顺序判断：

1. 主要职责是否是预测标签或数值？是则放入 `discriminative/`。
2. 主要职责是否是通过重构、掩码预测或对比目标学习表示？是则放入 `representation/`。
3. 主要职责是否是建模或采样数据分布？是则放入 `generative/`，再根据对抗、潜变量、扩散、流或自回归机制选择子目录。
4. 若多个类别均适用，以核心训练目标和算法职责为准，不按 Network 名称、条件形式或数据模态判断。

未来出现自回归生成方法时，可以增加 `generative/autoregressive/`。强化学习属于基于环境交互和奖励优化的独立训练范式；确有相关实现时可以增加顶层 `reinforcement/`，但不将其视为与现有类别数学上完全同维的分类。

新增子目录前应确认现有类别无法准确表达该算法。不要仅因实验名称、条件形式、数据模态或 Network 不同而创建新类别；同一算法家族的变体应保留在同一目录中。
