# `tests` 文件说明

`tests` 目录存放项目的自动化测试代码，主要使用 `pytest` 验证 Hydra 配置、MNIST 数据模块、模型训练、模型评估、断点续训和参数搜索等功能。

## 根目录文件

### `__init__.py`

- 作用：将 `tests` 标记为 Python 包。
- 测试目的：本身不包含测试，主要用于支持测试模块之间的导入。

### `conftest.py`

- 作用：定义多个测试文件共享的 `pytest fixture`。
- 主要 fixture：
  - `cfg_train_global`：加载默认训练配置，并将训练规模缩小为适合测试的参数。
  - `cfg_eval_global`：加载默认评估配置，并限制测试批次数量。
  - `cfg_train`：为单个测试复制训练配置，并使用临时目录保存日志和输出。
  - `cfg_eval`：为单个测试复制评估配置，并使用临时目录保存日志和输出。
- 测试目的：为训练和评估测试提供统一、隔离且运行时间较短的 Hydra 配置。
- 是否直接执行测试：否，它负责准备测试环境和测试数据。

### `test_configs.py`

- 作用：测试训练配置和评估配置能否正确加载及实例化。
- 测试内容：
  - 检查配置中是否存在 `data`、`model` 和 `trainer`。
  - 使用 Hydra 实例化数据模块、模型和 Lightning Trainer。
- 测试目的：尽早发现配置缺失、配置路径错误、类名错误或构造参数不匹配等问题。

### `test_datamodules.py`

- 作用：测试 `MNISTDataModule` 的数据准备、数据集划分和 DataLoader。
- 测试内容：
  - 分别测试批大小 `32` 和 `128`。
  - 验证 MNIST 数据目录是否存在。
  - 验证训练集、验证集和测试集是否成功初始化。
  - 验证三个 DataLoader 是否可以创建。
  - 验证数据总量是否为 `70,000`。
  - 验证一个批次的大小以及输入、标签的数据类型。
- 测试目的：确保 MNIST 数据能够正确准备，并以模型需要的格式提供批次数据。

### `test_eval.py`

- 作用：测试从训练到加载检查点并执行评估的完整流程。
- 测试内容：
  - 训练模型一个 epoch。
  - 检查是否生成 `last.ckpt`。
  - 使用生成的检查点执行独立评估。
  - 验证测试准确率大于 `0`。
  - 比较训练结束时和独立评估时得到的测试准确率。
- 测试目的：确保模型检查点能够正确保存、加载，并且评估结果具有一致性。
- 测试标记：使用 `slow`，表示该测试运行时间相对较长。

### `test_sweeps.py`

- 作用：通过命令行测试 Hydra 多运行模式和超参数搜索功能。
- 主要测试：
  - `test_experiments`：运行 `configs/experiment/` 中的全部实验配置。
  - `test_hydra_sweep`：测试多个学习率组合的普通 Hydra sweep。
  - `test_hydra_sweep_ddp_sim`：测试多进程 CPU 模拟 DDP 下的 Hydra sweep。
  - `test_optuna_sweep`：测试 Optuna 超参数搜索。
  - `test_optuna_sweep_ddp_sim_wandb`：测试 Optuna、DDP 模拟和 WandB 日志记录的组合。
- 测试目的：确保实验配置、命令行覆盖、多任务运行、分布式配置和超参数搜索可以正常启动。
- 运行条件：需要 `sh` 包；最后一个测试还需要 `wandb`。缺少依赖时会自动跳过。
- 测试标记：这些测试均使用 `slow`。

### `test_train.py`

- 作用：测试模型训练入口在不同训练模式下是否可以正常工作。
- 主要测试：
  - `test_train_fast_dev_run`：在 CPU 上各执行一个训练、验证和测试批次。
  - `test_train_fast_dev_run_gpu`：在 GPU 上执行快速开发测试。
  - `test_train_epoch_gpu_amp`：测试 GPU 混合精度训练。
  - `test_train_epoch_double_val_loop`：测试每个 epoch 执行两次验证。
  - `test_train_ddp_sim`：使用两个 CPU 进程模拟 DDP 分布式训练。
  - `test_train_resume`：测试保存检查点后继续训练，并验证准确率得到提升。
- 测试目的：覆盖基础训练、GPU、混合精度、验证频率、分布式训练和断点续训等关键训练场景。
- 运行条件：GPU 测试仅在检测到可用 GPU 时运行；部分测试使用 `slow` 标记。

## `helpers` 辅助目录

### `helpers/__init__.py`

- 作用：将 `tests.helpers` 标记为 Python 包。
- 测试目的：本身不执行测试，用于支持辅助模块导入。

### `helpers/package_available.py`

- 作用：检测当前环境、操作系统和可选依赖是否可用。
- 检测内容：
  - TPU
  - Windows 系统
  - `sh`
  - DeepSpeed
  - FairScale
  - WandB
  - Neptune
  - Comet
  - MLflow
- 测试目的：为条件测试提供环境信息，避免在缺少硬件或可选依赖时错误地运行测试。
- 是否直接执行测试：否。

### `helpers/run_if.py`

- 作用：提供 `RunIf` 条件测试装饰器。
- 支持条件：
  - GPU 数量
  - PyTorch 最低或最高版本
  - Python 最低版本
  - Windows 平台
  - TPU
  - `sh`
  - FairScale
  - DeepSpeed
  - WandB、Neptune、Comet 和 MLflow
- 测试目的：只有当运行环境满足测试要求时才执行相应测试，否则通过 `pytest.mark.skipif` 自动跳过。
- 是否直接执行测试：否，它负责控制其他测试是否运行。

### `helpers/run_sh_command.py`

- 作用：使用 `sh` 包启动 Python 命令，并将非零退出状态转换为 `pytest` 测试失败。
- 使用位置：主要由 `test_sweeps.py` 调用。
- 测试目的：验证真实命令行方式启动训练和参数搜索时，程序能否成功完成。
- 是否直接执行测试：否，它是命令执行辅助函数。

## 测试分类总结

| 文件                           | 主要测试目标                         | 是否包含测试用例 |
| ------------------------------ | ------------------------------------ | ---------------- |
| `__init__.py`                  | 将测试目录标记为 Python 包           | 否               |
| `conftest.py`                  | 准备共享训练和评估配置               | 否               |
| `test_configs.py`              | Hydra 配置加载与对象实例化           | 是               |
| `test_datamodules.py`          | MNIST 数据准备和 DataLoader          | 是               |
| `test_eval.py`                 | 训练、检查点保存和独立评估           | 是               |
| `test_sweeps.py`               | 实验配置、Hydra sweep 和 Optuna 搜索 | 是               |
| `test_train.py`                | CPU、GPU、AMP、DDP 和断点续训        | 是               |
| `helpers/__init__.py`          | 将辅助目录标记为 Python 包           | 否               |
| `helpers/package_available.py` | 检测硬件、平台和可选依赖             | 否               |
| `helpers/run_if.py`            | 根据环境条件跳过测试                 | 否               |
| `helpers/run_sh_command.py`    | 在测试中运行 Python 命令             | 否               |

## 常用运行方式

```bash
# 运行全部测试
pytest

# 只运行配置测试
pytest tests/test_configs.py

# 跳过带有 slow 标记的测试
pytest -m "not slow"

# 运行测试并统计 src 目录覆盖率
pytest --cov src
```
