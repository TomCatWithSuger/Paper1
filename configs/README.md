# `configs` 配置目录说明

本目录使用 **Hydra 1.3 + OmegaConf** 管理训练和评估配置。

配置文件的作用不是直接执行代码，而是：

1. Hydra 从入口配置开始，组合多个 YAML 文件；
2. 组合结果作为 `DictConfig` 传入 `src/train.py` 或 `src/eval.py`；
3. 代码读取配置值，并根据 `_target_` 实例化数据模块、模型、回调、日志器和 Trainer。

______________________________________________________________________

## 1. 目录结构

| 路径               | 配置内容                                       | 典型用法                                |
| ------------------ | ---------------------------------------------- | --------------------------------------- |
| `train.yaml`       | 训练入口配置，决定默认加载哪些配置组           | `python src/train.py`                   |
| `eval.yaml`        | 评估入口配置，要求提供 checkpoint              | `python src/eval.py ckpt_path=...`      |
| `data/`            | 数据集、DataModule、数据预处理参数             | `data=vctk`                             |
| `data/vctk_split/` | VCTK 数据划分策略                              | `data/vctk_split=ratio`                 |
| `model/`           | LightningModule、网络、优化器和调度器          | `model=conditional_mean_flow`           |
| `trainer/`         | Lightning Trainer、设备、精度和训练轮数        | `trainer=gpu`                           |
| `callbacks/`       | checkpoint、早停、模型摘要、进度条             | `callbacks=default`                     |
| `logger/`          | TensorBoard、WandB、CSV 等日志器               | `logger=tensorboard`                    |
| `experiment/`      | 一组完整实验参数，用于固定数据、模型和训练设置 | `experiment=conditional_mean_flow_vctk` |
| `debug/`           | 快速调试、过拟合测试、异常检测、性能分析       | `debug=limit`                           |
| `paths/`           | 数据、预训练权重、日志和输出目录               | 默认加载 `paths=default`                |
| `extras/`          | 打印配置、标签检查、warning 控制               | 默认加载 `extras=default`               |
| `hydra/`           | Hydra 自身的输出目录和日志行为                 | 默认加载 `hydra=default`                |
| `hparams_search/`  | Optuna 超参数搜索空间与搜索策略                | `hparams_search=mnist_optuna`           |
| `local/`           | 机器或用户私有配置，可选且通常不提交           | `local=default`                         |

`configs/__init__.py` 只用于打包时包含配置目录，不负责加载 YAML。

______________________________________________________________________

## 2. 配置是如何被读取的

训练入口位于 `src/train.py`：

```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    ...
```

这段装饰器确定了两个关键点：

- `config_path="../configs"`：配置搜索目录是项目的 `configs/`；
- `config_name="train.yaml"`：训练从 `configs/train.yaml` 开始组合配置。

评估入口同理，但从 `configs/eval.yaml` 开始：

```python
@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    ...
```

因此，**不是放进 `configs/` 的每个 YAML 都会被自动读取**。一个配置需要满足以下至少一种情况：

1. 被入口配置的 `defaults` 列表选中；
2. 被其他已加载配置的 `defaults` 列表引用；
3. 在命令行中被明确选择或覆盖。

______________________________________________________________________

## 3. `defaults` 如何组合配置

`train.yaml` 中的核心内容如下：

```yaml
defaults:
  - _self_
  - data: mnist
  - model: mnist
  - callbacks: default
  - logger: null
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - experiment: null
  - hparams_search: null
  - optional local: default
  - debug: null
```

它表示默认组合：

```text
configs/train.yaml
├── configs/data/mnist.yaml
├── configs/model/mnist.yaml
├── configs/callbacks/default.yaml
├── configs/trainer/default.yaml
├── configs/paths/default.yaml
├── configs/extras/default.yaml
└── configs/hydra/default.yaml
```

其中：

- `data: mnist` 对应 `configs/data/mnist.yaml`；
- `model: mnist` 对应 `configs/model/mnist.yaml`；
- `callbacks: default` 对应 `configs/callbacks/default.yaml`；
- `logger: null` 表示默认不加载日志器；
- `experiment: null` 表示默认不加载实验配置，需要命令行选择；
- `optional local: default` 表示尝试读取 `configs/local/default.yaml`，文件不存在也不报错；
- `_self_` 表示当前 YAML 自身内容在组合顺序中的位置；
- 配置发生冲突时，通常由组合顺序靠后的值覆盖靠前的值。

可以先查看最终组合结果，而不开始训练：

```bash
python src/train.py --cfg job
```

查看 Hydra 自身配置和任务配置：

```bash
python src/train.py --cfg all
```

______________________________________________________________________

## 4. YAML 如何变成 Python 对象

### 4.1 `_target_` 指定需要实例化的类

例如 `data/mnist.yaml`：

```yaml
_target_: src.data.mnist_datamodule.MNISTDataModule
data_dir: ${paths.data_dir}
batch_size: 128
num_workers: 0
pin_memory: false
```

训练代码执行：

```python
datamodule = hydra.utils.instantiate(cfg.data)
```

它近似等价于：

```python
from src.data.mnist_datamodule import MNISTDataModule

datamodule = MNISTDataModule(
    data_dir=cfg.paths.data_dir,
    batch_size=128,
    num_workers=0,
    pin_memory=False,
)
```

同样的机制用于：

```python
model = hydra.utils.instantiate(cfg.model)
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
```

所以配置中的键必须与目标类构造函数能够接收的参数对应。

### 4.2 `_partial_: true` 表示暂不立即构造

模型配置中的优化器通常写成：

```yaml
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
```

Hydra 此时生成的是一个等待补充参数的构造函数。模型稍后传入需要优化的参数，例如：

```python
optimizer(params=self.parameters())
```

### 4.3 回调和日志器是逐项实例化的

`callbacks/default.yaml` 会继续加载：

```yaml
defaults:
  - model_checkpoint
  - early_stopping
  - model_summary
  - rich_progress_bar
  - _self_
```

随后 `instantiate_callbacks()` 遍历 `cfg.callbacks`，实例化所有包含 `_target_` 的项目。

`logger/` 的处理方式相同。`logger=many_loggers` 可以一次组合多个日志器。

______________________________________________________________________

## 5. 配置引用与变量插值

Hydra/OmegaConf 使用 `${...}` 引用其他配置值。

例如：

```yaml
data_dir: ${paths.data_dir}
default_root_dir: ${paths.output_dir}
```

`paths/default.yaml` 中还使用环境变量：

```yaml
root_dir: ${oc.env:PROJECT_ROOT}
vctk_data_dir: ${oc.env:VCTK_DATA_DIR,${paths.root_dir}/data/VCTK}
```

含义是：

- `PROJECT_ROOT` 由 `rootutils.setup_root()` 在程序启动时设置；
- 如果设置了 `VCTK_DATA_DIR`，优先使用环境变量；
- 否则使用项目内的 `data/VCTK`。

Hydra 的本次运行输出目录通过以下配置生成：

```yaml
output_dir: ${hydra:runtime.output_dir}
```

实际目录规则定义在 `hydra/default.yaml`，例如：

```text
logs/train/runs/年-月-日_时-分-秒/
```

______________________________________________________________________

## 6. 常用运行方式

### 使用默认 MNIST 配置

```bash
python src/train.py
```

### 切换数据、模型和 Trainer

```bash
python src/train.py data=vctk model=conditional_mean_flow trainer=gpu
```

### 使用完整实验配置

```bash
python src/train.py experiment=conditional_mean_flow_vctk
```

实验配置会同时覆盖多个配置组。例如：

```yaml
# configs/experiment/conditional_mean_flow_vctk.yaml
defaults:
  - override /data: vctk
  - override /data/vctk_split: unseen_speaker_sentence
  - override /model: conditional_mean_flow
  - override /callbacks: default
  - override /trainer: default
```

这里的 `override /model: conditional_mean_flow` 表示替换入口配置原本选择的 `model: mnist`。

### 覆盖单个参数

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=32
```

### 启用日志器

```bash
python src/train.py logger=tensorboard
```

### 启用调试配置

```bash
python src/train.py debug=limit
```

### 运行评估

```bash
python src/eval.py \
  data=vctk \
  model=conditional_mean_flow \
  ckpt_path=/path/to/model.ckpt
```

`eval.yaml` 中的 `ckpt_path: ???` 表示这是必填值，未提供时 Hydra 会报错。

### 超参数搜索

```bash
python src/train.py -m hparams_search=mnist_optuna experiment=example
```

`-m` 表示 multirun。Optuna sweeper 会按照 `hparams_search/mnist_optuna.yaml` 中的搜索空间多次运行训练。

______________________________________________________________________

## 7. 如何新增一个配置

### 7.1 新增模型配置

创建：

```text
configs/model/my_model.yaml
```

内容示例：

```yaml
_target_: src.models.my_model_module.MyModelLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.0001

net:
  _target_: src.models.components.my_network.MyNetwork
  hidden_size: 256

compile: false
```

使用：

```bash
python src/train.py model=my_model
```

能够被读取的原因是：

1. `train.yaml` 已声明 `model` 配置组；
2. `model=my_model` 会让 Hydra 在 `configs/model/` 中查找 `my_model.yaml`；
3. 组合后内容位于 `cfg.model`；
4. `train.py` 调用 `hydra.utils.instantiate(cfg.model)`。

### 7.2 新增完整实验配置

创建：

```text
configs/experiment/my_experiment.yaml
```

内容示例：

```yaml
# @package _global_

defaults:
  - override /data: vctk
  - override /model: conditional_mean_flow
  - override /trainer: gpu

tags: ["my-experiment"]
seed: 12345

data:
  batch_size: 16

trainer:
  max_epochs: 100

model:
  optimizer:
    lr: 0.0001
```

使用：

```bash
python src/train.py experiment=my_experiment
```

`# @package _global_` 使配置内容合并到全局根节点，因此其中的 `data`、`model`、`trainer` 会覆盖最终的同名配置。

### 7.3 新增配置后如何检查

推荐先运行：

```bash
python src/train.py model=my_model --cfg job --resolve
```

重点检查：

- 是否成功找到配置文件；
- 最终的 `_target_` 是否正确；
- `${...}` 是否成功解析；
- 参数是否出现在预期的 `data`、`model` 或 `trainer` 节点下。

需要注意：`--cfg job` 只验证配置组合和插值，不一定能发现目标类构造参数错误。实际实例化错误仍需通过正常运行或测试发现。

______________________________________________________________________

## 8. “写了配置就一定会生效”需要满足什么条件

不能只靠“文件位于 `configs/`”保证生效，需要同时满足：

1. **入口正确**：运行的是带有 `@hydra.main` 的 `src/train.py` 或 `src/eval.py`；
2. **配置被选中**：它出现在 `defaults` 中，或通过命令行显式选择；
3. **组合位置正确**：配置进入代码实际读取的节点，例如 `cfg.model`；
4. **键名正确**：参数名与 Python 类构造函数或代码读取的字段一致；
5. **没有被后续覆盖**：检查 `defaults` 顺序、实验配置和命令行覆盖；
6. **代码确实使用该字段**：普通 YAML 字段只有在代码读取它，或 `instantiate()` 将它传给 `_target_` 时才产生作用。

最直接的检查方式是：

```bash
python src/train.py <配置选择> --cfg job --resolve
```

最终打印出来的配置，才是代码运行时收到的 `cfg`。正常训练时，项目还会通过 `extras.print_config` 打印配置树，并在本次运行目录中保存 `config_tree.log`，便于确认实际生效值。
