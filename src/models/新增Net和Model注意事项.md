# 新增 Net 和 Model 注意事项

本项目中：

- **Net**：放在 `src/models/components/`，负责网络结构和前向计算。
- **Model**：放在 `src/models/`，继承 `LightningModule`，负责训练、验证、测试和优化器配置。

## 1. Model 不要依赖具体 Net

Model 只声明自己需要的最小接口：

```python
from typing import Protocol


class ExampleNetwork(Protocol):
    custom_value: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def custom_method(self, x: torch.Tensor) -> torch.Tensor: ...
```

这样可以替换不同的 Net，而不需要修改 Model。

## 2. 初始化时必须保存 Net

构造函数参数仍可声明为 `torch.nn.Module`，内部使用 `cast()`指定业务接口：

```python
from typing import cast


def __init__(self, net: torch.nn.Module, ...) -> None:
    super().__init__()
    self.net: ExampleNetwork = cast(ExampleNetwork, net)
```

不要只写类型标注：

```python
self.net: ExampleNetwork  # 错误：没有保存传入的 net
```

`cast()`只影响静态检查，不转换或验证运行时对象。

## 3. `torch.compile()`后保持接口类型

```python
def setup(self, stage: str) -> None:
    if self.compile_model and stage == "fit":
        self.net = cast(ExampleNetwork, torch.compile(self.net))
```

否则 Pylance 可能把编译结果推断为普通函数，并报告自定义字段或方法不存在。

## 4. 可选的运行时防御检查

`cast()`不会检查对象。如果需要尽早发现错误，可在构造函数边界验证必要成员：

```python
if not hasattr(net, "custom_value"):
    raise TypeError("net must define custom_value")
if not callable(getattr(net, "custom_method", None)):
    raise TypeError("net must define custom_method()")
```

## 5. 同步添加配置和测试

新增模型时至少检查：

- 在 `configs/model/`添加对应 Hydra 配置。
- `_target_`指向正确的 Model 和 Net 类。
- 在 `tests/models/`测试 Net 输出形状和 Model 的 `model_step()`。
- 运行静态检查，确认没有 Pylance 类型错误。
- 运行对应测试，例如 `pytest tests/models/test_example.py`。

## 6. 提交前检查

- Net 的输入、输出形状与数据模块一致。
- Protocol 包含 Model 实际使用的所有自定义字段和方法。
- 优化器能够获取正确的模型参数。
- `self.net`已实际赋值，而不只是添加类型标注。
- 普通模式和 `torch.compile()`模式的类型保持一致。
