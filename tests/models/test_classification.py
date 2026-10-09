"""通用分类 Lightning Module 测试。"""

from functools import partial

import pytest
import torch
from torch import nn

from src.models.methods.discriminative.classification import Classification
from src.models.modules.classification import ClassificationLitModule


def _module(num_classes=3):
    method = Classification(nn.Linear(4, num_classes))
    return ClassificationLitModule(
        method=method,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        num_classes=num_classes,
    )


def test_classification_module_supports_configurable_classes():
    module = _module(num_classes=3)
    inputs = torch.randn(5, 4)
    targets = torch.tensor([0, 1, 2, 1, 0])

    loss, predictions, returned_targets = module.model_step((inputs, targets))

    assert loss.ndim == 0
    assert predictions.shape == targets.shape
    assert torch.equal(returned_targets, targets)
    assert module.train_acc.num_classes == 3


def test_classification_module_rejects_invalid_class_count():
    with pytest.raises(ValueError, match="num_classes"):
        _module(num_classes=1)
