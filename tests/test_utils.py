"""验证实验辅助工具的配置、日志、异常清理和多进程日志行为。"""

# ====================
# 1. Imports
# ====================

import logging
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from src.utils import instantiators, logging_utils, pylogger, rich_utils, utils

# ====================
# 2. Tests
# ====================


@pytest.mark.parametrize(
    "instantiate", [instantiators.instantiate_callbacks, instantiators.instantiate_loggers]
)
def test_instantiation_empty_and_invalid_configs(instantiate):
    """空配置返回空列表，非配置字典明确报错。"""
    assert instantiate(None) == []
    assert instantiate(OmegaConf.create({})) == []
    with pytest.raises(TypeError, match="DictConfig"):
        instantiate({"invalid": True})


def test_instantiation_creates_real_callbacks_and_loggers(tmp_path):
    """仅实例化带目标的配置，验证真实回调与日志对象。"""
    callbacks = instantiators.instantiate_callbacks(
        OmegaConf.create(
            {
                "ignored": "value",
                "disabled": {},
                "stop": {
                    "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                    "monitor": "val/loss",
                    "patience": 4,
                },
            }
        )
    )
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], EarlyStopping)
    assert callbacks[0].monitor == "val/loss"
    assert callbacks[0].patience == 4
    loggers = instantiators.instantiate_loggers(
        OmegaConf.create(
            {
                "ignored": None,
                "disabled": {},
                "csv": {
                    "_target_": "lightning.pytorch.loggers.CSVLogger",
                    "save_dir": str(tmp_path),
                    "name": "test",
                },
            }
        )
    )
    assert len(loggers) == 1
    assert isinstance(loggers[0], CSVLogger)
    assert loggers[0].name == "test"
    assert loggers[0].save_dir == str(tmp_path)


def test_log_hyperparameters_counts_frozen_and_trainable_parameters():
    """日志包含准确的参数统计和训练配置，并发送给每个记录器。"""
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    loggers = [Mock(), Mock()]
    cfg = OmegaConf.create(
        {
            "model": {"width": 3},
            "data": {"batch_size": 4},
            "trainer": {"max_epochs": 2},
            "seed": 7,
            "tags": ["test"],
        }
    )
    logging_utils.log_hyperparameters(
        {"cfg": cfg, "model": model, "trainer": SimpleNamespace(logger=True, loggers=loggers)}
    )
    for logger in loggers:
        logger.log_hyperparams.assert_called_once()
        params = logger.log_hyperparams.call_args.args[0]
        assert params["model/params/total"] == 8
        assert params["model/params/trainable"] == 6
        assert params["model/params/non_trainable"] == 2
        assert params["data"] == {"batch_size": 4}
        assert params["model"] == {"width": 3}
        assert params["trainer"] == {"max_epochs": 2}
        assert params["seed"] == 7
        assert params["tags"] == ["test"]
        assert params["callbacks"] is None


def test_log_hyperparameters_without_logger():
    """未启用日志时不访问记录器列表。"""
    logging_utils.log_hyperparameters(
        {
            "cfg": OmegaConf.create({}),
            "model": torch.nn.Linear(1, 1),
            "trainer": SimpleNamespace(logger=None),
        }
    )


def test_log_hyperparameters_rejects_non_mapping():
    """主配置不是映射时给出明确错误。"""
    with pytest.raises(TypeError, match="mapping"):
        logging_utils.log_hyperparameters({"cfg": OmegaConf.create([])})


def test_extras_dispatches_only_enabled_options(monkeypatch):
    """辅助选项关闭时不执行，开启时正确传递配置及保存参数。"""
    enforce = Mock()
    print_tree = Mock()
    monkeypatch.setattr(rich_utils, "enforce_tags", enforce)
    monkeypatch.setattr(rich_utils, "print_config_tree", print_tree)
    utils.extras(OmegaConf.create({}))
    utils.extras(OmegaConf.create({"extras": {"print_config": False}}))
    enforce.assert_not_called()
    print_tree.assert_not_called()
    cfg = OmegaConf.create(
        {"extras": {"ignore_warnings": True, "enforce_tags": True, "print_config": True}}
    )
    with warnings.catch_warnings():
        utils.extras(cfg)
        assert warnings.filters[0][0] == "ignore"
    enforce.assert_called_once_with(cfg, save_to_file=True)
    print_tree.assert_called_once_with(cfg, resolve=True, save_to_file=True)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("wandb_state", ["absent", "idle", "running"])
def test_task_wrapper_preserves_results_and_cleans_up(monkeypatch, tmp_path, fails, wandb_state):
    """任务成功或失败均按需关闭活动日志，且保留返回值或原始异常。"""
    wandb = SimpleNamespace(run=wandb_state == "running", finish=Mock())
    monkeypatch.setattr(utils, "find_spec", lambda name: wandb_state != "absent")
    importer = Mock(return_value=wandb)
    monkeypatch.setattr(utils, "import_module", importer)
    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path)}})
    error = ValueError("原始任务失败")
    expected = ({"loss": 1}, {"model": object()})

    def task(cfg):
        assert cfg.paths.output_dir == str(tmp_path)
        if fails:
            raise error
        return expected

    wrapped = utils.task_wrapper(task)
    if fails:
        with pytest.raises(ValueError) as caught:
            wrapped(cfg)
        assert caught.value is error
    else:
        metrics, objects = wrapped(cfg)
        assert metrics is expected[0]
        assert objects is expected[1]
    assert wandb.finish.call_count == int(wandb_state == "running")
    assert importer.call_count == int(wandb_state != "absent")


def test_get_metric_value_handles_missing_and_tensor_values():
    """指标读取支持不指定名称，并拒绝缺失指标。"""
    assert utils.get_metric_value({}, None) is None
    assert utils.get_metric_value({"score": torch.tensor(0.75)}, "score") == 0.75
    with pytest.raises(Exception, match="Metric value not found.*missing"):
        utils.get_metric_value({}, "missing")


def test_print_config_tree_resolves_and_saves(tmp_path, capsys):
    """打印顺序、插值展开和持久化文本保持一致。"""
    cfg = OmegaConf.create(
        {
            "data": {"size": 3},
            "model": {"size": "${data.size}"},
            "tags": ["test"],
            "paths": {"output_dir": str(tmp_path)},
        }
    )
    rich_utils.print_config_tree(
        cfg, print_order=("model", "missing", "data"), resolve=True, save_to_file=True
    )
    output = capsys.readouterr().out
    saved = (tmp_path / "config_tree.log").read_text()
    for text in (output, saved):
        assert "CONFIG" in text
        assert "size: 3" in text
        assert "${data.size}" not in text
        assert text.index("model") < text.index("data")
        assert "tags" in text


@pytest.mark.parametrize("existing", [True, False])
def test_enforce_tags_preserves_or_prompts_and_saves(monkeypatch, tmp_path, existing):
    """已有标签不询问，缺失标签时清理输入并写入配置与文件。"""
    monkeypatch.setattr(
        rich_utils,
        "HydraConfig",
        lambda: SimpleNamespace(cfg=OmegaConf.create({"hydra": {"job": {}}})),
    )
    prompt = Mock(return_value="one, two,,three")
    monkeypatch.setattr(rich_utils.Prompt, "ask", prompt)
    cfg = OmegaConf.create(
        {"paths": {"output_dir": str(tmp_path)}, "tags": ["kept"] if existing else []}
    )
    OmegaConf.set_struct(cfg, True)
    rich_utils.enforce_tags(cfg, save_to_file=True)
    expected = ["kept"] if existing else ["one", "two", "three"]
    assert list(cfg.tags) == expected
    assert prompt.call_count == int(not existing)
    assert all(tag in (tmp_path / "tags.log").read_text() for tag in expected)


def test_enforce_tags_rejects_untagged_multirun(monkeypatch):
    """多实验模式缺少标签时拒绝交互输入。"""
    monkeypatch.setattr(
        rich_utils,
        "HydraConfig",
        lambda: SimpleNamespace(cfg=OmegaConf.create({"hydra": {"job": {"id": 1}}})),
    )
    prompt = Mock()
    monkeypatch.setattr(rich_utils.Prompt, "ask", prompt)
    with pytest.raises(ValueError, match="Specify tags"):
        rich_utils.enforce_tags(OmegaConf.create({"tags": []}))
    prompt.assert_not_called()


@pytest.mark.parametrize(
    ("current_rank", "zero_only", "requested_rank", "emitted"),
    [
        (0, True, None, True),
        (1, True, None, False),
        (1, False, None, True),
        (1, False, 1, True),
        (1, False, 0, False),
    ],
)
def test_ranked_logger_filters_ranks(
    monkeypatch, caplog, current_rank, zero_only, requested_rank, emitted
):
    """按当前进程和指定 rank 过滤日志，并保留 rank 前缀。"""
    monkeypatch.setattr(pylogger.rank_zero_only, "rank", current_rank)
    logger = pylogger.RankedLogger("test.ranks", rank_zero_only=zero_only)
    with caplog.at_level(logging.INFO, logger="test.ranks"):
        logger.log(logging.INFO, "测试消息", rank=requested_rank)
    assert len(caplog.records) == int(emitted)
    if emitted:
        assert f"[rank: {current_rank}]" in caplog.records[0].message
        assert "测试消息" in caplog.records[0].message


def test_ranked_logger_requires_rank_only_for_enabled_level(monkeypatch):
    """仅实际写日志时要求 rank，禁用级别不报错。"""
    monkeypatch.setattr(pylogger.rank_zero_only, "rank", None)
    logger = pylogger.RankedLogger("test.no_rank")
    monkeypatch.setattr(logger.logger, "level", logging.WARNING)
    logger.log(logging.DEBUG, "忽略")
    with pytest.raises(RuntimeError, match="rank.*set"):
        logger.log(logging.ERROR, "失败")
