from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """整理并记录 Lightning 实验的超参数。

    除主要配置外，还会记录模型参数数量。

    :param object_dict: 包含主配置、Lightning 模型和 Trainer 的对象字典。
    """
    hparams: Dict[str, Any] = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    if not isinstance(cfg, dict):
        raise TypeError("Expected the main config to be a mapping")

    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # 统计模型总参数、可训练参数和冻结参数。
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # 将超参数发送到所有已启用的日志记录器。
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
