import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """在任务启动前执行可选的辅助操作。

    包括忽略 Python 警告、设置实验标签和打印配置树。

    :param cfg: Hydra 配置树。
    """
    # 未配置 extras 时直接返回。
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # 按配置关闭 Python 警告。
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # 缺少标签时要求用户补充实验标签。
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # 使用 Rich 打印并保存配置树。
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """统一处理任务执行、异常记录和日志资源清理。

    使用示例：
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: 需要包装的任务函数。
    :return: 包装后的任务函数。
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 执行任务并收集指标和运行对象。
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # 记录异常后继续向上抛出。
        except Exception as ex:
            log.exception("")
            raise ex

        # 无论任务是否成功都执行清理操作。
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # 关闭未结束的 W&B 运行，避免影响后续任务。
            if find_spec("wandb"):
                wandb = import_module("wandb")

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """从 LightningModule 的日志指标中读取指定值。

    :param metric_dict: 指标名称与指标值的映射。
    :param metric_name: 需要读取的指标名称。
    :return: 指标对应的浮点值；未指定名称时返回 ``None``。
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
