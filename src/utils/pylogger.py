import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """支持多进程 rank 标识的命令行日志记录器。"""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """初始化支持多进程输出的日志记录器。

        :param name: 日志记录器名称。
        :param rank_zero_only: 是否只允许 rank 0 进程输出日志。
        :param extra: 提供额外上下文信息的映射。
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """为消息添加当前进程 rank，并按指定 rank 输出日志。

        :param level: 日志级别。
        :param msg: 日志消息。
        :param rank: 允许输出日志的进程 rank；未指定时允许所有进程。
        :param args: 传递给底层日志函数的位置参数。
        :param kwargs: 传递给底层日志函数的关键字参数。
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
