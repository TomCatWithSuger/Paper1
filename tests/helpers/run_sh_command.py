"""模块说明：在测试中执行并校验外部命令的退出状态。"""

from importlib import import_module
from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE


def run_sh_command(command: List[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    if not _SH_AVAILABLE:
        pytest.skip("可选依赖 sh 未安装或当前平台不支持")
    sh = import_module("sh")
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
