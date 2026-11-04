"""模块说明：

- 在当前 Python 环境执行依赖、物理行数、类型与文件说明检查。
- 对外提供 main 命令；不安装工具，不下载运行时，不修改被检查源码。
"""

# ====================
# 1. Imports
# ====================

import argparse
import ast
import importlib.util
import json
import os
import shutil
import subprocess  # nosec B404 - 仅执行已定位的本地工具，不使用命令解释器。
import sys
import tokenize
from collections.abc import Sequence
from io import StringIO, TextIOWrapper
from pathlib import Path

# ====================
# 2. Definitions
# ====================

SOURCE_ROOTS = ("src", "tests", "scripts", "configs")
CHECKS = ("architecture", "length", "types", "templates", "selftest")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ====================
# 3. Utilities
# ====================


def source_files(root: Path) -> list[Path]:
    """列出项目源码和测试，不扫描数据、权重或外部虚拟环境。"""
    sources = sorted(
        source
        for directory in SOURCE_ROOTS
        for source in (root / directory).rglob("*.py")
        if "__pycache__" not in source.parts
    )
    if not sources:
        raise ValueError("未收集到 Python 文件")
    return sources


def run_command(command: Sequence[str], root: Path) -> int:
    """执行已安装工具，保留其退出码和诊断。"""
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(  # nosec B603 - 参数数组由内部工具入口构造，不经过命令解释器。
        list(command),
        cwd=root,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode


def require_module(name: str) -> Path:
    """确认工具已安装；缺失时失败，不自动安装。"""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"缺少质量工具：{name}，请按开发规范安装")
    return Path(spec.origin).parent


def type_command(root: Path) -> list[str]:
    """直接调用打包的 Pyright 与已有 Node，阻止包装器隐式下载。"""
    package = require_module("pyright")
    node = shutil.which(os.environ.get("QUALITY_NODE", "node"))
    if node is None:
        raise RuntimeError("缺少 Node；请设置 QUALITY_NODE 为已有 Node 路径")
    entry = package / "dist" / "index.js"
    if not entry.is_file():
        raise RuntimeError("Pyright 安装不完整：缺少 dist/index.js")
    config = root / "pyrightconfig.json"
    if not config.is_file():
        raise RuntimeError("缺少 pyrightconfig.json")
    return [node, str(entry), "--project", str(config), "--pythonpath", sys.executable]


def template_errors(source: Path) -> list[str]:
    """检查模块说明与已有编号章节顺序；不推断中文说明的语义。"""
    with tokenize.open(source) as stream:
        content = stream.read()
    module = ast.parse(content, filename=str(source))
    errors: list[str] = []
    if "模块说明" not in (ast.get_docstring(module) or ""):
        errors.append("缺少中文模块说明")
    sections: list[int] = []
    for token in tokenize.generate_tokens(StringIO(content).readline):
        if token.type != tokenize.COMMENT or token.start[1] != 0:
            continue
        heading = token.string.lstrip("# ")
        number, separator, _ = heading.partition(". ")
        if separator and number.isdigit():
            sections.append(int(number))
    if sections != sorted(set(sections)) or any(number not in range(1, 8) for number in sections):
        errors.append("顶层章节编号必须在 1 到 7 之间且按序、不重复")
    return errors


# ====================
# 4. Internal Functions
# ====================


def _architecture(root: Path) -> int:
    require_module("importlinter")
    sources = source_files(root)
    missing: set[Path] = set()
    for source in sources:
        if not source.is_relative_to(root / "src"):
            continue
        parent = source.parent
        while parent != root:
            if not (parent / "__init__.py").is_file():
                missing.add(parent)
            parent = parent.parent
    if missing:
        for directory in sorted(missing):
            print(f"{directory.relative_to(root)}: 缺少 __init__.py，依赖图可能漏扫")
        return 1
    config = root / ".importlinter"
    if not config.is_file():
        raise RuntimeError("缺少 .importlinter")
    return run_command(
        [
            sys.executable,
            "-c",
            "from importlinter.cli import lint_imports_command; " "lint_imports_command()",
            "--config",
            str(config),
            "--no-cache",
        ],
        root,
    )


def _length(root: Path) -> int:
    require_module("pylint")
    sources = source_files(root)
    return run_command(
        [
            sys.executable,
            "-m",
            "pylint",
            "--disable=all",
            "--enable=too-many-lines",
            "--max-module-lines=500",
            "--score=no",
            "--persistent=no",
            *[str(source.relative_to(root)) for source in sources],
        ],
        root,
    )


def _templates(root: Path) -> int:
    failed = False
    for source in source_files(root):
        for message in template_errors(source):
            print(f"{source.relative_to(root)}: {message}")
            failed = True
    return int(failed)


def _types(root: Path) -> int:
    source_files(root)
    result = subprocess.run(  # nosec B603 - 仅使用已定位的运行时与本地类型检查入口。
        [*type_command(root), "--outputjson"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    report = json.loads(result.stdout)
    summary = report["summary"]
    if summary["filesAnalyzed"] == 0:
        raise RuntimeError("Pyright 未分析任何文件")
    for diagnostic in report["generalDiagnostics"]:
        line = diagnostic.get("range", {}).get("start", {}).get("line", 0) + 1
        rule = diagnostic.get("rule", diagnostic["severity"])
        print(f'{diagnostic["file"]}:{line}: {diagnostic["message"]} ({rule})')
    print(
        f'分析 {summary["filesAnalyzed"]} 个文件，'
        f'{summary["errorCount"]} 个错误，{summary["warningCount"]} 个警告'
    )
    return result.returncode or int(summary["errorCount"] > 0)


def _selftest(root: Path) -> int:
    directory = root / "tests" / "quality"
    if not list(directory.glob("test_*.py")):
        raise RuntimeError("未收集到门禁自测")
    return run_command(
        [
            sys.executable,
            "-c",
            "import sys, unittest; "
            "suite = unittest.defaultTestLoader.discover(sys.argv[1]); "
            "assert suite.countTestCases() > 0, '未收集到门禁自测'; "
            "result = unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(not result.wasSuccessful())",
            str(directory),
        ],
        root,
    )


# ====================
# 6. Public API
# ====================


def main(argv: Sequence[str] | None = None) -> int:
    """执行指定门禁；全量运行收集所有失败，但不将失败降级为警告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("all", *CHECKS), default="all", nargs="?")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    checks = CHECKS if args.check == "all" else (args.check,)
    failed: list[str] = []
    for check in checks:
        print(f"\n=== {check} ===", flush=True)
        try:
            action = {
                "architecture": _architecture,
                "length": _length,
                "types": _types,
                "templates": _templates,
                "selftest": _selftest,
            }[check]
            result = action(root)
        except (
            OSError,
            RuntimeError,
            ValueError,
            SyntaxError,
            tokenize.TokenError,
            KeyError,
            TypeError,
        ) as error:
            print(f"{check}: {error}", file=sys.stderr)
            result = 1
        if result:
            failed.append(check)
    if failed:
        print("失败门禁：" + ", ".join(failed), file=sys.stderr)
    return int(bool(failed))


# ====================
# 7. Entry Point
# ====================

if __name__ == "__main__":
    # 门禁含中文诊断；Windows 终端与重定向管道均使用稳定的 UTF-8 输出。
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
