"""模块说明：

- 使用临时源码和已安装工具验证质量门禁的正例、负例及失败退出码。
- unittest 独立运行，不加载论文数据夹具，不下载资源。
"""

# ====================
# 1. Imports
# ====================

import json
import shutil
import subprocess  # nosec B404 - 只运行本地门禁脚本的受控正反例。
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from packaging.requirements import Requirement

from scripts.quality import (
    PROJECT_ROOT,
    main,
    source_files,
    template_errors,
    type_command,
)

# ====================
# 5. Core Logic
# ====================


class GateTests(unittest.TestCase):
    """在临时目录验证工具实际行为，而不是只断言配置文本。"""

    def setUp(self) -> None:
        """建立独立源码目录。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "__init__.py").write_text(
            '"""模块说明：临时测试包。"""\n', encoding="utf-8"
        )

    def write_source(self, content: str) -> Path:
        """写入仅用于本测试的临时源码。"""
        source = self.root / "src" / "sample.py"
        source.write_text(content, encoding="utf-8")
        return source

    def run_gate(self, check: str) -> subprocess.CompletedProcess[str]:
        """捕获真实门禁进程，避免污染测试工作区。"""
        return subprocess.run(  # nosec B603 - 固定脚本、门禁名称和临时目录，不经过命令解释器。
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "quality.py"),
                check,
                "--root",
                str(self.root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_length_boundary(self) -> None:
        """500 物理行通过；包含空行和注释的 501 行失败。"""
        for count in (500, 501):
            with self.subTest(count=count):
                self.write_source(
                    '"""模块说明：边界样例。"""\n' + "# 注释\n\n" * 249 + "\n" * (count - 499)
                )
                result = self.run_gate("length")
                self.assertEqual(
                    result.returncode,
                    0 if count == 500 else 1,
                    result.stdout + result.stderr,
                )
                if count == 501:
                    self.assertIn("too-many-lines", result.stdout)

    def test_reverse_import_and_positive_control(self) -> None:
        """同一规则下合法导入通过，反向导入失败。"""
        for name in ("high", "low"):
            (self.root / "src" / f"{name}.py").write_text("", encoding="utf-8")
        (self.root / ".importlinter").write_text(
            "[importlinter]\nroot_package = src\n"
            "[importlinter:contract:layers]\nname = Layers\ntype = layers\n"
            "layers =\n    src.high\n    src.low\n",
            encoding="utf-8",
        )
        self.assertEqual(self.run_gate("architecture").returncode, 0)
        (self.root / "src" / "low.py").write_text("import src.high\n", encoding="utf-8")
        result = self.run_gate("architecture")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BROKEN", result.stdout)

    def test_missing_package_is_not_silently_skipped(self) -> None:
        """存在命名空间目录时拒绝不完整的依赖扫描。"""
        directory = self.root / "src" / "nested"
        directory.mkdir()
        (directory / "sample.py").write_text("", encoding="utf-8")
        result = self.run_gate("architecture")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("__init__.py", result.stdout)

    def test_project_contracts_reject_framework_and_reverse_math_imports(self) -> None:
        """复用正式配置，拒绝直接/间接框架依赖和数学层反向依赖。"""
        packages = (
            "models",
            "models/modules",
            "models/methods",
            "models/networks",
            "models/components",
            "models/algorithms",
            "losses",
            "data",
            "analysis",
        )
        for package in packages:
            directory = self.root / "src" / package
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("", encoding="utf-8")
        for name in ("train", "eval", "utils"):
            (self.root / "src" / f"{name}.py").write_text("", encoding="utf-8")
        shutil.copyfile(PROJECT_ROOT / ".importlinter", self.root / ".importlinter")
        self.assertEqual(self.run_gate("architecture").returncode, 0)
        cases = (
            ("models/methods", "import lightning\n"),
            ("models/methods", "import src.utils\n"),
            ("models/algorithms", "import src.models.networks\n"),
        )
        (self.root / "src" / "utils.py").write_text("import lightning\n", encoding="utf-8")
        for package, statement in cases:
            with self.subTest(package=package, statement=statement):
                source = self.root / "src" / package / "sample.py"
                source.write_text(statement, encoding="utf-8")
                result = self.run_gate("architecture")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("BROKEN", result.stdout)
                source.unlink()

    def test_types_positive_and_negative(self) -> None:
        """实际类型检查能拒绝公开返回契约错误。"""
        (self.root / "pyrightconfig.json").write_text(
            json.dumps({"include": ["src"], "typeCheckingMode": "basic"}),
            encoding="utf-8",
        )
        self.write_source("def value() -> int:\n    return 1\n")
        self.assertEqual(self.run_gate("types").returncode, 0)
        self.write_source('def value() -> int:\n    return "wrong"\n')
        result = self.run_gate("types")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reportReturnType", result.stdout)

    def test_template_and_section_order(self) -> None:
        """缺模块说明、章节倒序分别报错；函数内部注释不误判。"""
        source = self.write_source('"""模块说明：示例。"""\n# 1. Imports\n# 6. Public API\n')
        self.assertEqual(template_errors(source), [])
        source = self.write_source('"""模块说明：示例。"""\n# 6. Public API\n# 1. Imports\n')
        self.assertTrue(template_errors(source))
        source = self.write_source("value = 1\n")
        self.assertTrue(template_errors(source))

    def test_types_empty_configuration_fails(self) -> None:
        """配置未选中任何源码时不能返回成功。"""
        (self.root / "pyrightconfig.json").write_text(
            json.dumps({"include": ["missing"]}), encoding="utf-8"
        )
        self.assertNotEqual(self.run_gate("types").returncode, 0)

    def test_empty_test_module_fails(self) -> None:
        """只有空测试文件而没有实际测试时必须失败。"""
        directory = self.root / "tests" / "quality"
        directory.mkdir(parents=True)
        (directory / "test_empty.py").write_text("", encoding="utf-8")
        self.assertNotEqual(self.run_gate("selftest").returncode, 0)

    def test_empty_collection_fails(self) -> None:
        """空项目不能被标记为质量检查通过。"""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                source_files(Path(directory))

    def test_environment_quality_requirements_match(self) -> None:
        """环境中的显式质量依赖必须与固定版本清单一致，且可作为包要求解析。"""
        environment = yaml.safe_load((PROJECT_ROOT / "environment.yaml").read_text())
        pip_dependencies = next(
            entry["pip"] for entry in environment["dependencies"] if isinstance(entry, dict)
        )
        parsed = {Requirement(value).name: str(Requirement(value)) for value in pip_dependencies}
        requirements = (PROJECT_ROOT / "requirements-quality.txt").read_text().splitlines()
        for value in requirements:
            if value.strip() and not value.lstrip().startswith("#"):
                requirement = Requirement(value)
                self.assertEqual(parsed.get(requirement.name), str(requirement))

    def test_missing_tool_fails(self) -> None:
        """工具缺失不能自动安装或返回成功。"""
        with patch("scripts.quality.require_module", side_effect=RuntimeError("missing tool")):
            self.assertEqual(main(["length", "--root", str(self.root)]), 1)

    def test_missing_node_fails_without_download(self) -> None:
        """本机无 Node 时失败，不调用 Pyright 的下载包装器。"""
        with patch("scripts.quality.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                type_command(self.root)


# ====================
# 7. Entry Point
# ====================

if __name__ == "__main__":
    unittest.main()
