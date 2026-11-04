# `.github` 文件说明

`.github` 目录用于存放 GitHub 仓库相关的协作模板、自动化工作流和服务配置。

## 根目录文件

### `PULL_REQUEST_TEMPLATE.md`

- 作用：Pull Request（PR）描述模板。创建 PR 时，GitHub 会自动填充其中的内容。
- 检查目的：提醒贡献者说明修改内容、关联 Issue 和破坏性变更，并在提交前运行 `pytest` 与 `pre-commit`。
- 是否执行测试：否，它只是提交检查清单，不会自动运行命令。

### `codecov.yml`

- 作用：配置 Codecov 如何判断代码覆盖率检查是否通过。
- 测试目的：检查整个项目以及 PR 新增或修改代码的测试覆盖率变化。
- 当前规则：`project` 和 `patch` 的阈值均设置为 `100%`，表示覆盖率下降达到配置阈值时检查会失败。
- 是否执行测试：否，它只定义覆盖率判定规则；覆盖率数据由 `workflows/test.yml` 生成并上传。

### `release-drafter.yml`

- 作用：Release Drafter 的配置文件，用于自动整理下一版本的发布说明。
- 主要内容：
  - 根据 PR 标签将变更分为功能、Bug 修复、维护和文档。
  - 根据 `major`、`minor`、`patch` 标签确定版本号。
  - 默认按补丁版本递增。
- 是否执行测试：否，与测试无关，主要用于版本发布管理。

## `workflows` 工作流

### `workflows/code-quality-main.yaml`

- 触发条件：代码推送到 `main` 分支时。
- 作用：在 Ubuntu 和 Python 3.10 环境中，对仓库全部文件运行 `pre-commit`。
- 检查目的：检查代码格式、静态质量以及 `.pre-commit-config.yaml` 中配置的其他规则。
- 检查范围：仓库中的所有文件。

### `workflows/code-quality-pr.yaml`

- 触发条件：向 `main`、`release/*` 或 `dev` 分支提交 PR 时。
- 作用：找出 PR 修改的文件，并只对这些文件运行 `pre-commit`。
- 检查目的：在合并前发现格式、代码规范和静态检查问题。
- 检查范围：当前 PR 新增或修改的文件。

### `workflows/release-drafter.yml`

- 触发条件：代码推送到 `main` 分支时。
- 作用：运行 Release Drafter，根据已合并 PR 和 `.github/release-drafter.yml` 自动创建或更新草稿 Release。
- 是否执行测试：否，它只负责生成发布说明和管理发布草稿。

### `workflows/test.yml`

- 触发条件：
  - 代码推送到 `main` 分支。
  - 向 `main`、`release/*` 或 `dev` 分支提交 PR。
- 测试环境：Python 3.10，并分别在 Ubuntu、macOS 和 Windows 上运行。
- 主要测试：执行 `pytest -v`，运行 `tests/` 目录中的测试，验证训练、评估、配置、数据模块和参数搜索等功能。
- 跨平台目的：确保项目在 Linux、macOS 和 Windows 环境下都能正确安装依赖并通过测试。
- 覆盖率任务：在 Ubuntu 上执行 `pytest --cov src`，统计 `src/` 的代码覆盖率并上传到 Codecov。

## 总结

| 文件                               | 用途                               | 是否直接执行测试或检查 |
| ---------------------------------- | ---------------------------------- | ---------------------- |
| `PULL_REQUEST_TEMPLATE.md`         | PR 描述和提交前检查模板            | 否                     |
| `codecov.yml`                      | 代码覆盖率判定规则                 | 否                     |
| `release-drafter.yml`              | 发布说明分类与版本规则             | 否                     |
| `workflows/code-quality-main.yaml` | 检查 `main` 分支全部文件的代码质量 | 是，运行 `pre-commit`  |
| `workflows/code-quality-pr.yaml`   | 检查 PR 修改文件的代码质量         | 是，运行 `pre-commit`  |
| `workflows/release-drafter.yml`    | 自动更新草稿 Release               | 否                     |
| `workflows/test.yml`               | 跨平台运行单元测试和覆盖率统计     | 是，运行 `pytest`      |
