<div align="center">

# Paper1

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Tom 的语音lab

## Installation

项目使用 Python 3.10，支持 Conda 和 uv。必需使用 uv 并且生成对应的 uv.lock 文件，才能保证 github Actions 上的测试通过。

### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# install conda-lock
conda install -c conda-forge conda-lock

# create environment from the locked dependencies
conda-lock install -n Lightning-Template conda-lock.yml

# activate environment
conda activate Lightning-Template
```

修改 `environment.yaml` 中的依赖后，可以重新生成跨平台锁文件：

```bash
conda-lock -f environment.yaml -p linux-64 -p win-64 -p osx-64
```

### uv

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create the virtual environment and install dependencies
uv sync

# activate environment (Linux/macOS)
source .venv/bin/activate

# activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

使用 uv 时，也可以不手动激活环境，直接运行：

```bash
uv run python src/train.py trainer=cpu
```

## Tests and code quality

项目已经配置以下自动化检查：

- pytest：运行 `tests/` 中的配置、数据模块、训练、评估和超参数搜索测试。
- pre-commit：检查并格式化 Python、YAML、Markdown、Shell 和 Notebook 文件，同时执行代码质量、安全及拼写检查。
- GitHub Actions：在推送到 `main` 或创建 Pull Request 时，自动运行跨平台测试、代码覆盖率和 pre-commit 检查。

```bash
# run all tests
pytest

# run tests with coverage
pytest --cov src

# run all pre-commit checks
pre-commit run --all-files
```

使用 uv 时，在以上命令前添加 `uv run`，例如：

```bash
uv run pytest
uv run pre-commit run --all-files
```

## Git commit message format

提交消息建议遵循 Conventional Commits 格式：

```txt
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

- `type`：本次修改的类型，必填。
- `scope`：修改涉及的模块或组件，可选，例如 `data`、`model`、`trainer`、`configs`。
- `subject`：简短说明本次修改，建议使用英文祈使句，不以句号结尾。
- `body`：补充修改原因、实现方式或与旧行为的区别，可选。
- `footer`：关联 Issue 或声明破坏性变更，可选。

### Commit types

| 类型       | 含义                           | 示例                                        |
| ---------- | ------------------------------ | ------------------------------------------- |
| `feat`     | 新增功能                       | `feat(model): add attention module`         |
| `fix`      | 修复 Bug                       | `fix(data): correct MNIST split ratio`      |
| `docs`     | 修改文档                       | `docs(readme): add uv installation guide`   |
| `style`    | 调整格式，不改变代码逻辑       | `style(configs): normalize yaml formatting` |
| `refactor` | 重构代码，不新增功能或修复 Bug | `refactor(train): simplify metric logging`  |
| `perf`     | 性能优化                       | `perf(data): improve dataloader throughput` |
| `test`     | 新增或修改测试                 | `test(model): add checkpoint resume test`   |
| `build`    | 修改构建系统或依赖             | `build(deps): update lightning version`     |
| `ci`       | 修改 CI/CD 配置                | `ci(actions): add Windows test job`         |
| `chore`    | 其他维护性修改                 | `chore: update project metadata`            |
| `revert`   | 撤销之前的提交                 | `revert: remove attention module`           |

### Examples

```txt
feat(model): add CNN classifier

fix(train): restore optimizer state when resuming

test(data): cover multiple batch sizes

docs(readme): document conda-lock workflow

ci(actions): run pytest on pull requests
```

关联 Issue 时，可在 footer 中填写：

```txt
fix(eval): load checkpoint on CPU

Fixes #12
```

存在不兼容修改时，在类型后添加 `!`，并在 footer 中说明：

```txt
feat(configs)!: rename model configuration fields

BREAKING CHANGE: `hidden_size` has been renamed to `hidden_dims`.
```
