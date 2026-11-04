# AGENTS

## 禁止

- 不要自己自动下载任何内容，每次运行下载命令之前都需要询问用户，并且明确说明下载的内容和作用。

## 虚拟环境

不创建新的虚拟环境，使用 `conda activate Lightning-Template` 激活使用现有的虚拟环境。

本机 Conda 路径为 `/mnt/data2/home/tsx/miniconda3/condabin/conda`。不引入 uv；Conda 源缺失的工具在获得下载授权后使用环境内的 pip。

## 语言风格

尽可能简洁，不要说太多不需要的东西。

注释要使用中文。

## 模块结构与开发流程

- 新增或重构模型、数据和实验前，阅读[研究模板开发规范](docs/architecture/%E7%A0%94%E7%A9%B6%E6%A8%A1%E6%9D%BF%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83.md)，按职责与契约确定位置，不能只增加算法目录。
- 新实验接入使用[新增研究实验 Skill](.github/skills/add-research-experiment/SKILL.md)；客户端不支持 Skill 时仍按其中步骤执行。
- 新增 Python 文件不超过 500 物理行，含空行、注释和 docstring。超长存量文件应单独规划拆分或明确例外，不在无关任务中大规模重构，不通过压缩代码绕过限制。
- 下述章节模板省略空章节；保留模块说明、清晰的公开接口以及必要的中文语义说明。
- AI 指令不代替质量门禁。依赖检查、类型检查或长度检查工具缺失时必须报告未验证，不能自动下载/安装或声称检查通过。
- 统一门禁：`python scripts/quality.py all`；门禁自测：`python scripts/quality.py selftest`。全仓失败必须报告，不添加隐式白名单。
- 类型检查需要已有 Node；不在 PATH 时设置 `QUALITY_NODE` 为其可执行文件路径，禁止隐式下载运行时。

## py代码模版

python代码严格参考[Python代码规范](docs/prompt/python_code_style_guide.md.md)。
