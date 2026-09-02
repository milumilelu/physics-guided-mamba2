# AGENTS.md

## 工作流约定

- 每次阶段任务完成后，自动提交 git 并推送到 GitHub（origin）。
  - 只暂存与本阶段相关的文件；无关的未跟踪文件保持原样。
  - 提交信息使用中文，风格与仓库历史一致（概要 + 冒号 + 要点）。
  - 提交前检查 `git status` 与 `git diff`，不提交密钥或临时文件。

## 环境与验证

- Python 固定使用仓库根目录 `.venv`（3.12.13）。
- 提交前如改动了代码，先跑 `.\.venv\Scripts\python.exe -m unittest discover -s tests`。
