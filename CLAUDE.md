# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指导规则。

## 提交规范

1. **每次修改文件后都要提交一次 commit**，方便后续追踪与回滚。
2. **每次有代码新增或者修改，都要补充或更新测试用例**（`tests/` 目录下）。
3. **只有所有测试都通过后，才能提交代码。**

## 环境规范

1. 项目根目录下创建虚拟环境，名称为 `.venv`（已在 `.gitignore` 中忽略）。
2. 所有依赖都安装在 `.venv` 虚拟环境中。
3. Python 版本要求 ≥ 3.11（见 `pyproject.toml`）。

首次初始化：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e . pytest ruff
```

## 测试规范

1. 测试框架使用 pytest，测试目录为 `tests/`（配置见 `pyproject.toml`）。
2. 每次有代码新增或修改，都要补充或更新测试用例。
3. 提交前必须运行测试，全部通过才能提交：

```bash
.venv/Scripts/python -m pytest
```

## 代码规范

1. 使用 ruff 做代码检查和格式化（行宽 100，配置见 `pyproject.toml`）。
2. 提交前必须通过检查：

```bash
.venv/Scripts/ruff check src tests
.venv/Scripts/ruff format src tests
```
