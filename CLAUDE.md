# CLAUDE.md

本文件为 Claude Code 在本仓库工作时的指导规则。

## 提交规范

1. **每次修改文件后都要提交一次 commit**，方便后续追踪与回滚。
2. **每次有代码新增或者修改，都要补充或更新测试用例**（`tests/` 目录下）。
3. **只有所有测试都通过后，才能提交代码。**

## 环境规范

1. 项目根目录下创建虚拟环境，名称为 `.venv`（已在 `.gitignore` 中忽略）。
2. 所有依赖都安装在 `.venv` 虚拟环境中。
3. Python 版本要求 ≥ 3.12（见 `pyproject.toml`，OpenHands SDK 的下限）。

首次初始化：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[sandbox]" --group dev
```

国内网络访问 `files.pythonhosted.org` 会超时，安装依赖时需指定镜像源（只在命令行传参，不改全局配置）：

```bash
.venv/Scripts/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 <包名>
```

## 测试规范

1. 测试框架使用 pytest，测试目录为 `tests/`（配置见 `pyproject.toml`）。
2. 每次有代码新增或修改，都要补充或更新测试用例。
3. 提交前必须运行测试，全部通过才能提交：

```bash
.venv/Scripts/python -m pytest
```

4. `filterwarnings = ["error"]` 已开启：任何 pytest 警告都会导致失败。领域模型若以 `Test` 开头（如 `TestPlan`、`TestRun`），必须在类内声明 `__test__ = False`，否则会被误当作测试类收集。

## 代码规范

提交前四项门禁必须全绿：

```bash
.venv/Scripts/python -m pytest          # 测试
.venv/Scripts/python -m mypy            # 类型检查（strict + pydantic 插件）
.venv/Scripts/ruff check src tests      # Lint
.venv/Scripts/ruff format --check src tests   # 格式
```

1. **ruff**：行宽 100。`[tool.ruff.lint]` 使用显式 `select` 而非 `extend-select` —— ruff 0.17 把默认规则集从约 30 条扩到 400+，继承默认集会让一次 ruff 升级凭空引入几十条新错误、把提交门禁打红。新增规则要显式加进 `select`。
2. **mypy**：`strict = true`，并启用 `pydantic.mypy` 插件。领域层全是 pydantic 模型，没有插件 mypy 就看不出「给 frozen 模型赋值」是错误、也不会按字段校验 `__init__` 签名。
3. **类型导出**：包内声明类型的模块必须带 `py.typed`，否则外部调用方拿不到类型信息。
4. 反序列化返回 `Self` 而非基类，这样 `Requirement.from_json(...)` 对类型检查器仍是 `Requirement`，子类字段不会丢失。

## Harness 规范

平台的 agent（PM / 架构 / 开发 / QA）必须遵循 [`docs/HARNESS_SPEC.md`](docs/HARNESS_SPEC.md)。开发或修改任一角色 agent 前，先过一遍该文档 §9 的 HARNESS_CHECKLIST。

