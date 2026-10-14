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

## 评估集（Evals）

按 HARNESS_SPEC.md §5.2，**每个角色 agent 上线前必须有 golden task 评估集**，改动 prompt 或流程后必须跑一遍防回退。

1. 评估用例定义在 `src/sdlc_agent/evals/golden/*.yaml`（每角色一个文件，3-5 条）。它们是数据而非代码，调期望值不需要动 runner。
2. **只用确定性检查**：schema 合法性、artifact kind、字段非空、正则命中/不命中、token 上限、工具集白名单。LLM-as-judge 只在确定性检查无法表达需求时才加。
3. **评估集为空不算通过**：`EvalReport.is_green` 在 0 个任务时返回 `False`。"没跑所以没失败"和 QA 领域模型拒绝的假绿是同一类错误。
4. 新增检查类型时同步更新 `CHECK_TYPES`，否则加载器报错信息会不准。
5. **真实模型评测走录制/回放**（`tests/test_llm_golden.py`）：vcrpy 录一次 HTTP，之后离线回放、不花钱。有 cassette 时回放（自动注入哑 key，不会发出去）；无 cassette 且设了 `ANTHROPIC_API_KEY` 时录制（**真实扣费**，12 个任务各一次调用）；两者都没有则跳过——跳过是诚实的。改了 prompt、golden task 或模型配置后重新录制：

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/Scripts/python -m pytest tests/test_llm_golden.py --record-mode=all
```

6. **并非所有 golden task 都能由角色 agent 跑**：`qa_reports_not_run_honestly` 期望 `test_run`，那是测试执行阶段（executor）的产物，不是 QA 角色的。`exclusion_reason()` 会给出原因，`tests/test_role_runner.py` 锁定当前唯一被排除的就是它——sandbox executor 落地后这个断言要随之退役。

```bash
.venv/Scripts/python -c "from sdlc_agent.evals import load_all_tasks, summarize_coverage; import json; print(json.dumps(summarize_coverage(load_all_tasks()), indent=1))"
```

`summarize_coverage` 会列出每个角色**未被覆盖的评估维度**——覆盖有缺口不是错，但必须是知情的决定而不是意外。

## 配置

`config.example.yaml` 是范例（已被测试覆盖，改坏会红）。复制为 `config.yaml` 后使用。

**配置文件会进 git，所以只能写环境变量的名字，绝不能写密钥本身**（HARNESS_SPEC.md §4.1）。凭据在运行时从环境读取，加载配置本身不需要任何密钥，因此 CI 可以校验配置。

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

注意 `temperature` 默认是 `None` 而不是数字：当代 Claude 模型对任何非默认采样值都会返回 400。质量与成本的调节杆是 `effort`。

## 数据库

人工门禁依赖 checkpoint 才能跨进程存活，生产用 Postgres（`postgres_checkpointer`，见 `src/sdlc_agent/pipeline/postgres.py`）。连接串是凭据，走环境变量 `SDLC_DATABASE_URL`，不进配置文件。

测试默认不需要数据库。要跑「门禁跨重启存活」的集成测试（`tests/pipeline/test_postgres.py`），提供一个可写入的 Postgres：

```bash
export SDLC_TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/sdlc_test
```

未设置时该测试跳过——跳过是诚实的；对着假的数据库跑绿不是。


