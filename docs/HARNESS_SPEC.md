# Harness Engineering 开发规范

> 本规范约束 **平台自身的 agent**（PM / 架构 / 开发 / QA 四个角色）的设计与实现。
> 基准：OpenAI / Anthropic / Martin Fowler 的社区共识 + `awesome-harness-engineering` 的 HARNESS_CHECKLIST。
> 确认日期：2026-10-13

## 0. 为什么需要这份规范

核心公式：**智能体 = 大模型 + Harness**。

Harness 是围绕模型的一切脚手架：循环控制、工具接口、上下文管理、权限沙箱、验证闭环。
证据表明它是**主要性能杠杆**，而非模型本身：

- 同一模型经过不同 harness，基准得分 **68% vs 88%**（2026 行业审计，246 个开源 harness）
- LangChain 仅改 harness（不换模型），编码 agent 在 Terminal Bench 2.0 从**第 30 名升至前 5**
- LangChain 仅调 harness，Nemotron 3 Ultra 逼近 Opus 4.8，成本降至 **1/10**（$4.48 vs $43.48）
- 缩小工具空间（statewright）让本地模型在 SWE-bench 子集从 **2/10 升至 10/10**

同时，**60% 的 harness 没有任何测试或评估就上线** —— 这是本规范强制 Evals 的原因。

## 1. Harness 四要素与技术栈映射

学术构成性定义（arXiv 2606.10106）：一个 harness 必须具备且仅需四个要素。

| 要素 | 职责 | 本项目实现 |
|---|---|---|
| **Agent Loop** | 观察→规划→行动→验证的循环 | LangGraph `StateGraph`，节点=角色，边=流转 |
| **Tool Interface** | 模型与外部世界的唯一接触面 | OpenHands SDK `tool/` + `workspace/`；自建工具遵循 §3 |
| **Context Management** | 有限上下文的选择性供给与压缩 | OpenHands `event/condenser.py`；工件文件化（§6） |
| **Control Mechanisms** | 终止、门禁、预算、中断 | LangGraph `interrupt()` + checkpointer + 任务级预算（§5.3） |

**判定规则**：新增任何组件前，先说明它属于哪个要素。不属于任何一个的，大概率是过度设计。

## 2. Agent 循环与控制

### 2.1 Handoff 上限：硬约束 ⚠️

**Microsoft 实证：超过 4 次 agent 间 handoff 几乎必然失败。**

我们的流水线 `PM → 架构 → 开发 → QA` **恰好 4 次**，已在上限。因此：

1. **禁止**在此基础上增加新的角色 handoff（如"需求分析 agent"、"文档 agent"）
2. 新增能力一律实现为**现有角色内部的工具或子步骤**，而非新角色
3. 每次 handoff 必须传递**强类型 pydantic 工件**，禁止自由文本传递（GitHub 实证：多 agent 系统即分布式系统，handoff 需要 typed schema + 边界校验）
4. QA 失败回流开发节点时，**必须带重试上限**（默认 3 次），超限则 `interrupt()` 交人工

### 2.2 确定性优先于自主性

- 流程走向由 **LangGraph 条件边（代码）** 决定，不由 LLM 自由判断
- LLM 只负责**产出工件内容**，不负责**决定下一步去哪**
- 每个循环必须有**确定性终止条件**（成功 / 重试超限 / 预算耗尽），禁止依赖模型"自觉结束"

### 2.3 卡死检测

复用 OpenHands `conversation/stuck_detector.py`。检测 agent 原地打转（重复相同工具调用、无进展）并强制打断，转入 `interrupt()`。**多 agent 系统最常见的失败模式就是静默打转烧钱。**

### 2.4 门禁必须是独立节点（LangGraph 实测约束）⚠️

**LangGraph 在 resume 时会从头重新执行整个节点函数**，`interrupt()` 之前的代码会再跑一次。已实测确认：

```
第一次 invoke 后的调用次数 : 1
resume 之后的调用次数      : 2
```

因此**调用 LLM 的节点与 `interrupt()` 门禁必须拆成两个节点**。若合在一个节点内，每次人工评审都会让 LLM 调用计费两次——这直接违反 §5.3 的预算护栏，且因为产出被 checkpointer 覆盖而不会表现为错误，只会表现为账单翻倍。

正确拓扑：

```
[role_node: 调 LLM，产出工件]  →  [gate_node: 仅 interrupt + 记录决定]  →  下一环节
```

推论：任何**有副作用**的操作（写文件、推分支、开 PR）都必须放在 `interrupt()` **之后**的节点，否则 resume 会重复执行该副作用。


## 3. 工具设计（Agent UX）

Anthropic《Writing Effective Tools for Agents》：**工具设计就是 agent 的用户体验**。

### 3.1 强制规则

| 规则 | 说明 |
|---|---|
| **单一职责** | 一个工具只做一件概念上的事 |
| **最小 schema** | 不放 agent 不会用的可选字段 |
| **错误信息指向下一步** | 不能只说"失败了"，必须说"接下来该做什么" |
| **返回结构一致** | 成功与失败返回同一形状 |
| **命名无歧义** | 名字要让模型从字面就能正确选用 |

### 3.2 优先文件系统原语，而非定制工具

Microsoft Azure SRE Agent 实证：从 **100+ 定制工具 + 规定式 prompt** 改为把源码/runbook/查询 schema/历史笔记**全部暴露为文件**，让 agent 用 `read_file` / `grep` / `find` / `shell` —— **"Intent Met" 得分从 45% 升至 75%**。

因此：
- 默认给 agent 通用的 `read_file` / `grep` / `glob` / `shell`（沙箱内）
- 只在通用原语**确实无法表达**时才建定制工具
- 新增定制工具前必须回答："为什么不能让 agent 读文件/跑脚本解决？"

### 3.3 用 schema 约束行为，而非运行时权限检查

OpenDev 实证：**schema-filtered planning subagent** —— 通过工具 schema 本身限定 agent 在某阶段能调什么，比运行时权限检查更可靠。

配合 statewright 结论（缩小工具空间 → 本地模型 2/10 变 10/10）：
**每个角色 agent 只暴露其职责所需的工具子集。** QA agent 不需要"改代码"工具，开发 agent 不需要"发 PR"工具。

## 4. 上下文工程

### 4.1 长生命周期状态放文件，不放 prompt

规范核心：**计划、决策、进度必须是文件**。prompt 里只放"当前这一步需要什么"。

- 上下文按任务裁剪，**禁止把整个代码库塞进去**
- 多会话任务必须有明确的压缩策略（复用 OpenHands condenser）
- 上下文中**禁止出现任何密钥、凭据**（复用 `conversation/secret_registry.py`）

### 4.2 不要机器生成上下文文件 ⚠️

ETH Zurich 实证：**机器生成的上下文文件会降低成功率，同时增加推理成本。**

因此：`AGENTS.md` 类工件由**人写或人审**，不让 agent 自动生成后直接投喂给下游 agent。

### 4.3 关键规则不能依赖压缩存活

Claude Code 压缩机制实证：压缩后**初始指令、中间决策、风格规则会丢失**，只保留当前任务、近期错误、文件名。

因此：不可丢失的规则（安全约束、验收标准）必须放在**每次重建上下文时都会重新注入**的位置，不能指望它从历史里活下来。

## 5. 验证闭环与评估

### 5.1 验证必须 agent 自执行

- 每个工件的验收标准**在任务开始前写定**，不允许事后补
- agent 必须**能自己运行验证命令**（如 `pytest`），而非只等人工评审
- 任务完成时**自动触发验证**，不只是 PR 时触发
- 优先**可执行脚本**而非文字规则（2026 审计的核心建议）

### 5.2 Evals：P0 阶段建立轻量评估集 ⚠️

**每个角色 agent 配 3-5 个 golden task 评估用例**（固定输入 + 期望产出标准）。
**改 prompt 或改流程后必须跑评估集防回退** —— 评估集是 harness 工作的"训练数据"。

采用 OpenAI 分层原则（避免评估成本压垮 CI）：

1. **先做确定性检查**：JSONL trace 捕获，校验命令序列、token 预算、仓库整洁度、工件 schema 合法性
2. **只在确定性检查不足处**才用 rubric / LLM-as-judge
3. 评估四维度：**outcome（结果）、process（过程）、style（风格）、efficiency（效率）**

工具：`pytest-recording` / `vcrpy` 录制回放 LLM 调用 —— 否则评估既慢又贵又 flaky，无法进 CI。

### 5.3 预算护栏

- **任务级预算**：每个任务设 LLM 成本上限，超限自动暂停并告警（借鉴 MetaGPT `investment` / `NoMoneyException`）
- litellm 统一计量 token 与费用
- ⚠️ **慎用 reviewer agent**：2026 审计发现**评审 agent 可能反而降低性能**。加评审环节前需有评估数据支撑

## 6. 工件与状态存储

**决策：文件为主 + DB 索引。**

| 层 | 内容 | 位置 |
|---|---|---|
| **文件**（权威源） | PRD、设计文档、PLAN.md、IMPLEMENT.md、测试计划 | 沙箱仓库内 markdown，**随 git 提交** |
| **DB**（索引） | 任务状态、工件路径指针、审计日志、预算记录 | Postgres |

理由：符合"长生命周期状态放文件"规范；工件可人工审阅、可版本化、可 diff；断点续跑时 agent 直接从文件恢复上下文（Anthropic 长任务 harness 模式：以 feature list / git commit / test gate 作为跨会话状态）。

### 6.1 计划工件

非平凡任务必须有 `PLAN.md`：

- 里程碑**带显式验证命令**
- 明确写出 **in-scope / out-of-scope** 边界
- `IMPLEMENT.md` 边做边记决策与偏离，不允许事后补写

## 7. 权限与沙箱

- **最小权限**：agent 只获得完成该任务所需的最小权限
- **破坏性操作需显式确认**
- 文件系统访问**限定在项目目录内**
- 网络访问**尽可能收窄**
- 命令风险分析用 **tree-sitter AST 解析**（复用 OpenHands `security/_shell_ast.py`），**禁止用正则匹配危险命令**
- 复用 OpenHands `security/defense_in_depth` + `ensemble.py` 多分析器投票

## 8. 组件移除条件（Anti-Rot）⚠️

Anthropic 核心洞察：**每个 harness 组件的存在，都假设了模型做不到某件事。这些假设会过期。**

实例：Anthropic 为 Opus 5 **删除了约 80% 的系统提示词**（那些是在修正模型已不再有的行为），**评估无回退**。

### 强制要求

每个 harness 组件（约束、检查器、提示词段落、工具限制）都必须记录移除条件：

| 组件 | 存在因为 | 可移除当 |
|---|---|---|
| _(示例)_ 强制 PRD 模板校验 | 模型自由格式输出不稳定 | 结构化输出成功率稳定 >99% |

### 消融方法论（每次模型升级时执行）

1. 完全剥离该约束/提示词段落
2. 只加回**能修复重复出现失败**的那几行
3. 用评估集验证无回退
4. 每次模型发布时重复此流程

**累积的指令是对模型能力的下注，会过期。** 不做消融，harness 会持续腐烂并拖慢、变贵、变差。

## 9. 交付门禁：HARNESS_CHECKLIST

每个角色 agent 上线前必须逐项通过。**未通过项为阻塞项；跳过项需书面说明理由。**

### Agent 指令
- [ ] 项目概述准确且最新
- [ ] 仓库结构反映当前布局
- [ ] 工具权限显式声明 —— allowed / restricted / not-allowed 三者都写明
- [ ] 验证门禁已定义且命令正确
- [ ] 无可能被多种解读的模糊指令

### 工具设计
- [ ] 每个工具名称清晰无歧义
- [ ] 工具 schema 最小化
- [ ] 错误信息告诉 agent 下一步做什么
- [ ] 返回值形状一致（成功与失败）
- [ ] 没有工具做超过一件概念上的事

### 上下文供给
- [ ] 上下文裁剪到本任务所需，非整个代码库
- [ ] 长生命周期状态（计划、决策、进度）在文件里，不在 prompt 里
- [ ] 多会话任务已定义压缩策略
- [ ] agent 可访问的上下文中无敏感数据

### 计划工件
- [ ] 非平凡任务有 PLAN.md
- [ ] 里程碑带显式验证命令
- [ ] 范围边界（in-scope / out-of-scope）已写下
- [ ] IMPLEMENT.md 实时记录决策与偏离

### 权限与沙箱
- [ ] agent 以完成任务所需的最小权限运行
- [ ] 破坏性操作需显式确认
- [ ] 网络访问已尽可能收窄
- [ ] 文件系统访问限定在项目目录

### 验证闭环
- [ ] agent 产出物有测试
- [ ] agent 能自己运行验证命令（不只是"人工评审"）
- [ ] 任务完成时自动运行验证，不只在 PR 时
- [ ] 评估标准在任务开始前已写下

### 评估集（本项目追加）
- [ ] 该角色有 3-5 个 golden task 评估用例
- [ ] 评估可离线跑（LLM 调用已录制回放）
- [ ] 改 prompt/流程后评估无回退
- [ ] 确定性检查优先，LLM-as-judge 仅用于必要处

### 移除条件（本项目追加）
- [ ] 每个 harness 组件都记录了移除条件
- [ ] 模型升级时的消融流程已定义

## 10. 可观测性

**没有 tracing 的多 agent 系统等于盲飞。**

- OpenTelemetry 标准埋点（每次推理、每个工具调用一个 span）
- Trace 后端：Langfuse（自托管）或 Arize Phoenix
- 记录：token 消耗、成本、延迟、每步决策、handoff 内容
- 支持**离线重放**每一步推理与工具调用（Phoenix / AgentOps）

## 11. 参考来源

**规范定义**
- arXiv 2606.10106 — What makes a harness a harness（四要素构成性定义）
- arXiv 2604.18071 — Architectural Design Decisions in AI Agent Harnesses（70 个系统实证）
- LangChain — The Anatomy of an Agent Harness（五原语 + 共演化警告）
- Martin Fowler / Birgitta Böckeler — Harness Engineering（三系统模型、前馈+反馈）

**工程实践**
- OpenAI — Harness Engineering / Unrolling the Codex Agent Loop / Run Long-Horizon Tasks
- Anthropic — Building Effective Agents / Harness Design for Long-Running Apps / Writing Effective Tools for Agents / Effective Harnesses for Long-Running Agents / Beyond Permission Prompts / Demystifying Evals
- Microsoft — Azure SRE Agent 架构与上下文工程（35000+ 事件、40.5h→3min）
- GitHub — Multi-Agent Workflows Often Fail（typed schema + 边界校验）
- marmelab — The State of AI Harness Engineering 2026（246 仓库审计）
- Lilian Weng — Harness Engineering for Self-Improvement
- Boris Cherny (YC, 2026-07) — Building Claude Code（消融方法论、删 80% 系统提示词）

**清单与模板**
- `ai-boost/awesome-harness-engineering` ⭐4.8k — HARNESS_CHECKLIST / AGENTS.md / PLAN.md / IMPLEMENT.md 模板
- `github/spec-kit` — spec.md → plan.md → tasks.md 规格驱动开发
