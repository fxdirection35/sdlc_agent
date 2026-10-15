# sdlc_agent

AI 驱动的软件研发生命周期平台：需求进，**带真实代码变更的拉取请求**出。

平台内置四个角色 agent（产品经理 → 架构师 → 开发 → QA），三处人工门禁（PRD 评审、设计评审、代码合入）之间全自动推进。开发角色不碰 git——它产出结构化的文件编辑（补丁桥），确定性节点负责 clone、落盘、提交、推送、开 PR；平台**永不合并**，合并是人在 GitHub 上做的决定。

```
START → checkout → pm → prd_review → architect → design_review
      → developer → apply → qa → verify
                                ├─ APPROVED → deliver → merge_review → finalize → END
                                ├─ REJECTED → developer（返工循环，最多 max_qa_retries 次）
                                └─ 超预算/失联 → escalate / budget_review（人工）
```

- **checkout**：clone 目标仓库进 run 专属 workspace（`sdlc-agent/<run_id>` 分支基线），并把文件树交给后续角色。
- **apply**：把 developer 声明的 FileEdit（每文件全文内容）写进 workspace 并提交，回填 branch/commit_sha。
- **deliver**：QA 通过后、合入门禁前推分支开 PR——门禁上审的就是 GitHub 上的真实 diff。
- `eval://` 合成源在 checkout 跳过，全链退化为纯工件流（golden 评估即此形态）。

角色与流程的工程规范见 [`docs/HARNESS_SPEC.md`](docs/HARNESS_SPEC.md)（每角色 golden task 评估集、确定性优先、token 纪律）。

## 安装

Python ≥ 3.12。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[sandbox]" --group dev
```

国内网络访问 PyPI 超时，安装时指定镜像源（不要改全局配置）：

```bash
.venv/Scripts/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 90 <包名>
```

## 配置

```bash
cp config.example.yaml config.yaml
export DEEPSEEK_API_KEY=sk-...        # 默认路由；切回 Claude 见文件内说明
export GH_TOKEN=github_pat-...        # github section 存在即必需（见下）
```

配置文件会进 git，所以里面**只有环境变量的名字，绝无密钥本身**；凭据运行时从环境读取。`github` section 的存在就是代码接入的开关：example 默认携带（clone → 补丁桥 → PR），删掉该 section 即回到纯工件拓扑。服务启动时校验凭据，缺了按名字拒绝启动，不会出现「健康但每个 run 都 500」的假绿。

## 开发

每次提交前四项门禁必须全绿（另有前端改动时的前端门禁，见下）：

```bash
.venv/Scripts/python -m pytest                  # 测试
.venv/Scripts/python -m mypy                    # 类型检查（strict + pydantic 插件）
.venv/Scripts/ruff check src tests              # Lint
.venv/Scripts/ruff format --check src tests     # 格式
```

前端（React SPA，Vite + TypeScript，改前端必过）：

```bash
cd frontend && npm install
npm test          # vitest：组件 + API client，全部离线
npm run build     # tsc --noEmit + vite build
```

评估集（golden task）默认离线回放已录制的 LLM 调用，不花钱；改了 prompt / golden task / 模型配置后需重录（真实扣费）：

```bash
export DEEPSEEK_API_KEY=sk-...
.venv/Scripts/python -m pytest tests/test_llm_golden.py --record-mode=all
```

重录前先删对应 cassette——record 模式是**追加**不是覆盖，陈旧交互残留在回放时会先被匹配。凭据扫描是评估的一部分：任何 cassette 出现密钥形状的字符串即红。

## 本地起服务

```bash
export SDLC_CONFIG_PATH=config.yaml              # 缺省即 config.yaml
export SDLC_DATABASE_URL=postgresql://...        # 缺省用内存 saver（重启丢门禁，会打 warning）
export SDLC_WORKSPACES_DIR=</somewhere/durable>  # 缺省系统临时目录
uvicorn sdlc_agent.service.main:app
```

## 部署（Docker Compose）

一台团队内部服务器，一条命令起全部：API + Postgres + 构建好的 SPA（同源服务）。

```bash
cp config.example.yaml config.yaml        # 按需改模型与预算
printf 'POSTGRES_PASSWORD=设置一个数据库密码\nDEEPSEEK_API_KEY=sk-...\nGH_TOKEN=github_pat-...\n' > .env
docker compose up --build -d              # http://<host>:8000
```

- `.env` 已被 gitignore，Compose 自动读取；缺变量时 `up` 直接按名字报错，不会带空凭据启动。
- 人工门禁存 Postgres、per-run clone 存 workspaces——都在 named volume 里，`docker compose restart` / 宿主机重启都不丢；workspace 丢失由 deliver 的 sha 守卫兜底（升级而非推旧代码）。
- 镜像内不含任何密钥：`config.yaml` 运行时只读挂载，key、token 与数据库密码只在 `.env`。
- 国内构建默认走清华 pip 镜像（构建参数默认值）；境外用
  `PIP_INDEX_URL=https://pypi.org/simple docker compose build`。
  基础镜像拉取超时则为 Docker daemon 配 registry mirror（daemon 侧配置，与本仓库无关）。

## 项目结构

```
src/sdlc_agent/
  domain/         # pydantic 领域模型（工件、门禁、FileEdit 补丁桥契约）
  pipeline/       # LangGraph 图与角色（脚本角色 + litellm 角色、checkout/apply/deliver 节点）
  integrations/   # git 子进程客户端、GitHub REST（开 PR 幂等）
  sandbox/        # OpenHands SDK 执行器（QA 测试运行）
  evals/          # golden task 评估集（yaml 数据 + 确定性检查 runner）
  service/        # FastAPI 服务：graph 的 HTTP 边界 + 同源 SPA
frontend/         # React SPA（Vite + TypeScript，hash 路由，无 UI 框架）
tests/            # pytest；tests/cassettes/ 为 LLM golden 的录制回放
docs/HARNESS_SPEC.md  # 角色 agent 的 harness 工程规范（§9 为上线门禁清单）
compose.yaml / Dockerfile   # 部署：node 构建 SPA → python:3.12-slim 装 sandbox extra
```
