# sdlc_agent

AI-driven software development lifecycle (SDLC) assistant.

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
```

## 开发

```bash
pip install -e ".[dev]"  # 或: pip install -e . --group dev
pytest                   # 运行测试
ruff check src tests     # 代码检查
```

## 部署（Docker Compose）

一台团队内部服务器，一条命令起全部：API + Postgres + 构建好的 SPA（同源服务）。

```bash
cp config.example.yaml config.yaml        # 按需改模型与预算
printf 'POSTGRES_PASSWORD=设置一个数据库密码\nDEEPSEEK_API_KEY=sk-...\n' > .env
docker compose up --build -d              # http://<host>:8000
```

- `.env` 已被 gitignore，Compose 自动读取；缺变量时 `up` 直接按名字报错，不会带空凭据启动。
- 人工门禁存在 Postgres（named volume）里，`docker compose restart` / 宿主机重启都不丢。
- 镜像内不含任何密钥：`config.yaml` 运行时只读挂载，key 与数据库密码只在 `.env`。
- 国内构建默认走清华 pip 镜像（构建参数默认值）；境外用
  `PIP_INDEX_URL=https://pypi.org/simple docker compose build`。
  基础镜像拉取超时则为 Docker daemon 配 registry mirror（daemon 侧配置，与本仓库无关）。

## 项目结构

```
src/sdlc_agent/   # 主包
tests/            # 测试
frontend/         # React SPA（Vite + TypeScript）
compose.yaml      # Docker Compose 部署（API + Postgres + SPA）
Dockerfile        # 多阶段镜像：node 构建 SPA → python 服务
```
