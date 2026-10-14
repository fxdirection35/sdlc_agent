# SDLC Agent 前端

React SPA：运行列表、三个评审门禁（PRD / 设计 / 合并）与预算决定的操作界面。
没有路由库（两个页面，手写 hash 路由，run 链接可直接分享）、没有 UI 框架（一份
`styles.css`）。线格式类型在 `src/types.ts`，与 `src/sdlc_agent/service/models.py`
手工镜像——两边刻意不耦合构建。

## 开发

```bash
npm install
npm run dev          # Vite 开发服务器，/api 代理到 localhost:8000
```

先在仓库根目录起服务（另一个终端）：

```bash
export SDLC_CONFIG_PATH=config.yaml
export ANTHROPIC_API_KEY=sk-ant-...
export SDLC_DATABASE_URL=postgresql://...   # 不设则内存模式（重启丢门禁）
.venv/Scripts/python -m uvicorn sdlc_agent.service.main:app --port 8000
```

## 测试与构建

```bash
npm test             # vitest（组件 + API client，全部离线）
npm run typecheck    # tsc --noEmit
npm run build        # typecheck + vite build -> dist/
```

## 部署形态

`vite build` 产出 `dist/`；把它交给服务即可同源服务（无需代理）：

```bash
export SDLC_STATIC_DIR=frontend/dist
```

服务把 `dist/` 挂在 `/`（注册在所有 `/api` 路由之后，API 永远优先）。
