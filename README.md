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

## 项目结构

```
src/sdlc_agent/   # 主包
tests/            # 测试
```
