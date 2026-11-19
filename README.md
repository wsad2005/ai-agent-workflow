# AI Agent Execution Platform（ai-agent-workflow）

![CI](https://github.com/wsad2005/ai-agent-workflow/actions/workflows/ci.yml/badge.svg)

把 v0.2 的最小 Agent 原型升级为**可工程化部署的 AI Agent 执行平台**：
任务持久化 + 七态状态机 + 规划器抽象（规则式默认 / LLM 可选）+ 工具权限分级 +
人工确认 + JSONL 追踪 + FastAPI 服务，原有 CLI 与全部测试保留。

> 当前分支：`feature/agent-platform-upgrade`（已合入 main 的全部能力 + 平台化增强）。

## 核心概念

- **Planner（规划器）**：把用户目标变成计划（工具调用步骤列表或直接回答）。
  `RuleBasedPlanner` 确定性离线；`LLMPlanner` 走 OpenAI 兼容接口，输出经
  工具白名单 + 参数 Schema 校验，非法即拒绝（PLANNER_MODE=llm 启用）。
- **Tool Registry（工具注册表）**：工具唯一命名、参数 Schema 校验、
  **权限分级（readonly / side_effect）**、超时限制。
- **Executor（执行引擎）**：事件流式执行——逐步产出 success/failed/timeout/
  confirmation_required/cancelled 事件；超时重试、最大步数、步骤间取消检查。
- **人工确认**：有副作用的工具未获批准时任务进入 `waiting_confirmation`，
  批准后从断点恢复，拒绝则失败并记录原因（持久化，重启不丢）。
- **Trace（追踪）**：JSONL 追踪（task_id/trace_id/tool_name/action/status/
  duration/error），敏感参数自动脱敏，不同任务互不混淆。

## 快速开始

```bash
py -3.14 -m venv .venv          # 或 py -3.11；本机 Git Bash 的 python 是 MSYS2 版，用 py 启动器
source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 前端：<http://127.0.0.1:8000/>（若 frontend/ 存在）
- API 文档：<http://127.0.0.1:8000/docs>

## API 使用示例

```bash
# 创建任务
curl -X POST http://127.0.0.1:8000/api/tasks \
  -H "Content-Type: application/json" -d '{"input":"计算 2+3*4 等于多少？"}'

# 执行（返回执行后的任务状态）
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/execute

# 查看任务与追踪
curl http://127.0.0.1:8000/api/tasks/<task_id>
curl http://127.0.0.1:8000/api/tasks/<task_id>/traces

# 工具清单（含权限与超时）
curl http://127.0.0.1:8000/api/tools
```

## 配置

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `PLANNER_MODE` | `rule` | `rule`（离线规则式）或 `llm`（LLM 规划，需三项 AI 配置） |
| `LLM_BASE_URL` / `LLM_API_KEY` / `GENERATION_MODEL` | 空 | OpenAI 兼容接口；LLM 规划与 real 模型客户端共用 |
| `AGENT_MODE` | `mock` | v0.2 遗留：legacy Agent 的 mock/real 模型 |
| `DATA_DIR` | `data` | SQLite、JSONL 追踪的根目录 |
| `DATABASE_URL` | SQLite | 可选 PostgreSQL |
| `LLM_TIMEOUT_SECONDS` | `30` | LLM 请求超时 |

## 测试

```bash
python -m pytest -q    # 109 项：原有 53（认证/工具/循环等）+ 平台 56（规划/执行/状态机/确认/追踪/API）
ruff check .
```

## Docker

```bash
docker compose up -d     # SQLite 与追踪持久化在宿主机 ./data
```

> Docker 配置在未安装 Docker 的机器上编写，容器构建未经实测（如实说明）。

## 安全设计

- 工具权限分级：`readonly` 可自动执行；`side_effect` **必须人工批准**；
- 计算器使用 AST 白名单求值（绝不 eval），`write_note` 防路径穿越；
- 追踪与日志对疑似敏感参数（key/token/secret/password）自动脱敏；
- 状态机拒绝一切非法跳转；终态不可再变更；
- LLM 计划强制工具白名单 + 参数 Schema 校验，不直接执行模型文本。

## 后续改进方向

1. 执行步骤的异步并发与全局超时；
2. 令牌化审批（审批链接/审批人记录）；
3. trace 按任务分文件与 Web 查看器；
4. PostgreSQL 生产验证与 Docker 镜像实测。
