# atom-memory

面向 AI 助手的 **Atom-first 长期记忆服务**。

经历先写入原料（source），再固化蒸馏成短原子（atom）。对话默认只注入一句断言；需要细节时再展开。独立部署，HTTP 接到任意助手 / Agent。

## 为什么是 wiki，不是日志

| 日志式记忆 | atom-memory |
|---|---|
| 每轮塞一段原文 | 蒸馏成可复用的认识 |
| 越聊越吵 | 同主题改写同一 key，库越用越稠 |
| 全量检索硬塞上下文 | 短 statement 注入 + 按需 expand |
| 难纠正、难删除 | revision / evidence / 按来源删除 |

核心信念：**遗忘是功能**——不值得长期保留时，固化可以零操作。

## 数据流

```
经历（对话 / 纠正 / 日记 / 文档 …）
  → source                 不可变快照
  → consolidation          LLM 固化（唯一写通路）
  → atom                   statement ≤80 字 + detail ≤300 字
  → atom_revision + evidence
  → recall / expand        给上游对话注入
```

- **space**：一份隔离的记忆库（按 `uid`）
- **source**：原料；可按 `external_ref` 隐私删除
- **atom**：wiki 页；`kind` + `key` 定位，改写优先于新建
- **canonical key**：服务端合并称呼 / 人设等近义 key，减轻页膨胀

## 快速开始

要求：Python ≥ 3.11；OpenAI 兼容的 LLM 端点（固化与可选语义召回）。

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env

# macOS / Linux
# .venv/bin/pip install -e ".[dev]"
# cp .env.example .env
```

编辑 `.env`（变量均带 `ATOMMEM_` 前缀）：

```env
ATOMMEM_DATABASE_URL=sqlite:///./atom_memory.db
ATOMMEM_LLM_BASE_URL=http://127.0.0.1:8000/v1
ATOMMEM_LLM_API_KEY=EMPTY
ATOMMEM_LLM_MODEL=your-model
```

启动：

```bash
# Windows
.venv\Scripts\python -m atom_memory.main

# 或
.venv\Scripts\uvicorn atom_memory.main:app --host 0.0.0.0 --port 8020 --reload
```

| 入口 | 地址 |
|---|---|
| 健康检查 | http://127.0.0.1:8020/health |
| OpenAPI | http://127.0.0.1:8020/docs |
| 原子预览 | http://127.0.0.1:8020/ui |
| 聊天调试 | http://127.0.0.1:8020/chat |

测试：

```bash
.venv\Scripts\python -m pytest
```

## 作为产品怎么用

### 接到你的助手（推荐）

每轮对话：

1. `POST /spaces/{uid}/recall` —— 取 `context_block` 注入 prompt  
2. 模型回复后 `POST .../sources` —— 写入本轮经历  
3. 定期或纠正后 `POST .../consolidate` —— 固化进 wiki  
4. statement 不够时 `POST .../atoms/expand` —— 按 key 展开 detail  

纠正 / 「请记住」：用 `kind=correction`、较高 `salience`，并立刻 consolidate。

更细的字段与注入格式见 [docs/contract.md](docs/contract.md)。

### 内置聊天调试（/chat）

用于体感「记得 / 假失忆」，不是生产 SDK：

1. 每轮自动轻量召回（称呼 sticky + BM25 / 语义）
2. 模型可用 `memory_search` / `memory_expand` / `memory_save`
3. 漏存时由 `memory_judge` 子调用补判（有门闩，避免新闻闲聊乱记）
4. 快写 Source → 后台固化；日志落在 `logs/chat/`

浏览器打开 `/chat` 即可；勾选「允许写入记忆」后才会挂 save / judge。

## Atom 类型（kind）

`lesson` · `event` · `person` · `belief` · `procedure` · `self`

约定示例：称呼偏好 → `person` / `user-preferred-name`；人设口癖 → `self` / `persona`。

## API 速览

```
POST /spaces
GET  /spaces?owner_id=

POST /spaces/{uid}/sources
POST /spaces/{uid}/consolidate
GET  /spaces/{uid}/runs

GET  /spaces/{uid}/index
GET  /spaces/{uid}/atoms
GET  /spaces/{uid}/atoms/{key}          (+ /revisions /evidence)
POST /spaces/{uid}/atoms/expand
POST /spaces/{uid}/atoms/{key}/rollback
POST /spaces/{uid}/atoms/{key}/archive

POST /spaces/{uid}/recall
POST /spaces/{uid}/sources/delete-by-ref
POST /spaces/{uid}/sources/delete-by-ref/preview

GET  /ui
GET  /chat
GET  /health
```

召回常用参数：`method`（bm25 / fuzzy / llm）、`detail`（statement / full）、`max_atoms`、`budget_chars`。

可选鉴权：设置 `ATOMMEM_API_KEY` 后请求带 `X-API-Key`。

## 配置要点

| 变量 | 含义 |
|---|---|
| `ATOMMEM_DATABASE_URL` | 默认 SQLite；可换 Postgres |
| `ATOMMEM_LLM_*` | 固化 / 语义召回 / chat demo |
| `ATOMMEM_API_KEY` | 留空则不鉴权 |
| `ATOMMEM_CONSOLIDATE_MAX_SOURCES` | 单次固化消费的 pending 上限 |
| `ATOMMEM_CHAT_HISTORY_MAX_MESSAGES` | `/chat` 短期上下文条数；`0` 不截断 |
| `ATOMMEM_DASHSCOPE_*` / `ATOMMEM_WEB_SEARCH_MCP_URL` | `/chat` 可选联网搜索 |

## 仓库结构

```
atom_memory/
  models/           space · source · atom · revision · evidence · run
  repositories/     持久化
  consolidation/    两阶段固化 · canonical 合并 · prompts
  recall/           fuzzy · bm25 · llm · 预算裁剪
  demo/             /chat 会话 · judge · 联网
  api/              HTTP 路由
  web/              /ui · /chat 前端
docs/contract.md    输入与召回契约
tests/              FakeLLM 端到端
```
