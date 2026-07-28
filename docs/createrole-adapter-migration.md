# createrole adapter 跟进（atom-memory Atom-first）

本仓 `WikiMemoryStore` 已按 Atom 契约对接。产品域仍用 `MemoryItem.hook` /
`pages_touched` 等自研名，adapter 负责映射：

| atom-memory | createrole |
|---|---|
| `GET/POST /atoms…` + `key`；详情可用 `?include=revisions,evidence` | `get`/`list`/`archive`；`MemoryItem.id` |
| `kind` / `statement` / `detail` | `memory_type` / `hook` / `content` |
| `atoms_touched` / `atoms_to_*` | `pages_touched` / `pages_*`（对外 API 暂未改名） |
| recall `detail: "statement"` | 注入 `<item key="…">`；工具参数 `key` |

纠正路径（对话侧尚未接）：`kind=correction` + `salience≥0.8` + 立即 consolidate。
