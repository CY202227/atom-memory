# 对外契约：输入内容规范与召回注入格式

atom-memory 不耦合任何上游产品。身份用 space uid 隔离。

**Atom-first：** 固化产物是短 `statement` 原子。分类字段只用 **`kind`**（不用 type）。
标识只用 **`key`**（不用 slug）。断言只用 **`statement`**（不用 hook）。

## 1. 输入规范（ingest source）

`POST /spaces/{uid}/sources`：

| kind | content 格式 | salience 建议 |
|---|---|---|
| `turn` | `用户：…\nAI：…` | 0.0~0.3；失败/情绪波动 0.5+ |
| `diary` | `[YYYY-MM-DD] 日记正文` | 0.2~0.5 |
| `correction` | 用户纠正/「记住这个」原话 | **0.8~1.0** |
| `skill_run` | `任务：… 工具：… 结果：…` | 失败 0.6+，成功 0.2 |
| `document` / `manual` | 正文或投喂 | 按重要性 |

### 1.1 任务级人类反馈

纠正或明确「记住」：`kind=correction`，`salience ≥ 0.8`，写入后**立即**
`POST /consolidate`。

## 2. 召回

`POST /spaces/{uid}/recall`

```json
{
  "query": "当前用户输入",
  "method": "bm25",
  "max_atoms": 5,
  "budget_chars": 400,
  "include_recent_sources": true,
  "detail": "statement",
  "neighbor_hops": 0
}
```

| `detail` | 行为 |
|---|---|
| `statement`（默认） | 只注入 statement |
| `full` | statement + detail，仍受 `budget_chars` |

| `neighbor_hops` | 行为 |
|---|---|
| `0`（默认） | 不扩展邻居（Velora 默认不变） |
| `1` | 主命中后沿 `about` / `derived_from` 补一跳邻居（score×0.5），仍受 `max_atoms` / 预算约束 |

派生 atom（有 `derived_from` 出边）召回时 score 再乘以其 `confidence`，同等相关度下事实优先于推论。

30B 级上下文建议：`detail=statement`，`budget_chars` 日常 **200–400**；不够再
`POST …/atoms/expand`。勿默认 `detail=full`。

响应另含：`chars_used`（计入预算的 statement/detail 字符）、`atoms_clipped`
（因预算裁掉的 hit 数），便于上游调参。

空检索或清单式问句（如「你记得什么」）会按 `updated_at` 补最近的
person/event/lesson/self，仍受 `max_atoms` / `budget_chars` 约束。
注入 `context_block` 时：若 `happened_on` 等于当天，或 statement 已含该日，
则不再前缀日期（避免写入日噪声）。

写入口径：偏好与可复用用户侧事实（含一次性披露）可记；对已有事实的纯机制追问、
助手未确认臆测可不记。

`include_recent_sources`（默认 true）：pending 且 salience≥0.5 的近期 source 拼进
`<recent_sources>`。

命中字段：`key` / `kind` / `statement` / `score` / `happened_on`；（`full` 时另有 `detail`）

```xml
<recalled_memory>
<item key="lao-zhang" kind="person"> 称呼偏好老张，勿称张先生</item>
</recalled_memory>
```

## 2.1 展开与详情

`GET /spaces/{uid}/atoms`：管理向列表，分页信封（**勿整页注入 prompt**）：

```json
{"count": 123, "page": 1, "page_size": 50, "results": [/* atom 行 */]}
```

查询参数：`page`（默认 1）、`page_size`（默认 50，最大 200）、`kind`、`status`、
`updated_after` / `updated_before`（ISO datetime）。对话注入仍用 `recall` / `expand`。

`GET /spaces/{uid}/atoms/{key}`：默认轻量详情；`?include=revisions,evidence` 附加修订与出处。  
`GET /spaces/{uid}/atoms/{key}/neighbors`：出边 / 入边（`about` / `derived_from` / `contradicts`）。  
`POST /spaces/{uid}/atoms/expand`：`{"keys":["…"], "with_evidence": false}`

## 2.2 类型化链接（atom_link）

有向边，闭集三种：

| kind | 语义 | 失效传播 |
|---|---|---|
| `derived_from` | A 由 B 推出 | 是：B 删/改 → A 降置信度并清向量 |
| `about` | 弱关联（导航） | 否 |
| `contradicts` | 显式冲突 | 否 |

固化 op 可带可选 `links: [{"to":"existing-key","kind":"about"}]`；`to` 必须已存在，非法 key 静默丢弃。  
派生 atom 的 evidence 仍写底层 source 并集；`derived_from` 补「为何会变」与递归传播。

## 3. 固化与综合

`POST /spaces/{uid}/consolidate`：`{"trigger":"manual"}` 或 `{}`  
响应含 `atoms_touched`。

pending 按 `salience` 降序、同 salience 按时间升序入批。  
本批 ops 未 cite 的 source：低 salience 可标 `skipped`（遗忘是功能）；
`kind=correction` 或 `salience≥0.8` 未消费则仍 `pending`，下次 consolidate 优先入批。

`POST /spaces/{uid}/synthesize`：从已有 atom 归纳更高层认识（独立触发，宜 cron）。  
每条综合须 `derived_from` ≥2、`confidence` 必填且服务端压到 ≤0.8；禁止引入父 atom 之外的新事实。

## 4. 典型循环

```python
r = post(f"/spaces/{uid}/recall", json={"query": user_input, "method": "bm25"})
if r["context_block"]:
    prompt = r["context_block"] + "\n\n" + user_input
post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "...", "salience": 0.2})
# 纠正后：
post(.../sources, json={"kind": "correction", "content": "...", "salience": 0.9})
post(.../consolidate, json={"trigger": "correction"})
# 定时综合（可选）：
post(.../synthesize, json={})
```

## 5. 按来源删除

`.../sources/delete-by-ref[/preview]` → `atoms_to_*` / `deleted_atoms` /
`reconsolidated_atoms` / `derived_affected_atoms`（沿 `derived_from` 反向闭包降置信度）。

## 6. 上游迁移要点

- 路径 `/pages` → `/atoms`；字段 `body` → `detail`；`type`/`slug`/`hook` 不再返回
- 纠正：高 salience + 即时 consolidate
