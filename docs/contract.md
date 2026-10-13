# 对外契约：输入内容规范与召回注入格式

atom-memory 不耦合任何上游产品。身份用 space uid 隔离。

**Atom-first：** 固化产物是短 `statement` 原子。分类字段只用 **`kind`**（不用 type）。
标识只用 **`key`**（不用 slug）。断言只用 **`statement`**（不用 hook）。

## 0. 记忆分层（L0–L3）

Chat Memory 采用与业界分层叙事对齐的四层结构（设计参考
[TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/feat/server_team/README_CN.md)
的 L0→L3；**实现独立，不含 Skill/Wiki/CodeGraph/Hub**）。

| 层 | 业界说法 | atom-memory 落点 | 写入 | 召回角色 |
|---|---|---|---|---|
| L0 | Conversation | `source`（不可变原料） | ingest | 默认不注入；`expand` / evidence 下钻 |
| L1 | Atom | `atom.memory_layer=1` | `POST /consolidate` | 精确事实补洞（BM25/hybrid） |
| L2 | Scenario | `atom.memory_layer=2` | `POST /synthesize` | 场景综合块（少而稠） |
| L3 | Persona | `atom.memory_layer=3` | `POST /persona` | sticky 优先占预算 |

回填/读时推断：`derived_from` 出边 → L2；`kind=self` 或 key ∈
`{persona, user-preferred-name}` → L3；其余 → L1。

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
  "neighbor_hops": 0,
  "policy": "layered"
}
```

| `detail` | 行为 |
|---|---|
| `statement`（默认） | 只注入 statement |
| `full` | statement + detail，仍受 `budget_chars` |

| `policy` | 行为 |
|---|---|
| `layered`（默认） | L3 核心身份优先（≤2），其他层由同一检索策略统一排序；不固定插入 L2 背景，仍受 `max_atoms` / 预算 |
| `flat` | 旧平铺检索（对照/评测） |

| `neighbor_hops` | 行为 |
|---|---|
| `0`（默认） | 不扩展邻居 |
| `1` | 主命中后沿 `about` / `derived_from` 补一跳邻居（score×0.5），仍受 `max_atoms` / 预算约束 |

派生 atom（有 `derived_from` 出边）召回时 score 再乘以其 `confidence`，同等相关度下事实优先于推论。
命中可含 `memory_layer`。

30B 级上下文建议：`detail=statement`，`budget_chars` 日常 **200–400**；不够再
`POST …/atoms/expand`。勿默认 `detail=full`。

响应另含：`chars_used`（计入预算的 statement/detail 和时间信息字符，不含格式标签）、`atoms_clipped`
（因预算裁掉的 hit 数），便于上游调参。

只有清单式问句（如「你记得什么」）才按 `updated_at` 补最近的
person/event/lesson/self，仍受 `max_atoms` / `budget_chars` 约束。
注入 `context_block` 时：若 `happened_on` 等于当天，或 statement 已含该日，
则不再前缀日期（避免写入日噪声）。

写入口径：偏好与可复用用户侧事实（含一次性披露）可记；对已有事实的纯机制追问、
助手未确认臆测可不记。

`include_recent_sources`（默认 true）：pending 且 salience≥0.5 的近期 source 拼进
`<recent_sources>`。

命中字段：`key` / `kind` / `statement` / `score` / `happened_on` / `memory_layer`；
（`full` 时另有 `detail`）

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

## 3. 固化 / 综合 / 画像

`POST /spaces/{uid}/consolidate`：`{"trigger":"manual"}` 或 `{}`  
产物默认 `memory_layer=1`（L1）；canonical `persona` / `user-preferred-name` 为 L3。  
响应含 `atoms_touched`。

pending 按 `salience` 降序、同 salience 按时间升序入批。  
本批 ops 未 cite 的 source：低 salience 可标 `skipped`（遗忘是功能）；
`kind=correction` 或 `salience≥0.8` 未消费则仍 `pending`，下次 consolidate 优先入批。

`POST /spaces/{uid}/synthesize`：从已有 atom 归纳场景认识（**L2**，独立触发，宜 cron）。  
每条综合须 `derived_from` ≥2、`confidence` 必填且服务端压到 ≤0.8；禁止引入父 atom 之外的新事实；
写入 `memory_layer=2`。

`POST /spaces/{uid}/persona`：收敛稳定画像（**L3**，独立触发，宜 cron）。  
key 收敛到 `persona` / `user-preferred-name`；LLM 失败时仅把已有 canonical / self 标为 L3。

## 4. 典型循环

```python
r = post(f"/spaces/{uid}/recall", json={
    "query": user_input, "method": "bm25", "policy": "layered",
})
if r["context_block"]:
    prompt = r["context_block"] + "\n\n" + user_input
post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "...", "salience": 0.2})
# 纠正后：
post(.../sources, json={"kind": "correction", "content": "...", "salience": 0.9})
post(.../consolidate, json={"trigger": "correction"})
# 定时：L2 场景综合 + L3 画像收敛
post(.../synthesize, json={})
post(.../persona, json={})
```

## 5. 按来源删除

`.../sources/delete-by-ref[/preview]` → `atoms_to_*` / `deleted_atoms` /
`reconsolidated_atoms` / `derived_affected_atoms`（沿 `derived_from` 反向闭包降置信度）。

## 6. 上游迁移要点

- 路径 `/pages` → `/atoms`；字段 `body` → `detail`；`type`/`slug`/`hook` 不再返回
- 纠正：高 salience + 即时 consolidate


## 可靠投递与纠正确认

`POST /spaces/{uid}/sources` 支持可选 `idempotency_key`（1–128 字符）。
上游必须先持久化 payload 与该键，再发送请求；超时、断线或重启后使用同一个键重试。
同一 space 内，同键同 payload 返回原 source（包括最新 status），不创建重复证据；
同键不同 payload 返回 409。未提供键时保留原行为，相同内容也允许成为不同经历。
键不跨 space 去重；按来源隐私删除时键随 source 删除，上游必须同步取消对应待投递记录，避免重新写入已删除经历。

`GET /spaces/{uid}/sources/{source_id}` 返回该 space 内的 source；不存在返回 404。
HTTP 200 或 consolidation run 的 `succeeded` 只说明运行结束，不能证明某次纠正已生效。
上游还须确认对应 source 的 status 为 `consolidated`；`pending` 应显示待固化并允许补偿重试。
普通 source 的 `skipped` 表示按策略遗忘，也是已处理状态。

本次升级启动时自动给已有 SQLite / Postgres source 表添加 nullable idempotency_key
与 (space_id, idempotency_key) 唯一索引；旧行无需回填，旧请求仍兼容。
升级前停止服务并备份数据库，先在副本启动新版、验证 sources / recall 后再切换。
迁移由 `atom_memory_schema` 记录已执行版本，启动时按序执行；新于当前程序的数据库会拒绝启动。详见 [升级与联动恢复](upgrade-and-delivery.md)。

### 引用时间与回滚

`recall` 和 `expand` 的 hit 增加 `source_time`（可空）与 `time_context`。前者仅是当前修订所引用材料的日期，不能当成事件日期；后者只给出有依据的年、月、日或周精度，不把“上周”补成具体某一天。没有原始引用或引用时间冲突时，不使用处理日期猜测。历史导入应提供原始会话时间头或 `occurred_at`，后者的日期按 UTC 读取。

短断言也会在 `context_block` 中附带受预算约束的时间信息，字段正文计入 `chars_used`；所有 hit 都超过预算时可以返回空结果。文本和属性转义后渲染，原文中的标签不是新的记忆节点。

`GET …/atoms/{key}?include=evidence` 额外返回 `current_source_ids`，区别于所有历史出处。回滚会复制目标修订仍有效、同一空间内的来源引用；已删除的来源不会被恢复。读取不按相同文字猜测它属于哪一轮经历。
