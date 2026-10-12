# LoCoMo 记忆试点与诊断

官方数据来源：https://github.com/snap-research/locomo ，固定提交 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376。
数据 SHA256：79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4。
本仓库仅发布评测结果和复现程序，下载数据位于被忽略的 data/public-eval。

## 离线检索试点

取第一段完整对话 conv-26，保留全部会话，只评前 100 题；其中两题无证据标注，所以计分 98 题。
未按结果筛题，金标只用于计分；oracle 用原始对话逐条生成短原子，不使用问题或答案构造原子。
这是脚本原子下的证据召回，不是实际 LLM 固化能力、答案正确率或产品整体得分。

复现命令：

```sh
python -m atom_memory.eval.run_locomo /path/to/locomo10.json --consolidate oracle --score evidence --max-samples 1 --max-questions 100 --budget-chars 400 --out report.json
python -m atom_memory.eval.run_locomo /path/to/locomo10.json --consolidate oracle --score evidence --max-samples 1 --max-questions 100 --budget-chars 1600 --max-atoms 20 --out expanded.json
python scripts/diagnose_locomo.py /path/to/locomo10.json diagnosis.json
```

400 字符/5条：全部标注证据命中 42/98=42.9%，任一证据命中 57/98=58.2%。
1600 字符/20条：全部证据命中 50/98=51.0%，任一命中 68/98=69.4%。两参数同时变化，不能归因于单独预算。
这只是一个对话试点，不能代表完整十段对话，也不能与其他项目的官方排行榜直接比较。

## 原因诊断

诊断用相同对话重建实际 atom 库。42题通过；40题至少一条金标证据不在平铺 BM25 前100；16题金标在前100但超出前5。
没有失败题被归入金标 key 缺失或分层/预算裁剪。
诊断排序是 flat BM25、实际400字符召回是默认 layered，两者相同仅在此试点事实层数据中成立，不应推广到所有库。
主要改进方向是语义候选、实体/时间消歧、再次查找与详情展开，不应仅增大上下文。

## 真实模型试点

Velora 提供 scripts/run_live_memory_pilot.py，从已配置的模型连接读取凭据，不将凭据或私有地址保存进报告。
限定前三个会话，取其内有完整证据的前12题；真实固化后以答案 F1 计分。
真实模式不再强制将普通对话标成高显著性，且无进展会终止，避免保护原料造成无穷固化循环。
该题集、上下文和计分方式不同于98题 oracle试点，不能将得分变化当作直接提升。
模型别名不能证明实际模型身份；任何超时、固化或解析失败都必须报告，而非作为“通过”隐去。


2026-10-12 真实试点：前三会话、12个完整证据覆盖问题，单来源批次、120秒超时、1200字符/10条，模型别名 odb。
固化与问答运行结束，但12题全部回答 unknown（F1=0）；这不是能力验收通过，也不等于所有模型都会如此。
平均注入87字符，提示需要进一步验证保留/检索。再次只诊断固化时遇到 JSONDecodeError，快照未成功完成，不能据此断言原子数量或唯一根因。
之前大批次与较短超时试点也发生模型错误；现有连接的结构化固化表现不稳定。
真实报告与离线报告的题集及指标不同，不能把0和42.9直接作前后比较。
