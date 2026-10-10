# 数据库升级与 Velora 记忆恢复

## 升级步骤

1. 停止 memory 服务，备份数据库。SQLite 在服务停止后复制数据库文件；Postgres 使用数据库备份工具。
2. 在数据库副本上启动新版 atom-memory，检查 `/health`、sources、recall、纠正与删除功能。
3. 确认副本正常后，用新版启动正式 memory 服务，再启动有可靠投递功能的 Velora。
4. 回退时停止服务并恢复升级前备份，使用与该备份匹配的程序版本。迁移不提供破坏性自动降级。

`atom_memory_schema` 记录迁移版本；重复启动不重复执行已经完成的升级。
版本 1 补齐 atom 的 memory_layer、happened_on 和 revision 的 happened_on；旧 atom 默认 L1。
版本 2 添加 source 幂等键与同一 space 内的唯一索引，原 source 保留且无需回填。
SQLite 升级持有写锁；Postgres 升级持有事务 advisory lock。
迁移错误直接阻止启动；SQLite 的失败迁移会回退事务。不要手动提高版本号来绕过失败。
新程序拒绝打开高于其支持版本的数据库，防止旧结构假设破坏新数据。
迁移版本与 atom/source 自身的 schema_version 是不同概念：前者描述数据库结构，后者描述数据模型。

## 联动投递语义

Velora 将用户消息、助手消息及该轮待投递记忆一次提交到自身数据库。
因此进程在 HTTP 投递前退出时，聊天与待投递记录仍然一起存在；提交失败则两者都回退。
请求使用稳定幂等键，重启与超时补发不会产生重复 source。
后台每 30 秒取最多 20 条待投递记录，同一记忆库合批固化；失败条目轮转，避免堵住后面的记录。
固化和后台补发共用 Velora 的 space 锁，并保留 L2/L3 的既有调度节奏。

状态区分：queued=仍在 Velora 等待投递；pending=已保存到 memory 但尚未被固化消费；
consolidated=本次 source 已产生记忆更新。普通 source 的 skipped 是正常遗忘，视为处理完成。
固化返回 succeeded 不能独自证明纠正生效，必须核验相应 source 状态。
中断超过 30 分钟的 running 固化会标为 failed，pending source 保留供再次处理。

## 隐私与范围

待投递记录保存原始材料，需与聊天数据库一同保护。
上游隐私删除必须取消相应待投递记录，避免删除后的材料被再次投递。
本次联动不修改 Velora 固定的 atom-memory 依赖版本；部署时必须确保实际运行的服务具有幂等及 source 状态查询接口。
数据库迁移已在 SQLite 回归测试中验证；Postgres 路径尚未在真实数据库上运行验证。
