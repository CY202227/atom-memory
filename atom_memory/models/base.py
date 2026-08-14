"""模型公共基座：schema 版本号与统一的 UTC 时间函数。

所有核心表带 schema_version，为将来持久化结构演进留出余地。
时间一律存 UTC naive（SQLite/Postgres 通用，出入口再做时区转换）。
"""

from datetime import datetime, timezone

# v2: page 长文 → atom（statement/detail）；去掉 page_link
# v3: 重新引入类型化 atom_link（derived_from / about / contradicts）
# v4: atom.memory_layer（L1/L2/L3；L0 仍为 source）
SCHEMA_VERSION = 4


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
