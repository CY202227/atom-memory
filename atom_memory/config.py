from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务配置。全部可由环境变量 / .env 覆盖。"""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ATOMMEM_")

    # sqlite 单文件起步；换 Postgres 只需改 URL，如
    # postgresql+psycopg://user:pass@127.0.0.1:5432/atom_memory
    database_url: str = "sqlite:///./atom_memory.db"

    # 固化/召回所用 LLM（OpenAI 兼容端点，vLLM 等均可）
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = ""
    llm_timeout_seconds: float = 300.0

    # 设置后所有请求须带 X-API-Key 头；留空则不鉴权（内网/本机模式）
    api_key: str = ""

    # 单次固化最多消费的 pending source 数
    consolidate_max_sources: int = 20

    # /chat demo：喂给主模型的短期 history 条数上限（user+assistant 合计）；
    # 0 表示不截断。长上下文变笨/费钱时可调低。
    chat_history_max_messages: int = 40

    # /chat demo 联网搜索（DashScope WebSearch MCP SSE）
    dashscope_api_key: str = ""
    web_search_mcp_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse"
    )


settings = Settings()
