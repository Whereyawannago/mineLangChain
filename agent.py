"""LangChain Agent 的构造与配置。

本模块按官方文档（docs.langchain.com）实现了三类上下文管理：

1. 短期记忆  —— 基于 checkpointer 的多轮对话（同 thread_id 内跨轮保留）
2. 消息裁剪  —— @before_model 中间件按 token 数截断历史消息
3. 长期记忆  —— 基于 Store 的跨会话持久化（通过工具读写用户偏好）

对外只暴露 `build_agent()` 一个工厂函数，main.py 不需要知道内部细节。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.tools import ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 模块被 import 时就加载 .env，确保 build_llm 能读到 key
load_dotenv()

# ============ 常量 ============

USER_ID = "demo-user"
PREF_NAMESPACE = ("user_preferences", USER_ID)

# 超过这个 token 数（启发式估算）就触发消息裁剪
MAX_CONTEXT_TOKENS = 2000

SYSTEM_PROMPT = (
    "你是一个友好、简洁的中文助手。"
    "回答时先给一句话结论，再用要点展开。"
    "当用户告诉你他的偏好（如名字、语言、称呼），用 save_user_preference 工具保存。"
    "当用户问起他的偏好，用 get_user_preference 工具查询。"
    "如果不在你的知识范围内，直接告诉用户你无法回答。"
)


# ============ 1. LLM ============

def build_llm() -> ChatAnthropic:
    """构造一个指向 Anthropic 兼容端点（默认 minimax 代理）的 ChatAnthropic 实例。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置，请检查项目根目录的 .env 文件")

    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )


# ============ 2. 长期记忆工具 ============
# ToolRuntime 由框架在调用时注入，@tool 装饰器会自动识别

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """保存一条用户偏好到长期记忆。

    当用户告诉你他的偏好时（比如名字、语言、称呼），调用此工具保存。
    之后任何会话都可以用 get_user_preference 读回。

    Args:
        key: 偏好的键名，例如 "name"、"language"。
        value: 偏好的值，例如 "Alice"、"中文"。
    """
    runtime.store.put(PREF_NAMESPACE, key, {"value": value})
    return f"已保存偏好：{key}={value}"


@tool
def get_user_preference(key: str, runtime: ToolRuntime) -> str:
    """从长期记忆读取一条用户偏好。

    当用户问起他之前的偏好时，调用此工具查询。

    Args:
        key: 要查询的偏好键名。
    """
    item = runtime.store.get(PREF_NAMESPACE, key)
    if item is None:
        return f"未找到键 {key} 的偏好"
    return str(item.value.get("value", ""))


# ============ 3. 消息裁剪中间件 ============

def _approx_token_count(messages) -> int:
    """粗略估算消息列表的总 token 数（≈ 字符数 / 3 + 元数据开销）。"""
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
                    for v in block.values():
                        if not isinstance(v, (str, dict, list)):
                            total += len(str(v))
        # 每条消息加 4 个 token 的元数据开销
        total += 12
    return total // 3


def make_trim_middleware():
    """返回一个 @before_model 中间件，使用启发式 token 计数。

    为什么不直接用 runtime.model 计数？
      - Runtime 注入的对象没有 model 字段，需要在闭包里持有 model 实例。
    为什么不调 Anthropic 的 count_tokens API？
      - 该 API 严格校验 tool_use / tool_result 配对，
        trim 后的消息如果出现孤儿 tool_result 会被 400 拒绝。
    """
    @before_model
    def trim_history(state, runtime):
        # 没超阈值就不切
        if _approx_token_count(state["messages"]) <= MAX_CONTEXT_TOKENS:
            return None
        trimmed = trim_messages(
            state["messages"],
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",            # 保留最后 max_tokens 个 token
            token_counter=_approx_token_count,
            allow_partial=True,         # 允许切掉一个 tool_use/tool_result 对的中间部分
        )
        return {"messages": trimmed}

    return trim_history


# ============ 4. Agent 工厂 ============

def build_agent():
    """组装并返回配置好的 LangChain agent。

    返回的对象可以直接 `.invoke(input, config=config)` 调用，
    其中 `config={"configurable": {"thread_id": ...}}` 用于区分短期记忆会话。
    """
    llm = build_llm()

    # 短期记忆：内存 checkpointer —— 同 thread_id 共享会话历史
    checkpointer = InMemorySaver()

    # 长期记忆：内存 store —— 跨会话持久化用户偏好
    store = InMemoryStore()

    return create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[save_user_preference, get_user_preference],
        middleware=[make_trim_middleware()],
        checkpointer=checkpointer,
        store=store,
    )