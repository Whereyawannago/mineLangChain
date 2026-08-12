"""LangChain 基础 Agent —— 上下文管理示例

按官方文档（docs.langchain.com）实现了三类上下文管理：

1. 短期记忆  —— 基于 checkpointer 的多轮对话（同 thread_id 内跨轮保留）
2. 消息裁剪  —— @before_model 中间件按 token 数截断历史消息
3. 长期记忆  —— 基于 Store 的跨会话持久化（通过工具读写用户偏好）

启动方式：uv run python main.py
"""

import os
import sys
import uuid

# Windows 终端默认 GBK，改成 UTF-8 才能正常打印 emoji / 中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.tools import ToolRuntime, tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

# 演示用的 user_id 和存储命名空间
USER_ID = "demo-user"
PREF_NAMESPACE = ("user_preferences", USER_ID)


# ============ 1. LLM ============

def build_llm() -> ChatAnthropic:
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


# ============ 2. 长期记忆：保存 / 读取用户偏好的工具 ============
# ToolRuntime 由框架在调用时注入，tool 装饰器会自动识别

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


# ============ 3. 消息裁剪中间件（@before_model） ============
# 每次 LLM 调用前执行，按 token 阈值裁剪历史消息
# strategy="last" 表示保留最后 max_tokens 个 token
# 注意：
#   - Runtime 注入的对象没有 model 字段，用工厂函数把 token 计数器注入闭包
#   - 用启发式计数器（按字符数 / 3 估算），避免 Anthropic count_tokens API
#     对 tool_use/tool_result 配对做严格校验

MAX_CONTEXT_TOKENS = 2000


def _approx_token_count(messages) -> int:
    """粗略估算消息列表的总 token 数（≈ 字符数 / 3）。"""
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
                    # tool_call 等结构按字段长度估算
                    for v in block.values():
                        if not isinstance(v, (str, dict, list)):
                            total += len(str(v))
        # 每条消息加 4 个 token 的元数据开销
        total += 12
    return total // 3


def make_trim_middleware():
    """返回一个 @before_model 中间件实例，使用启发式 token 计数。"""

    @before_model
    def trim_history(state, runtime):
        # 如果消息少，不切
        if _approx_token_count(state["messages"]) <= MAX_CONTEXT_TOKENS:
            return None
        # 否则按 token 数裁剪，保留最新消息
        # allow_partial=True 允许切掉一个 tool_use/tool_result 对的中间部分
        trimmed = trim_messages(
            state["messages"],
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",
            token_counter=_approx_token_count,
            allow_partial=True,
        )
        return {"messages": trimmed}

    return trim_history


# ============ 4. 组装 Agent ============

def build_agent():
    llm = build_llm()

    # 短期记忆：内存 checkpointer —— 同 thread_id 共享会话历史
    checkpointer = InMemorySaver()

    # 长期记忆：内存 store —— 跨会话持久化用户偏好
    store = InMemoryStore()

    agent = create_agent(
        model=llm,
        system_prompt=(
            "你是一个友好、简洁的中文助手。"
            "回答时先给一句话结论，再用要点展开。"
            "当用户告诉你他的偏好（如名字、语言、称呼），用 save_user_preference 工具保存。"
            "当用户问起他的偏好，用 get_user_preference 工具查询。"
            "如果不在你的知识范围内，直接告诉用户你无法回答。"
        ),
        tools=[save_user_preference, get_user_preference],
        middleware=[make_trim_middleware()],
        checkpointer=checkpointer,
        store=store,
    )
    return agent


# ============ 5. 多轮对话主循环 ============

def render_message(content) -> str:
    """兼容 content 可能是 str 或 list[dict] 的情况。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def main() -> None:
    agent = build_agent()

    # 同 thread_id 内的多轮对话共享短期记忆
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== 会话 ID：{thread_id} ===")
    print("=== 三类上下文管理已启用 ===")
    print("  1) 短期记忆：本会话多轮对话")
    print("  2) 消息裁剪：超 {} token 自动截断".format(MAX_CONTEXT_TOKENS))
    print("  3) 长期记忆：保存用户偏好（跨会话保留）")
    print()
    print("试试：")
    print("  - 跟我说 '我叫 Alice'")
    print("  - 下轮再问 '我叫什么？'（验证短期记忆）")
    print("  - 退出会话后再次运行，问 '我叫什么？'（验证长期记忆）")
    print("  - 输入 quit / exit / 退出 结束")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in {"quit", "exit", "退出"}:
            print("再见！")
            break
        if not user_input:
            continue

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        last = result["messages"][-1]
        print(f"\n<<< {render_message(last.content)}\n")


if __name__ == "__main__":
    main()