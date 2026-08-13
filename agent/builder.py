"""Agent 组装层

把 LLM、上下文管理（短期/长期/裁剪）以及 system prompt 组合成最终可调用的 agent。

调用方式：
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "..."}]},
        config={"configurable": {"thread_id": "..."}},
    )
"""

from langchain.agents import create_agent

from .context import (
    create_checkpointer,
    create_store,
    make_trim_middleware,
    preference_tools,
)
from .llm import build_llm
from .tools import demo_tools

SYSTEM_PROMPT = (
    "你是一个友好、简洁的中文助手。"
    "回答时先给一句话结论，再用要点展开。"
    "当用户告诉你他的偏好（如名字、语言、称呼），用 save_user_preference 工具保存。"
    "当用户问起他的偏好，用 get_user_preference 工具查询。"
    "如果用户要求做一次慢速查询或查询某些信息，可以用 slow_lookup 工具。"
    "如果不在你的知识范围内，直接告诉用户你无法回答。"
)


def build_agent():
    """组装并返回配置好的 LangChain agent。"""
    return create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=preference_tools + demo_tools,
        middleware=[make_trim_middleware()],
        checkpointer=create_checkpointer(),
        store=create_store(),
    )