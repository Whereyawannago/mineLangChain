"""通用工具（非上下文管理相关）。

这里放的是"演示型"工具，用于展示 LangChain 的某些能力。
新增的产品级工具建议放到对应的 context/ 子模块。
"""

import time

from langchain.tools import tool
from langgraph.config import get_stream_writer


@tool
def slow_lookup(query: str) -> str:
    """模拟一次慢速查询，分 5 次推送进度。用于演示 stream_mode='custom'。

    当 agent 决定调用此工具时，会在每次循环里调用 get_stream_writer()
    推送一个 dict，调用方通过 agent.stream(stream_mode=["custom", ...]) 接收。

    Args:
        query: 要查询的内容（演示用，无实际含义）。
    """
    writer = get_stream_writer()
    for step in range(1, 6):
        time.sleep(0.3)
        writer({"event": "progress", "step": step, "total": 5, "query": query})
    return f"已完成 '{query}' 的查询（5/5）"


# 给 builder.py 导入用
demo_tools = [slow_lookup]