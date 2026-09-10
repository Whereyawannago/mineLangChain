"""请求身份（identity）—— user_id / thread_id 的解析与生成。

多用户服务里两件事必须分开隔离：
  - 短期记忆（对话历史）靠 **thread_id** 隔离；
  - 长期记忆（用户偏好）靠 **user_id** 命名空间隔离。

关键约束：这两者都必须来自**请求上下文**（认证后的身份 / 会话 ID），
不能是模块级硬编码常量，也不能每次会话随机 uuid——否则重启或换 worker 就"失忆"。

用法（服务端）：
    from agent.context import UserContext, new_thread_id

    user_id = authenticate(request)                      # 你自己的鉴权
    thread_id = new_thread_id(user_id, session_id)       # 同一会话稳定复用
    agent.invoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=user_id, thread_id=thread_id),
    )

工具侧（@tool）用 `user_id_from_runtime(runtime)` 读当前用户，见 long_term.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

# 没有显式身份时的兜底用户（本地 CLI / demo）。生产必须由鉴权层给出真实 user_id。
DEFAULT_USER_ID = "local-dev"


@dataclass(frozen=True)
class UserContext:
    """一次调用的身份上下文。

    通过 `create_agent(context_schema=UserContext)` 注册；`invoke(..., context=...)`
    或 `stream(..., context=...)` 传入。也接受等价 dict：`{"user_id": "..."}`。
    """

    user_id: str = DEFAULT_USER_ID
    thread_id: str | None = None


def new_thread_id(user_id: str, session_id: str | None = None) -> str:
    """生成 thread_id：``<user_id>:<session>``；session 缺省取随机短 id。

    同一 (user, session) 必须得到同一个 thread_id —— 这样短期记忆才能跨进程/跨请求恢复。
    生产建议由客户端或会话表提供稳定的 session_id（如会话主键），而不是让它随机。
    """
    sid = (session_id or uuid4().hex[:12]).strip()
    return f"{user_id}:{sid}"


def user_id_from_context(context: Any) -> str:
    """从 context 取 user_id，容忍多种形态（UserContext / dict / 带 user_id 的对象 / None）。"""
    if context is None:
        return DEFAULT_USER_ID
    if isinstance(context, UserContext):
        return context.user_id or DEFAULT_USER_ID
    if isinstance(context, Mapping):
        return str(context.get("user_id") or DEFAULT_USER_ID)
    return str(getattr(context, "user_id", None) or DEFAULT_USER_ID)


def user_id_from_runtime(runtime: Any) -> str:
    """在 @tool 里用：从 ToolRuntime 取出当前用户 id。

    ToolRuntime 由工具系统自动注入（参数名 `runtime` + 类型 `ToolRuntime`），
    其中的 `context` 就是 invoke 时传进来的那个对象。
    """
    return user_id_from_context(getattr(runtime, "context", None))
