"""请求身份（identity）—— user_id / thread_id / role 的解析与生成。

多用户服务里三件事必须分开隔离：
  - 短期记忆（对话历史）靠 **thread_id** 隔离；
  - 长期记忆（用户偏好）靠 **user_id** 命名空间隔离；
  - 知识检索（RAG）靠 **role** 控制可见文档集合。

关键约束：这三者都必须来自**请求上下文**（认证后的身份 / 会话 ID / 角色），
不能是模块级硬编码常量，也不能每次会话随机 uuid——否则重启或换 worker 就"失忆"。

用法（服务端）：
    from agent.context import UserContext, new_thread_id

    user_id = authenticate(request)                      # 你自己的鉴权
    role    = user.role                                   # 角色（admin/user/...）
    thread_id = new_thread_id(user_id, session_id)       # 同一会话稳定复用
    agent.invoke(
        {"messages": [...]},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=user_id, thread_id=thread_id, role=role),
    )

工具侧（@tool）用 `user_id_from_runtime(runtime)` 读当前用户，见 long_term.py；
RAG 检索侧通过 ContextVar 拿到当前角色，见 acl.py。
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)

# 没有显式身份时的兜底用户（本地 CLI / demo）。生产必须由鉴权层给出真实 user_id。
DEFAULT_USER_ID = "local-dev"

# 角色枚举：用 Literal 而不是裸 str —— 拼错的 "admn" 在静态检查阶段就被拦住，
# 而不是拖到运行时在 _resolve_role 里静默回落到 user（那是一类很难查的隐蔽 bug）。
# 新增角色时：往 Literal 里加字面量 + 往 VALID_ROLES 加常量 + 在 allowed_paths_for_role 里给策略。
Role = Literal["admin", "user"]
ROLE_ADMIN: Role = "admin"
ROLE_USER: Role = "user"
VALID_ROLES: tuple[Role, ...] = (ROLE_ADMIN, ROLE_USER)


def _resolve_role(role: str | None) -> Role:
    """规范化角色名（大小写 / 空白 / 非法值一律回落到 user）。

    入参故意宽到 ``str | None``：它面向的是外部输入（HTTP 头 / CLI 参数 / 历史数据），
    出参收窄到 ``Role``，让下游拿到的一定是合法枚举值。
    """
    r = (role or os.getenv("AGENT_DEFAULT_ROLE") or ROLE_USER).strip().lower()
    if r == ROLE_ADMIN:
        return ROLE_ADMIN
    if r != ROLE_USER:
        logger.debug("[identity] 非法角色 %r 回落到 %r", role, ROLE_USER)
    return ROLE_USER


# 默认角色允许访问的目录（顶层）—— admin 放行全集，user 仅限 BackEndNote（环境可覆盖）
#   AGENT_USER_ALLOWED_DIRS           逗号分隔的多个绝对路径（优先级最高）
#   AGENT_OBSIDIAN_ROOT               vault 根目录
#   AGENT_USER_ALLOWED_DIRS_RELATIVE  相对 vault 根的子目录，逗号分隔
_USER_ALLOWED_DIRS_DEFAULT = ("BackEndNote",)   # 相对于 Obsidian vault 根
#: 本地开发机的 vault 兜底位置。**只是兜底**：部署到别的机器请在 .env 里设
#: AGENT_OBSIDIAN_ROOT（或直接用 AGENT_USER_ALLOWED_DIRS 给绝对路径）。
#: demos/ingest_obsidian_notes.py 也读这个常量，避免同一个路径在仓库里写死两遍。
DEFAULT_OBSIDIAN_ROOT = r"G:/ObsidianNote"


def user_allowed_dirs() -> tuple[Path, ...]:
    """user 角色默认能见的目录集合（绝对路径形式）。

    **每次调用都重新读环境变量**，不在模块加载时求值成常量 —— 否则：
      - 测试里 ``monkeypatch.setenv("AGENT_USER_ALLOWED_DIRS", ...)`` 完全不生效；
      - 运行期想热更新白名单（改配置中心 / 发 SIGHUP）也做不到。
    代价只是几个 Path.resolve()，相对一次向量检索可以忽略。

    约定：优先看 AGENT_USER_ALLOWED_DIRS（绝对路径，逗号分隔），否则用
    AGENT_USER_ALLOWED_DIRS_RELATIVE 解析到 AGENT_OBSIDIAN_ROOT 下。
    """
    raw = os.getenv("AGENT_USER_ALLOWED_DIRS")
    if raw:
        return tuple(Path(p).expanduser().resolve() for p in raw.split(",") if p.strip())
    root = Path(os.getenv("AGENT_OBSIDIAN_ROOT") or DEFAULT_OBSIDIAN_ROOT).expanduser().resolve()
    rels = os.getenv("AGENT_USER_ALLOWED_DIRS_RELATIVE") or ",".join(_USER_ALLOWED_DIRS_DEFAULT)
    return tuple((root / r.strip()).resolve() for r in rels.split(",") if r.strip())


def __getattr__(name: str) -> Any:
    """兼容旧写法 ``from agent.context.identity import USER_ALLOWED_DIRS``。

    常量已改成函数 :func:`user_allowed_dirs`（见其 docstring 说明的原因）。
    这里保留一个模块级惰性属性，取到的永远是**当前环境变量**下的结果，
    而不是 import 那一刻的快照。新代码请直接调函数。
    """
    if name == "USER_ALLOWED_DIRS":
        return user_allowed_dirs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 当前请求的上下文：用 ContextVar 在请求线程里安全传递，RAG/工具层读取
_current_context: ContextVar[UserContext | None] = ContextVar("current_user_context", default=None)


@dataclass(frozen=True)
class UserContext:
    """一次调用的身份上下文。

    通过 `create_agent(context_schema=UserContext)` 注册；`invoke(..., context=...)`
    或 `stream(..., context=...)` 传入。也接受等价 dict：`{"user_id": "...", "role": "..."}`。
    """

    user_id: str = DEFAULT_USER_ID
    thread_id: str | None = None
    role: Role = ROLE_USER
    # 允许把"本请求额外的 ACL 路径白名单"塞进来，比如某条租约临时放行某个目录；
    # 不写也可以，因为默认从 role 推导。
    extra_allowed_paths: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # 规范化 role（user 输入非法值时回落 user）
        object.__setattr__(self, "role", _resolve_role(self.role))


# ────────────────────────────── 上下文传递 ──────────────────────────────


def set_current_context(ctx: UserContext | None) -> object:
    """把当前请求的 UserContext 设进 ContextVar（API 层调用）。返回 token 用于 reset。"""
    return _current_context.set(ctx)


def reset_current_context(token: object) -> None:
    _current_context.reset(token)


def current_context() -> UserContext | None:
    """从 ContextVar 取出当前请求的上下文（retriever / 工具层调用）。"""
    return _current_context.get()


# ────────────────────────────── 生成 / 解析 ──────────────────────────────


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


def role_from_context(context: Any) -> Role:
    if context is None:
        return ROLE_USER
    if isinstance(context, UserContext):
        return context.role
    if isinstance(context, Mapping):
        return _resolve_role(context.get("role"))
    return _resolve_role(getattr(context, "role", None))


def user_id_from_runtime(runtime: Any) -> str:
    """在 @tool 里用：从 ToolRuntime 取出当前用户 id。

    ToolRuntime 由工具系统自动注入（参数名 `runtime` + 类型 `ToolRuntime`），
    其中的 `context` 就是 invoke 时传进来的那个对象。
    """
    return user_id_from_context(getattr(runtime, "context", None))


def role_from_runtime(runtime: Any) -> Role:
    return role_from_context(getattr(runtime, "context", None))


# ────────────────────────────── ACL 辅助 ──────────────────────────────


def allowed_paths_for_role(role: Role, extra: Iterable[str] = ()) -> tuple[Path, ...]:
    """角色 → 该角色默认允许访问的目录集合（绝对路径）。

    admin → 空 tuple（调用方视作"全集放行"）；
    user  → user_allowed_dirs() + extra（额外白名单）。
    """
    if role == ROLE_ADMIN:
        return ()
    extras = tuple(Path(p).expanduser().resolve() for p in extra)
    return user_allowed_dirs() + extras

