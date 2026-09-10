"""持久化后端选择 —— 默认 SQLite（跨进程存活），可回退内存。

环境变量：
    AGENT_MEMORY_BACKEND     sqlite（默认）| memory
    AGENT_MEMORY_DB          checkpointer 的 sqlite 路径
    AGENT_MEMORY_STORE_DB    store 的 sqlite 路径
    默认目录                  <项目根>/data/memory/（已 gitignore）

为什么 checkpointer 与 store **各用独立文件**：
    两者都基于 sqlite，各自持有连接并在初始化时写表。若指向同一文件，
    `setup()` 会因跨连接写锁冲突报 `database is locked`（已实测）。分成
    checkpoints.sqlite / store.sqlite 各自独立，零锁争用，也更好分别备份。

为什么默认 sqlite 而不是 InMemory：
    InMemorySaver / InMemoryStore 进程一退出就清空，只在同进程内跨会话有效——
    对"重启即丢"这一条根本不成立。sqlite 单文件、零依赖、跨进程持久化，适合
    本地 / 单机服务。真上生产多实例则应换 PostgresSaver / PostgresStore（见 README）。

多进程写入安全：连接开启 WAL + busy_timeout，允许"一写多读"并发。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DIR = _PROJECT_ROOT / "data" / "memory"


def backend_name() -> str:
    """当前后端名（默认 sqlite）。"""
    return os.getenv("AGENT_MEMORY_BACKEND", "sqlite").strip().lower() or "sqlite"


def use_memory_backend() -> bool:
    """是否显式要求内存后端（测试 / CI 用，避免碰磁盘）。"""
    return backend_name() == "memory"


def checkpoint_db_path() -> Path:
    """checkpointer 的 sqlite 文件路径。"""
    raw = os.getenv("AGENT_MEMORY_DB")
    return Path(raw).expanduser() if raw else DEFAULT_DIR / "checkpoints.sqlite"


def store_db_path() -> Path:
    """store 的 sqlite 文件路径。"""
    raw = os.getenv("AGENT_MEMORY_STORE_DB")
    return Path(raw).expanduser() if raw else DEFAULT_DIR / "store.sqlite"


def open_sqlite(path: Path) -> sqlite3.Connection:
    """打开（必要时创建）一个配置好并发参数的 sqlite 连接。

    - isolation_level=None：**必须**。置为 autocommit，由 SqliteSaver/SqliteStore
      自己显式 `BEGIN` 管理事务；否则 sqlite3 会先隐式开事务，撞出
      `cannot start a transaction within a transaction`（官方 from_conn_string 同样这么设）。
    - check_same_thread=False：本地单机服务常见做法，允许请求线程共用连接；
    - WAL + busy_timeout：多进程/多线程写时避免立刻 `database is locked`。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
