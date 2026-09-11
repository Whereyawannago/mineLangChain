"""evals 共享工具：计时 / 报告 / 指标 / 消息分析 / tracing 开关。

刻意不在模块顶层 import 任何 langchain / agent 包——离线评测（retrieval）只想用这里
的轻量函数，不引入模型相关依赖。
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # evals/
PROJECT = ROOT.parent                           # 项目根
RESULTS = ROOT / "results"                      # 运行产物（已 gitignore）


def ensure_utf8() -> None:
    """Windows 终端默认 GBK，先切成 UTF-8 保证中文/emoji 正常打印。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def ensure_results_dir(tag: str) -> Path:
    d = RESULTS / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp() -> str:
    """本地时间戳，用于结果文件名。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def bootstrap_runtime(*, log_level: str | int | None = None) -> None:
    """入口引导：加载 .env / 配 HF 缓存 / 配日志。

    **必须在检查 ANTHROPIC_API_KEY 之前调**。早先 evals 是靠 ``import agent`` 的模块级
    ``load_dotenv()`` 副作用把 key 带进环境的；那个副作用已经移除（见 agent/bootstrap.py），
    改成入口显式调用 —— 否则 ``import`` 完成前看到的是空环境，会误报「缺少 API key」。
    """
    from agent.bootstrap import bootstrap  # noqa: PLC0415

    bootstrap(log_level=log_level)


def load_json(path: str | Path) -> list | dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    print(f"    ⌛ {label}: {time.perf_counter() - t0:.2f}s")


def render_table(title: str, headers: list[str], rows: list[list]) -> None:
    """终端打印一个简单 ASCII 表格。"""
    print(f"\n=== {title} ===")
    if not rows:
        print("  （无数据）")
        return
    sr = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for row in sr:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in sr:
        print(fmt(row))


# ────────────────────────────── token 近似 ──────────────────────────────

def _content_to_text(content) -> str:
    """把消息 content（str 或 多模态 list）归一成文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def approx_tokens(content, per_msg_overhead: int = 12) -> int:
    """启发式 token 估算（≈字符/3 + 每条消息固定开销），够对比用，不追求精确。"""
    return max(1, len(_content_to_text(content)) // 3 + per_msg_overhead)


def count_message_tokens(messages) -> dict[str, int]:
    """把消息列表折算成 (input_like / output_like) 的近似 token。

    input_like：除最后一条 AI 外的全部消息内容；
    output_like：最后一条 AI 消息内容（约等于模型本次生成的量）。
    仅用于纵向对比，标注 approx。
    """
    if not messages:
        return {"input_approx": 0, "output_approx": 0}
    tail = messages[-1]
    body = messages[:-1]
    return {
        "input_approx": sum(approx_tokens(_content_to_text(m.content)) for m in body),
        "output_approx": approx_tokens(_content_to_text(tail.content)),
    }


# ────────────────────────────── 消息 / 工具调用分析 ──────────────────────────────

def collect_tool_calls(messages) -> list[dict]:
    """扫描消息流里 agent 发起过的所有工具调用：返回 [{"name","args"}]。"""
    calls: list[dict] = []
    for m in messages:
        tcs = getattr(m, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
            else:
                calls.append({"name": getattr(tc, "name", None),
                              "args": getattr(tc, "args", {})})
    return calls


def final_text(messages) -> str:
    """取最后一条 AI 消息的文本内容（用于关键词检查）。"""
    for m in reversed(list(messages)):
        if getattr(m, "type", "") == "ai":
            return _content_to_text(m.content)
    return ""


# ────────────────────────────── 检索质量指标 ──────────────────────────────
# 粒度统一到「文件」：把检索结果按 metadata.source 的文件名去重成一个有序序列，
# 相关集合也是文件名的子集。recall@k / MRR 都基于这个文件级序列计算。

def _doc_filename(doc) -> str:
    src = (doc.metadata.get("source") if hasattr(doc, "metadata") else "") or ""
    src = str(src).replace("\\", "/")
    return Path(src).name if src else "unknown"


def dedup_files(docs) -> list[str]:
    """把文档序列折叠成去重后的文件名有序列表（保留首个出现位置）。

    粒度统一到「文件」：评测相关集合是文件名子集，检索结果也按文件名判命中。
    """
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        name = _doc_filename(d)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def recall_at_k(pred_files: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    if not pred_files:
        return 0.0
    top = pred_files[:k]
    return sum(1 for r in relevant if r in top) / len(relevant)


def reciprocal_rank(pred_files: list[str], relevant: set[str]) -> float:
    for i, f in enumerate(pred_files, 1):
        if f in relevant:
            return 1.0 / i
    return 0.0


def relevant_hit_rank(pred_files: list[str], relevant: set[str]) -> int | None:
    """第一个被召回的相关文件在序列中的位次（从 1 起），没命中则 None。"""
    for i, f in enumerate(pred_files, 1):
        if f in relevant:
            return i
    return None


# ────────────────────────────── LangSmith / tracing ──────────────────────────────

def identity_context(run_id: str, user: str, thread: str, role: str = "user"):
    """构造一次 eval 的 UserContext。

    贯穿整个 run 的 user/thread 用同一个 run_id 前缀 —— 这样：
      - 一次 run 内部可以用不同 thread 测「跨会话」，但长期记忆同属一个用户；
      - 不同 run 之间天然隔离，不会读到上一次 run 留下的偏好。

    ``role`` 在这里一次给全：早先的写法是本函数造一个默认 role=user 的 ctx，
    调用方再自己重建一个带 role 的 ctx —— 前一个对象纯属白造，而且两处默认值一旦
    不一致就会出现「eval 看不到该看到的文档」这类难查问题（ACL 现在是 fail-closed，
    拿不到上下文直接返回空集，更容易踩到）。

    Args:
        run_id: 本次 run 的唯一标识（一般用 ``stamp()``）。
        user:   变体 / 场景名，用于在同一 run 内区隔不同对照组的长期记忆。
        thread: thread_id（短期记忆隔离粒度）。
        role:   ACL 角色，``"admin"`` 放行全集 / ``"user"`` 仅白名单目录。
    """
    from agent import UserContext  # noqa: PLC0415

    return UserContext(user_id=f"eval-{user}-{run_id}", thread_id=thread, role=role)


def enable_tracing_if_configured() -> bool:
    """检测到 LangSmith key 就打开 tracing，返回是否已开启。

    需要任意一种：
      - 环境变量 LANGSMITH_API_KEY（推荐，自动开 LANGSMITH_TRACING_V2）
      - 手动设好 LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY

    Returns:
        True 表示 tracing 已开启（本轮调用会上传 LangSmith）。
    """
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv(
        "LANGCHAIN_API_KEY"
    ):
        return True
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "mineLangChain-evals")
        return True
    return False
