"""延迟 / 成本 profiling —— 真实模型调用。

测量维度：
  1. 冷启动：build_agent()（embedding 加载 + Chroma 打开 + BM25 SHA1 校验 + 图构造）耗时；
  2. 每轮：总延迟、模型调用回合数、工具调用、输入/输出近似 token（成本代理指标）。

每次迭代用全新 thread_id，让「单轮」的测量不被历史消息撑大（想测多轮累积请改这里）。
需要更细的节点级耗时 → 用 main.py --debug 或开 LangSmith 看 trace。

用法：
    uv run python -m evals.latency --cold --iter 3
    uv run python -m evals.latency --query "SummarizationMiddleware 触发阈值？"  # 只测一条
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from evals.lib import (  # noqa: E402
    collect_tool_calls,
    count_message_tokens,
    enable_tracing_if_configured,
    ensure_results_dir,
    ensure_utf8,
    identity_context,
    render_table,
    stamp,
    write_jsonl,
)

# 覆盖三种典型负载：闲聊（不调工具）/ 算术（内置知识）/ RAG（会触发 search_docs）
DEFAULT_QUERIES = [
    "你好，用一句话自我介绍。",
    "1 加 1 等于几？",
    "SummarizationMiddleware 是怎么触发摘要的？",
]


def main() -> None:
    ensure_utf8()
    tracing = enable_tracing_if_configured()
    print("LangSmith tracing: " + ("ON" if tracing else "off（设 LANGSMITH_API_KEY 可开）"))

    ap = argparse.ArgumentParser(description="延迟 / 成本 profiling（真实模型）")
    ap.add_argument("--query", help="只测这一条 query")
    ap.add_argument("--iter", type=int, default=3, help="每条 query 重复次数取均值")
    ap.add_argument("--cold", action="store_true", help="额外计时 build_agent() 冷启动")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES

    if args.dry:
        print("将测量：")
        for q in queries:
            print(f"  • {q}")
        print(f"  每条重复 {args.iter} 次；冷启动计时={'开' if args.cold else '关'}")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n缺少 ANTHROPIC_API_KEY：请在项目根 .env 里配置后再跑（会消耗真实 token）。")
        return

    from agent import build_agent  # noqa: PLC0415

    cold_seconds = None
    if args.cold:
        print("\n计时 build_agent() 冷启动…")
        t0 = time.perf_counter()
        agent = build_agent()
        cold_seconds = time.perf_counter() - t0
        print(f"    ⌛ 冷启动（build_agent）：{cold_seconds:.2f}s")
    else:
        agent = build_agent()

    records: list[dict] = []
    rows: list[list] = []
    run_id = stamp()  # 本 run 唯一，避免持久化状态在多次 run 间累积影响单轮测量
    print(f"\n测 {len(queries)} 条 query × {args.iter} 次 … run={run_id}")

    for qi, q in enumerate(queries, 1):
        lat: list[float] = []
        in_toks: list[int] = []
        out_toks: list[int] = []
        model_rounds: list[int] = []
        tool_calls: set[str] = set()

        for it in range(args.iter):
            thread_id = f"lat-{run_id}-{qi}-{it}"
            config = {"configurable": {"thread_id": thread_id}}
            context = identity_context(run_id, "latency", thread_id)
            t0 = time.perf_counter()
            result = agent.invoke({"messages": [{"role": "user", "content": q}]},
                                  config=config, context=context)
            lat.append(time.perf_counter() - t0)

            msgs = result.get("messages", [])
            tok = count_message_tokens(msgs)
            in_toks.append(tok["input_approx"])
            out_toks.append(tok["output_approx"])
            calls = collect_tool_calls(msgs)
            tool_calls.update(c["name"] for c in calls)
            model_rounds.append(
                sum(1 for m in msgs
                    if getattr(m, "type", "") == "ai"
                    and (getattr(m, "tool_calls", None) or []))
            )

        rec = {
            "query": q,
            "lat_mean_s": round(sum(lat) / len(lat), 3),
            "lat_max_s": round(max(lat), 3),
            "tool_calls": sorted(tool_calls),
            "model_rounds_mean": round(sum(model_rounds) / len(model_rounds), 2),
            "input_tok_approx_mean": round(sum(in_toks) / len(in_toks)),
            "output_tok_approx_mean": round(sum(out_toks) / len(out_toks)),
            "cost_proxy_chars_per_turn": round((sum(in_toks) + sum(out_toks)) / len(in_toks)),
        }
        records.append(rec)
        rows.append([
            q[:26],
            f"{rec['lat_mean_s']:.2f}s",
            "/".join(rec["tool_calls"]) or "-",
            str(rec["model_rounds_mean"]),
            str(rec["input_tok_approx_mean"]),
            str(rec["output_tok_approx_mean"]),
        ])
        print(f"  • {q[:40]}: {rec['lat_mean_s']:.2f}s avg | "
              f"out~{rec['output_tok_approx_mean']} tok | 调 {rec['tool_calls']}")

    render_table("延迟 / token（approx）",
                 ["query", "延迟均", "工具", "模型回合均", "入tok均", "出tok均"], rows)
    avg_lat = sum(r["lat_mean_s"] for r in records) / len(records)
    print(f"\n平均单轮延迟：{avg_lat:.2f}s"
          + (f"　|　冷启动：{cold_seconds:.2f}s" if cold_seconds is not None else ""))

    out_dir = ensure_results_dir("latency")
    out = out_dir / f"{stamp()}.jsonl"
    write_jsonl(out, [{"cold_build_seconds": cold_seconds, "runs": records}])
    print(f"结果已写：{out}")


if __name__ == "__main__":
    main()
