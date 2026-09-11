"""端到端效果回归集 —— 走真实模型调用。

按 evals/agent_scenarios.json 的场景，逐条用 build_agent / build_structured_agent 跑，
对每个场景收集确定性判据（是否调用了该调的工具 / 结构化 schema / 引用是否真实），
输出 per-scenario 通过表 + JSONL。

用法：
    uv run python -m evals.agent --dry           # 只预览场景与判据，不调模型
    uv run python -m evals.agent                 # 实跑（需 .env 里 ANTHROPIC_API_KEY）
    uv run python -m evals.agent --limit 2       # 只跑前 2 条

说明：
  - 每条场景构建独立 agent（隔离短期/长期记忆，避免互相污染），会慢一些。
  - 每个场景间数据不交叉；long_term 场景内部用不同 thread 模拟跨会话。
  - 结果写 evals/results/agent/<时间戳>.jsonl。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ⚠️ 必须在检查 API key 之前调 bootstrap_runtime()（main() 的第一件事）。
# 早先这里靠 `import agent` 的模块级 load_dotenv() 副作用把 .env 带进环境；
# 那个副作用已经移除（见 agent/bootstrap.py），改成入口显式调用。
from evals.lib import (  # noqa: E402
    bootstrap_runtime,
    collect_tool_calls,
    enable_tracing_if_configured,
    ensure_results_dir,
    ensure_utf8,
    final_text,
    identity_context,
    load_json,
    render_table,
    stamp,
    write_jsonl,
)
from agent.context import set_current_context, reset_current_context  # noqa: E402


def _has_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _build_agent(scenario: dict):
    """按场景构造与产品一致的原生 agent（build_agent 或 build_structured_agent）。"""
    from agent import build_agent, build_structured_agent  # noqa: PLC0415
    from agent.structured import get_schemas  # noqa: PLC0415

    kind = scenario.get("kind", "agent")
    include_rag = scenario.get("include_rag", True)
    if kind == "structured":
        schema_name = scenario.get("schema", "ChatReply")
        return build_structured_agent(get_schemas()[schema_name], include_rag=include_rag)
    return build_agent()


def _run_scenario(scenario: dict, run_id: str) -> dict:
    """跑一个场景的所有轮次，返回记录（含各判据 ok / not）。"""
    agent = _build_agent(scenario)
    default_thread = scenario["name"]
    scope = f"{run_id}-{scenario['name']}"  # 本次 run 内该场景的记忆作用域

    turns = scenario["turns"]
    checks: list[dict] = []
    all_calls: list[dict] = []
    finals: list[str] = []
    rag_result = None
    n_model_rounds = 0

    for turn in turns:
        turn_thread = turn.get("thread") or default_thread
        thread_id = f"{scope}-{turn_thread}"   # run 唯一 → 跨 run 不串味，场景内按 turn 分会话
        config = {"configurable": {"thread_id": thread_id}}
        # 场景内所有 turn 同属一个 eval 用户（长期记忆跨 thread 共享），user 带 run_id 隔离。
        # 场景可显式指定 role（默认 user，不放行全集；ACL 才有效）。
        ctx = identity_context(run_id, scenario["name"], thread_id,
                               role=scenario.get("role", "user"))
        token = set_current_context(ctx)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": turn["text"]}]},
                                  config=config, context=ctx)
        finally:
            reset_current_context(token)
        messages = result.get("messages", [])
        all_calls.extend(collect_tool_calls(messages))
        finals.append(final_text(messages))
        n_model_rounds += sum(
            1 for m in messages
            if getattr(m, "type", "") == "ai" and (getattr(m, "tool_calls", None) or [])
        )
        if scenario.get("kind") == "structured" and rag_result is None and \
                scenario.get("expect", {}).get("verify_sources"):
            rag_result = result

    calls_now = {c["name"] for c in all_calls}
    expect = scenario.get("expect", {}) or {}

    # 判据 1：该调的工具都调了
    required = expect.get("calls_tool") or []
    missing = [t for t in required if t not in calls_now]
    checks.append({
        "check": "调用工具 " + ",".join(required) if required else "（无必须工具）",
        "ok": not missing, "detail": f"已调={sorted(calls_now)} 缺={missing}",
    })
    # 判据 2：不该调的工具没调
    forbidden = expect.get("not_calls_tool") or []
    used_forbidden = sorted(forbidden & calls_now)
    checks.append({
        "check": "不调用 " + ",".join(forbidden) if forbidden else "（无不许工具）",
        "ok": not used_forbidden, "detail": f"误调={used_forbidden}",
    })
    # 判据 3：RAGAnswer 引用真实
    if rag_result is not None:
        from agent.structured import verify_rag_sources  # noqa: PLC0415
        verified, fabricated = verify_rag_sources(rag_result)
        checks.append({
            "check": "sources 均为检索真实来源",
            "ok": not fabricated, "detail": f"伪造={fabricated}",
        })
    # 判据 4：答案里应出现的关键词（弱信号）
    answer_contains = expect.get("answer_contains") or []
    joined = "\n".join(finals)
    miss_kw = [k for k in answer_contains if k not in joined]
    checks.append({
        "check": "答案包含 " + ",".join(answer_contains) if answer_contains else "（无关键词判据）",
        "ok": not miss_kw, "detail": f"缺关键词={miss_kw}",
    })

    all_ok = all(c["ok"] for c in checks)
    return {
        "name": scenario["name"],
        "capability": scenario.get("capability", ""),
        "passed": all_ok,
        "n_checks": len(checks),
        "ok_checks": sum(1 for c in checks if c["ok"]),
        "checks": checks,
        "tool_calls": all_calls,
        "n_model_rounds": n_model_rounds,
        "answer_preview": "\n".join(finals)[:120],
    }


def main() -> None:
    ensure_utf8()
    bootstrap_runtime()
    tracing = enable_tracing_if_configured()
    print("LangSmith tracing: " + ("ON" if tracing else "off（设 LANGSMITH_API_KEY 可开）"))

    ap = argparse.ArgumentParser(description="端到端效果回归集（真实模型）")
    ap.add_argument("--scenarios", default=str(_PROJECT / "evals" / "agent_scenarios.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="只预览场景与判据，不调模型")
    args = ap.parse_args()

    raw = load_json(args.scenarios)
    scenarios = raw["scenarios"] if isinstance(raw, dict) else raw
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.dry:
        print(f"\n预览 {len(scenarios)} 个场景（不调用模型）：")
        for s in scenarios:
            exp = s.get("expect", {})
            print(f"  • {s['name']}  [{s.get('capability', '')}]"
                  + (f"  要求调用={exp.get('calls_tool')}" if exp.get("calls_tool") else ""))
        return

    if not _has_key():
        print("\n缺少 ANTHROPIC_API_KEY：请在项目根 .env 里配置后再跑（会消耗真实 token）。")
        print("先跑 --dry 预览，避免误扣费。")
        return

    rows: list[list] = []
    records: list[dict] = []
    run_id = stamp()  # 本次 run 的记忆命名空间前缀，避免与历史 run 串味
    print(f"\n跑 {len(scenarios)} 个场景（每条会真实调用模型，耐心等待）… run={run_id}")
    for s in scenarios:
        rec = _run_scenario(s, run_id)
        records.append(rec)
        rows.append([
            rec["name"],
            "✅" if rec["passed"] else "❌",
            f"{rec['ok_checks']}/{rec['n_checks']}",
            str(rec["n_model_rounds"]),
            rec["answer_preview"].replace("\n", " ")[:38],
        ])
        flag = "PASS" if rec["passed"] else "FAIL"
        print(f"  [{flag}] {rec['name']}  "
              f"({rec['ok_checks']}/{rec['n_checks']} checks)")
        for c in rec["checks"]:
            if not c["ok"]:
                print(f"         ✗ {c['check']}: {c['detail']}")

    render_table("端到端效果回归", ["scenario", "pass", "checks", "model回合", "答案预览"], rows)

    out_dir = ensure_results_dir("agent")
    out = out_dir / f"{stamp()}.jsonl"
    write_jsonl(out, records)
    print(f"\n结果已写：{out}")


if __name__ == "__main__":
    main()
