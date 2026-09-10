"""记忆复用探针 —— 重复相似问题时，agent 会不会调长期记忆来省 token？

想回答的问题：用户把偏好存进长期记忆后，在**新会话**（不同 thread_id、短期记忆为空）里
反复问相近的问题，agent 是调用 get_user_preference 稳定/简短地回答，还是每次都要用户
重讲一遍（无记忆模型）？

做法：两种 agent 各跑一遍同一剧本——
  Agent M（带 preference 工具，build_agent）：先存偏好，再在 4 个独立 thread 里问相近问题。
  Agent B（基线，无 preference 工具）：同样的剧本，但没有记忆能力。
统计各自：工具调用、总输入/输出近似 token、单问平均输出长度、答案是否还认得名字。

用法：
    uv run python -m evals.memory --dry        # 预览剧本，不调模型
    uv run python -m evals.memory              # 实跑（需 ANTHROPIC_API_KEY）
"""

from __future__ import annotations

import argparse
import os
import sys
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
    final_text,
    identity_context,
    render_table,
    stamp,
    write_jsonl,
)

# 剧本：save 在一个 thread，后续每个相近问题都在全新 thread（短期记忆清空）
SAVE_TURN = "请记住：我叫小明，而且回答尽量控制在 20 字以内。"
FACT = "小明"
QUESTIONS = [
    "我叫什么名字？",
    "你还记得我叫什么吗？",
    "我的名字是？",
    "我要求你回答多长？",
]


def _build_agent(with_memory: bool):
    """构造对照 agent。

    with_memory=True：产品原样 build_agent（含 preference 工具）。
    with_memory=False：基线——同一套 prompt/中间件/RAG，但**不含** preference 工具。
    """
    from langchain.agents import create_agent  # noqa: PLC0415

    from agent.builder import SYSTEM_PROMPT, _assemble_rag_tool  # noqa: PLC0415
    from agent.context import (  # noqa: PLC0415
        create_checkpointer,
        create_store,
        preference_tools,
    )
    from agent.llm import build_llm  # noqa: PLC0415
    from agent.middleware import (  # noqa: PLC0415
        make_human_in_the_loop_middleware,
        make_model_call_limit_middleware,
        make_summarization_middleware,
    )
    from agent.tools import demo_tools  # noqa: PLC0415

    if with_memory:
        from agent import build_agent  # noqa: PLC0415
        return build_agent()

    # 走到这说明 with_memory=False：基线 = 同一套装配，只是去掉 preference 工具
    tools = demo_tools + [_assemble_rag_tool()]
    return create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        middleware=[
            make_model_call_limit_middleware(),
            make_human_in_the_loop_middleware(),
            make_summarization_middleware(),
        ],
        checkpointer=create_checkpointer(),
        store=create_store(),
    )


def _run_protocol(with_memory: bool, run_id: str) -> dict:
    agent = _build_agent(with_memory)
    all_calls: list[dict] = []
    finals: list[str] = []
    in_tok = out_tok = 0
    save_called = pref_get_called = False
    # 本变体专属的 eval 用户（含 run_id）——保证两个变体、多次 run 之间记忆互不污染
    user = f"{'mem' if with_memory else 'base'}-{run_id}"

    def step(text: str, thread: str) -> str:
        nonlocal in_tok, out_tok
        thread_id = f"{user}-{thread}"
        config = {"configurable": {"thread_id": thread_id}}
        context = identity_context(run_id, user, thread_id)
        result = agent.invoke({"messages": [{"role": "user", "content": text}]},
                              config=config, context=context)
        msgs = result.get("messages", [])
        t = count_message_tokens(msgs)
        in_tok += t["input_approx"]
        out_tok += t["output_approx"]
        return msgs

    # 先存档
    msgs = step(SAVE_TURN, "thread-save")
    all_calls.extend(collect_tool_calls(msgs))
    finals.append(final_text(msgs))
    save_called = any(c["name"] == "save_user_preference" for c in all_calls)

    # 再在独立会话里反复问相近问题
    for i, q in enumerate(QUESTIONS, 1):
        msgs = step(q, f"thread-q{i}")
        all_calls.extend(collect_tool_calls(msgs))
        finals.append(final_text(msgs))

    pref_get_called = any(c["name"] == "get_user_preference" for c in all_calls)
    # 认不认识名字：问名字/长度的那几轮里是否出现过 FACT
    name_answers = finals[1:]  # 去掉 save 轮的致谢
    known = sum(1 for a in name_answers if FACT in a)

    return {
        "variant": "with_memory" if with_memory else "baseline_no_memory",
        "save_user_preference": save_called,
        "get_user_preference": pref_get_called,
        "get_pref_calls": sum(1 for c in all_calls if c["name"] == "get_user_preference"),
        "tot_in_approx": in_tok,
        "tot_out_approx": out_tok,
        "avg_q_out_approx": round(out_tok / max(1, len(QUESTIONS)), 1),
        "answers_contain_name": f"{known}/{len(QUESTIONS)}",
        "tool_calls": [c["name"] for c in all_calls],
    }


def main() -> None:
    ensure_utf8()
    tracing = enable_tracing_if_configured()
    print("LangSmith tracing: " + ("ON" if tracing else "off（设 LANGSMITH_API_KEY 可开）"))

    ap = argparse.ArgumentParser(description="记忆复用探针（真实模型）")
    ap.add_argument("--dry", action="store_true", help="只预览剧本，不调模型")
    args = ap.parse_args()

    print("\n剧本：")
    print(f"  1) {SAVE_TURN}  （thread-save）")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"  {i+1}) {q}  （thread-q{i}，全新会话）")
    print("\n  —— 对比两个 agent：with_memory vs baseline_no_memory ——")
    if args.dry:
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n缺少 ANTHROPIC_API_KEY：请在项目根 .env 里配置后再跑（会消耗真实 token）。")
        return

    recs = [_run_protocol(True), _run_protocol(False)]
    rows = [[
        r["variant"],
        "是" if r["save_user_preference"] else "否",
        "是" if r["get_user_preference"] else "否",
        str(r["get_pref_calls"]),
        str(r["tot_in_approx"]),
        str(r["tot_out_approx"]),
        str(r["avg_q_out_approx"]),
        r["answers_contain_name"],
    ] for r in recs]

    render_table("记忆复用 vs 无记忆基线",
                 ["variant", "存档", "调记忆", "记忆次数", "总入tok", "总出tok",
                  "均问答出tok", "还认得名字"], rows)

    m = recs[0]
    b = recs[1]
    print("\n解读：")
    print(f"  • 带记忆 agent 是否调用了 get_user_preference：{'是 ✅' if m['get_user_preference'] else '否 —— 值得关注，说明它没把重复问题引导到记忆'}")
    print(f"  • 平均每题输出 token（越少越省）：with_memory={m['avg_q_out_approx']} vs "
          f"baseline={b['avg_q_out_approx']}"
          + (" → 记忆确实更省 ✅" if m["avg_q_out_approx"] < b["avg_q_out_approx"] else ""))
    print(f"  • 还记得名字：with_memory={m['answers_contain_name']} vs "
          f"baseline={b['answers_contain_name']}")

    out_dir = ensure_results_dir("memory")
    out = out_dir / f"{stamp()}.jsonl"
    write_jsonl(out, recs)
    print(f"\n结果已写：{out}")


if __name__ == "__main__":
    main()
