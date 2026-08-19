"""多轮对话入口

五种 stream_mode 同时启用，把 LangChain 内部的数据流实时打到终端：

    1) "messages" —— LLM 逐 token 输出（打字机效果）
    2) "custom"   —— 工具内部用 get_stream_writer() 推送的任意数据
    3) "updates"  —— 每个节点执行完后 state 的增量（diff）
    4) "values"   —— 每个节点执行完后 state 的完整快照
    5) "events"   —— 底层图事件（start / end / token / on_tool_start ...）

启动方式：uv run python main.py
调试模式：uv run python main.py --debug   （额外打印 updates / values / events）
HITL 模式：HumanInTheLoopMiddleware 会在 slow_lookup 调用前暂停并询问用户
"""

import argparse
import json
import sys
import uuid

# Windows 终端默认 GBK，改成 UTF-8 才能正常打印 emoji / 中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langgraph.types import Command

from agent import MAX_CONTEXT_TOKENS, build_agent

_ALL_MODES = ("messages", "custom", "updates", "values", "events")


def _extract_text(chunk_content) -> str:
    """从 AIMessageChunk.content 里抽出可打印文本。

    content 可能是 str（已组装好的文本）或 list[dict]（多模态块流）。
    """
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        parts: list[str] = []
        for block in chunk_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "".join(parts)
    return ""


def _format_update(name: str, payload) -> str:
    """把 update 事件格式化成一行可读文本。"""
    if payload is None:
        return f"  ↳ [{name}] (no payload)"
    if not isinstance(payload, dict):
        return f"  ↳ [{name}] {type(payload).__name__}"
    msgs = payload.get("messages") or []
    if msgs:
        last = msgs[-1]
        last_type = type(last).__name__
        content_preview = _extract_text(getattr(last, "content", ""))
        tool_calls = getattr(last, "tool_calls", []) or []
        if tool_calls:
            tool_names = ",".join(tc.get("name", "?") for tc in tool_calls)
            return f"  ↳ [{name}] {last_type} → tool_call({tool_names})"
        if content_preview:
            preview = content_preview[:60].replace("\n", " ")
            return f"  ↳ [{name}] {last_type}: {preview}{'…' if len(content_preview) > 60 else ''}"
        return f"  ↳ [{name}] {last_type}"
    return f"  ↳ [{name}] {list(payload.keys())}"


def _prompt_for_hitl(action_requests: list[dict]) -> dict | None:
    """HITL 中断时打印待确认的工具调用，让用户决定 approve/edit/reject。

    返回 LangGraph 的 resume payload —— None 表示用户拒绝整个交互。
    """
    print("\n" + "─" * 60)
    print("⚠️  工具调用需要你的确认（Human-in-the-Loop）：")
    print("─" * 60)

    decisions: list[dict] = []
    for i, req in enumerate(action_requests, 1):
        tool_name = req.get("name", "?")
        tool_args = req.get("args", {})
        desc = req.get("description", "")
        print(f"\n[{i}] 工具名: {tool_name}")
        if desc:
            print(f"    描述:   {desc}")
        print(f"    参数:   {json.dumps(tool_args, ensure_ascii=False, indent=8)}")
        while True:
            choice = input("    决策 → (a)pprove / (e)dit / (r)eject: ").strip().lower()
            if choice in ("a", "approve", "是", "y"):
                decisions.append({"type": "approve"})
                print("    ✓ 已批准")
                break
            if choice in ("r", "reject", "否", "n"):
                decisions.append({"type": "reject"})
                print("    ✗ 已拒绝")
                break
            if choice in ("e", "edit", "改"):
                edited_args_str = input(
                    f"    请输入新的参数（JSON 格式，例: {json.dumps(tool_args, ensure_ascii=False)}）: "
                ).strip()
                try:
                    edited_args = json.loads(edited_args_str)
                    decisions.append({"type": "edit", "args": edited_args})
                    print("    ✓ 已编辑并批准")
                    break
                except json.JSONDecodeError:
                    print("    ✗ JSON 解析失败，请重新选择决策")

    print("─" * 60)
    return {"decisions": decisions}


def _drain_stream(agent, payload, config: dict, *, show_updates: bool) -> list:
    """执行一次 agent.stream，分发五种 mode 事件，返回所有 (mode, event) 列表。

    不抛 GraphInterrupt —— LangGraph 在 stream 模式下把中断作为
    `__interrupt__` update 事件 yield 出来，stream 自身会正常结束。
    """
    in_token_run = False
    chunks: list[tuple[str, object]] = []

    for mode, event in agent.stream(payload, config=config, stream_mode=list(_ALL_MODES)):
        chunks.append((mode, event))

        # ───── messages: LLM 逐 token ─────
        if mode == "messages":
            chunk, _meta = event
            text = _extract_text(chunk.content)
            if text:
                if not in_token_run:
                    sys.stdout.write("\n<<< ")
                    in_token_run = True
                sys.stdout.write(text)
                sys.stdout.flush()

        # ───── custom: 工具内 get_stream_writer() ─────
        elif mode == "custom":
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            sys.stdout.write(f"  ⚙ [custom] {event}\n")
            sys.stdout.flush()

        # ───── updates: 每节点 state 增量（含 __interrupt__） ─────
        elif mode == "updates":
            if not show_updates:
                continue
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            if isinstance(event, dict):
                for name, p in event.items():
                    sys.stdout.write(_format_update(name, p) + "\n")
            sys.stdout.flush()

        # ───── values: 每节点 state 完整快照 ─────
        elif mode == "values":
            if not show_updates:
                continue
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            msgs = (event or {}).get("messages", []) if isinstance(event, dict) else []
            sys.stdout.write(f"  ◇ [values] 当前消息数：{len(msgs)}\n")
            sys.stdout.flush()

        # ───── events: 底层图事件 ─────
        elif mode == "events":
            if not show_updates:
                continue
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            if hasattr(event, "name"):
                name = event.name
            elif isinstance(event, dict):
                name = event.get("name", "?")
            else:
                name = "?"
            sys.stdout.write(f"  ◆ [events] {name}\n")
            sys.stdout.flush()

    if in_token_run:
        sys.stdout.write("\n")

    return chunks


def _detect_interrupt(chunks: list[tuple[str, object]]) -> list[dict]:
    """从 stream chunks 中提取 HITL 中断产生的 action_requests。

    LangGraph 在 updates stream 里 yield 一个 `__interrupt__` 键，
    它的值是 (Interrupt, ...) 元组，每个 Interrupt.value 是：
        {"action_requests": [...], "review_configs": [...]}
    """
    action_requests: list[dict] = []
    for mode, event in chunks:
        if mode != "updates":
            continue
        if not isinstance(event, dict) or "__interrupt__" not in event:
            continue
        interrupts_tuple = event["__interrupt__"]
        if not interrupts_tuple:
            continue
        for intr in interrupts_tuple:
            v = getattr(intr, "value", None)
            if isinstance(v, dict) and "action_requests" in v:
                action_requests.extend(v["action_requests"])
    return action_requests


def stream_turn(agent, user_input: str, config: dict, *, show_updates: bool) -> None:
    """调用一次 agent.stream，分发五种 stream_mode 事件，并处理 HITL 中断。

    流式调用 LangGraph 时，HumanInTheLoopMiddleware 触发的中断会被框架
    作为 `__interrupt__` update 事件 yield 出来 —— 我们用 _detect_interrupt
    检测到后，提示用户决策，再用 Command(resume=...) 恢复 stream。
    """
    payload = {"messages": [{"role": "user", "content": user_input}]}

    # 第一次 stream
    chunks = _drain_stream(agent, payload, config, show_updates=show_updates)

    # 检查是否触发 HITL 中断
    action_requests = _detect_interrupt(chunks)
    if not action_requests:
        return

    # 提示用户做决策
    resume_value = _prompt_for_hitl(action_requests)
    if resume_value is None:
        return

    # 用 Command(resume=...) 恢复 stream
    sys.stdout.write("\n>>> 已恢复 stream（resume={...})\n")
    chunks2 = _drain_stream(
        agent,
        Command(resume=resume_value),
        config,
        show_updates=show_updates,
    )

    # 二次中断（理论上罕见）：提示但不无限循环
    if _detect_interrupt(chunks2):
        print("\n  ✗ 恢复后又触发 HITL 中断，请手动检查。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain 流式对话演示")
    parser.add_argument("--debug", action="store_true", help="额外打印每步 state 增量")
    args = parser.parse_args()

    agent = build_agent()

    # 同 thread_id 内的多轮对话共享短期记忆
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== 会话 ID：{thread_id} ===")
    print("=== 五类上下文管理 + 五类数据流 + 内置中间件三件套 ===")
    print("  中间件已启用:")
    print("    1) ModelCallLimitMiddleware  — thread_limit=15, run_limit=20")
    print("    2) HumanInTheLoopMiddleware  — slow_lookup 调用前询问")
    print("    3) make_trim_middleware      — 超 2000 token 自动裁剪")
    print("  数据流（stream_mode）:")
    print('    1) "messages" —— LLM 逐 token 输出')
    print('    2) "custom"   —— 工具内 get_stream_writer 推送')
    print('    3) "updates"  —— 每步 state 增量（--debug 开启时打印）')
    print('    4) "values"   —— 每步 state 完整快照（--debug 开启时打印）')
    print('    5) "events"   —— 底层图事件（--debug 开启时打印）')
    print()
    print("试试：")
    print('  - 跟我说 "你好"  （看 token 流）')
    print('  - 跟我说 "请用 slow_lookup 查一下北京天气"  （看 custom 流 + HITL 弹窗）')
    print('  - 用 --debug 启动，看每步 state 变化')
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

        try:
            stream_turn(agent, user_input, config, show_updates=args.debug)
        except Exception as e:
            print(f"\n  ✗ 出错了：{e}\n")


if __name__ == "__main__":
    main()