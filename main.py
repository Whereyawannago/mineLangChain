"""多轮对话入口（流式输出版）

三种 stream_mode 同时启用，把 LangChain 内部的数据流实时打到终端：

    1) "messages" —— LLM 逐 token 输出（打字机效果）
    2) "custom"   —— 工具内部用 get_stream_writer() 推送的任意数据
    3) "updates"  —— 每个节点执行完后 state 的增量（diff）

启动方式：uv run python main.py
调试模式：uv run python main.py --debug   （额外打印 updates）
"""

import argparse
import sys
import uuid

# Windows 终端默认 GBK，改成 UTF-8 才能正常打印 emoji / 中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from agent import MAX_CONTEXT_TOKENS, build_agent


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
    if not payload:
        return f"  ↳ [{name}] (no payload)"
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


def stream_turn(agent, user_input: str, config: dict, *, show_updates: bool) -> None:
    """调用一次 agent.stream，把三种 stream_mode 的事件分发到终端。"""
    # stream_mode 列表：同时订阅三种事件流
    # 返回 (mode, event) 元组迭代器
    stream = agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["messages", "custom", "updates"],
    )

    in_token_run = False  # 跟踪是否正在打印 token 流

    for mode, event in stream:
        if mode == "messages":
            chunk, _meta = event
            text = _extract_text(chunk.content)
            if text:
                if not in_token_run:
                    sys.stdout.write("\n<<< ")
                    in_token_run = True
                sys.stdout.write(text)
                sys.stdout.flush()

        elif mode == "custom":
            # 来自 get_stream_writer() 的 dict / 任意对象
            # 换行（如果上一个 mode 是 messages 正在打 token，先收尾）
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            sys.stdout.write(f"  ⚙ [custom] {event}\n")
            sys.stdout.flush()

        elif mode == "updates":
            if not show_updates:
                continue
            if in_token_run:
                sys.stdout.write("\n")
                in_token_run = False
            # event 是 {node_name: state_diff}
            for name, payload in event.items():
                sys.stdout.write(_format_update(name, payload) + "\n")
            sys.stdout.flush()

    if in_token_run:
        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain 流式对话演示")
    parser.add_argument("--debug", action="store_true", help="额外打印每步 state 增量")
    args = parser.parse_args()

    agent = build_agent()

    # 同 thread_id 内的多轮对话共享短期记忆
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== 会话 ID：{thread_id} ===")
    print("=== 三类上下文管理 + 三类数据流已启用 ===")
    print("  上下文:")
    print("    1) 短期记忆：本会话多轮对话")
    print(f"    2) 消息裁剪：超 {MAX_CONTEXT_TOKENS} token 自动截断")
    print("    3) 长期记忆：保存用户偏好（跨会话保留）")
    print("  数据流（stream_mode）:")
    print('    1) "messages" —— LLM 逐 token 输出')
    print('    2) "custom"   —— 工具内 get_stream_writer 推送')
    print('    3) "updates"  —— 每步 state 增量（--debug 开启时打印）')
    print()
    print("试试：")
    print('  - 跟我说 "你好"  （看 token 流）')
    print('  - 跟我说 "请用 slow_lookup 查一下北京天气"  （看 custom 流）')
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