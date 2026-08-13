"""多轮对话入口

只负责交互循环（输入、输出、退出），agent 本身在 agent.py 里构造。

启动方式：uv run python main.py
"""

import sys
import uuid

# Windows 终端默认 GBK，改成 UTF-8 才能正常打印 emoji / 中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from agent import MAX_CONTEXT_TOKENS, build_agent


def render_message(content) -> str:
    """兼容 content 可能是 str 或 list[dict] 的情况。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def main() -> None:
    agent = build_agent()

    # 同 thread_id 内的多轮对话共享短期记忆
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"=== 会话 ID：{thread_id} ===")
    print("=== 三类上下文管理已启用 ===")
    print("  1) 短期记忆：本会话多轮对话")
    print(f"  2) 消息裁剪：超 {MAX_CONTEXT_TOKENS} token 自动截断")
    print("  3) 长期记忆：保存用户偏好（跨会话保留）")
    print()
    print("试试：")
    print("  - 跟我说 '我叫 Alice'")
    print("  - 下轮再问 '我叫什么？'（验证短期记忆）")
    print("  - 退出会话后再次运行，问 '我叫什么？'（验证长期记忆）")
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

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        last = result["messages"][-1]
        print(f"\n<<< {render_message(last.content)}\n")


if __name__ == "__main__":
    main()