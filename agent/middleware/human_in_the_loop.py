"""人在回路中间件（Human-in-the-Loop）

在 agent 调用敏感工具前暂停，等待人工确认 / 编辑 / 拒绝。

官方文档：
    https://docs.langchain.com/oss/python/langchain/middleware/built-in#human-in-the-loop
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware


def make_human_in_the_loop_middleware() -> HumanInTheLoopMiddleware:
    """构造一个 HumanInTheLoopMiddleware 实例。

    interrupt_on 字典的 key 是工具名，value 是策略：
      - allowed_decisions: ["approve", "edit", "reject"] 之一
      - description:       给审批人看的描述
    """
    return HumanInTheLoopMiddleware(
        interrupt_on={
            "slow_lookup": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": (
                    "slow_lookup 是一个演示用的慢速查询，"
                    "会 sleep 5 次并推送进度。请确认是否执行。"
                ),
            },
        },
        description_prefix="⚠️ 工具调用需要确认",
    )