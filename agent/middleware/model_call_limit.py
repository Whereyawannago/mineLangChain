"""模型调用上限中间件（Model call limit）

防止 agent 进入死循环或单轮消耗过多模型调用。

官方文档：
    https://docs.langchain.com/oss/python/langchain/middleware/built-in
"""

from langchain.agents.middleware import ModelCallLimitMiddleware


def make_model_call_limit_middleware() -> ModelCallLimitMiddleware:
    """构造一个 ModelCallLimitMiddleware 实例。

    - thread_limit:   同一 thread_id 内累计允许的模型调用次数
    - run_limit:      一次 invoke/stream 内允许的模型调用次数
    - exit_behavior:  "end" 表示超过限制时温和结束（不抛异常），
                      对话会拿到一条提示信息然后退出当前轮
    """
    return ModelCallLimitMiddleware(
        thread_limit=15,
        run_limit=20,
        exit_behavior="end",
    )