"""evals —— 针对 mineLangChain agent 的评测工具集。

每个子模块是独立 CLI（python -m evals.<name>），对应一类评测：
    retrieval   检索质量离线评测（零模型成本，可本机直接跑）
    agent       端到端效果回归集（需真实模型调用）
    memory      记忆复用探针（重复相似问题是否调 get_user_preference 省 token）
    latency     延迟 / 成本 profiling

运行说明见 evals/README.md。
"""
