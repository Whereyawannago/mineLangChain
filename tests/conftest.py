"""pytest 全局配置。

关键：把记忆后端强制设为内存，避免测试往 data/memory/*.sqlite 里写真实数据、
也避免测试之间因持久化状态互相污染。需要真的验证 sqlite 的测试自行覆盖
AGENT_MEMORY_DB 到 tmp_path（见 test_persistence.py）。
"""

import os

# 必须早于任何 import agent.context 的模块发生 —— conftest 在收集测试前执行
os.environ.setdefault("AGENT_MEMORY_BACKEND", "memory")
