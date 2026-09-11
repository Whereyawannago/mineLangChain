"""pytest 全局配置。

这里只做一件事：**把测试跑起来所需的环境变量固定下来**。conftest 在收集测试模块
之前执行，是整个测试进程里唯一能保证「早于任何被测代码 import」的时机。

三条约定：

1. ``HF_HOME`` → 仓库内的 ``.huggingface``。
   ``huggingface_hub`` 在 import 期就把缓存根目录固化成模块常量，之后改环境变量无效。
   ``import agent`` 时 ``agent/__init__.py`` 顶部会调 ``configure_hf_cache()``，那已经
   早于任何 langchain import；这里再显式调一次，是为了把「测试也依赖这个时序」写在
   明面上（幂等，且能在固化值不对时打出 WARNING）。

2. 记忆后端强制内存，避免测试往 ``data/memory/*.sqlite`` 写真实数据、
   也避免测试之间因持久化状态互相污染。需要真的验证 sqlite 的测试自行覆盖
   ``AGENT_MEMORY_DB`` 到 tmp_path（见 test_persistence.py）。

3. ``ANTHROPIC_API_KEY`` 给占位值。
   ``build_llm()`` 只校验非空、不发请求，所以占位值足够让构造通过。
   这条很关键：``agent`` 包已经不在 import 期 ``load_dotenv()`` 了（见
   ``agent/bootstrap.py``），测试如果还指望 .env 里恰好有真 key 才能跑，
   那 CI 上就是必挂 —— 依赖必须在测试侧显式声明，不能靠隐式副作用碰运气。
   用 ``setdefault``：开发者本机若已导出真 key，不被覆盖。
"""

import os

# 2. 记忆后端走内存（必须早于任何 agent.context 的 import）
os.environ.setdefault("AGENT_MEMORY_BACKEND", "memory")

# 3. LLM 凭据占位（不会真的发请求）
os.environ.setdefault("ANTHROPIC_API_KEY", "pytest-placeholder-not-a-real-credential")

# 1. HF 缓存目录 —— import agent 时已经设好，这里再调一次做显式声明 + 时序校验
from agent import configure_hf_cache  # noqa: E402

configure_hf_cache()
