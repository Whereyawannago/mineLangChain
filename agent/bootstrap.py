"""运行时引导（bootstrap）—— 把「import 即配置」的隐式副作用收敛成显式、幂等的调用。

为什么要单独一个模块：
    早期版本在 ``agent/__init__.py`` 顶层直接 ``load_dotenv()`` + ``os.environ.setdefault(...)``，
    于是「导入 agent 包」这个动作会偷偷改全局环境：
      - 测试无法隔离环境变量（import 顺序决定结果，monkeypatch.setenv 可能被覆盖）；
      - 被别的进程当库复用时，宿主进程的 os.environ 被意外写入；
      - 配置来源不可见（HF_HOME 到底是谁设的？）。

现在的约定：
    - **入口脚本**（main.py / demos/* / evals/*）在最开头显式调用一次 :func:`bootstrap`；
    - **库内部**只读 ``os.environ``，不写；
    - 唯一例外是 :func:`configure_hf_cache`：``HF_HOME`` 必须在 ``huggingface_hub`` 被
      import **之前**生效（它在 import 期就把缓存根目录固化进模块常量）。而
      ``langchain_core`` 会顺手把 ``transformers → huggingface_hub`` 拖进来，所以这个
      调用被放在 ``agent/__init__.py`` 的最前面 —— 那里是本包任何 langchain import
      之前的最后一个时机。除此之外，import 期不写任何环境变量。

公开 API：
    PROJECT_ROOT, HF_CACHE_DIR, ENV_FILE   # 路径常量
    load_env()                             # 加载 .env（幂等）
    configure_hf_cache()                   # HF_HOME 指向项目内（幂等）
    configure_logging()                    # 统一日志级别 / 格式（幂等）
    bootstrap()                            # 上面三件事一把梭（幂等）
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

#: 项目根目录（本文件在 agent/ 下，所以 parent.parent）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
#: HuggingFace 模型缓存目录（已在 .gitignore 里，不占 C 盘）
HF_CACHE_DIR: Path = PROJECT_ROOT / ".huggingface"
#: 默认 .env 位置
ENV_FILE: Path = PROJECT_ROOT / ".env"

_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DEFAULT_LOG_LEVEL = "INFO"


def load_env(dotenv_path: str | Path | None = None, *, override: bool = False) -> bool:
    """把 .env 加载进 ``os.environ``。

    Args:
        dotenv_path: 显式指定 .env 路径；缺省用项目根的 ``.env``。
        override:    True 时 .env 里的值覆盖已存在的环境变量。默认 False ——
                     让「进程环境 > .env 文件」，符合 12-factor 习惯（容器里
                     用 -e 传的值不该被仓库里的 .env 悄悄改掉）。

    Returns:
        是否真的找到并加载了文件。找不到不算错误（CI / 容器里本来就没有 .env）。
    """
    path = Path(dotenv_path) if dotenv_path else ENV_FILE
    if not path.is_file():
        logger.debug("[bootstrap] .env 不存在，跳过：%s", path)
        return False
    load_dotenv(path, override=override)
    logger.debug("[bootstrap] 已加载 .env：%s", path)
    return True


def configure_hf_cache() -> str:
    """把 ``HF_HOME`` 指向项目内缓存目录；已设置则不覆盖。

    ⚠️ **时序要求**：必须在 ``huggingface_hub`` 被 import 之前调用。
    ``huggingface_hub.constants`` 在 import 期就把 ``HF_HOME`` 读成模块常量，之后再改
    环境变量对它无效 —— 表现是「模型又被下了一遍到用户目录」，不报错、很难查。

    所以本包的权威调用点在 ``agent/__init__.py`` 顶部（任何 langchain import 之前），
    而不是 ``agent/rag/embeddings.py`` —— 走到那里时 ``langchain_core`` 早已把
    ``transformers → huggingface_hub`` 拖进来了。

    Returns:
        生效的 ``HF_HOME`` 值。
    """
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    value = os.environ["HF_HOME"]
    _warn_if_hf_cache_already_frozen(value)
    return value


def _warn_if_hf_cache_already_frozen(expected: str) -> None:
    """huggingface_hub 已经 import 过、且固化下来的缓存根跟我们要的不一致时报警。"""
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is None:
        return
    raw = getattr(constants, "HF_HOME", None)
    if not raw:
        return
    try:
        if Path(raw) == Path(expected):
            return
    except (OSError, ValueError):
        return
    logger.warning(
        "[bootstrap] HF_HOME 设得太晚：huggingface_hub 已把缓存根固化为 %s，"
        "环境变量里的 %s 不会生效。请在 import agent 之前设置 HF_HOME。",
        raw, expected,
    )


def configure_logging(level: int | str | None = None, *, force: bool = False) -> None:
    """配置根 logger 的级别与格式（幂等：root 已有 handler 时默认不动）。

    Args:
        level: 显式级别；缺省读 ``AGENT_LOG_LEVEL``，再缺省 ``INFO``。
        force: True 时清掉已有 handler 重配（重复 bootstrap / 测试场景）。
    """
    raw = level if level is not None else os.getenv("AGENT_LOG_LEVEL", _DEFAULT_LOG_LEVEL)
    logging.basicConfig(
        level=raw.upper() if isinstance(raw, str) else raw,
        format=os.getenv("AGENT_LOG_FORMAT", _DEFAULT_LOG_FORMAT),
        stream=sys.stderr,
        force=force,
    )


def bootstrap(
    *,
    with_logging: bool = True,
    log_level: int | str | None = None,
    dotenv_path: str | Path | None = None,
) -> None:
    """入口脚本统一引导：.env → HF 缓存 → 日志。可重复调用，幂等。

    典型用法::

        from agent.bootstrap import bootstrap
        bootstrap()                      # main.py / demos / evals 的第一行
    """
    load_env(dotenv_path)
    configure_hf_cache()
    if with_logging:
        configure_logging(log_level)
    logger.debug("[bootstrap] done (project_root=%s)", PROJECT_ROOT)
