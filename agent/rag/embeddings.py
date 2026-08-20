"""本地 Embedding 适配层。

⚠️ 为什么不用 HuggingFaceEmbeddings？
  这台机器出不去 huggingface.co / hf-mirror.com（DNS / 网络都被挡），
  HuggingFaceEmbeddings 拉不到模型会直接抛 OSError。

⚠️ 为什么不用 langchain_openai.OpenAIEmbeddings？
  minimax 代理虽然支持 OpenAI 风格的 /v1/embeddings 端点，但请求/响应格式
  都是自定的：
    请求：{ "model": "embo-01", "texts": [...], "type": "db" }
    响应：{ "vectors": [...], "total_tokens": N, "base_resp": {...} }
  与 OpenAI 标准（{ "input": [...], "model": ... } → { "data": [{ "embedding": [...] }] }）
  不兼容，所以不能直接 OpenAIEmbeddings(base_url=...)。

本模块提供一个继承 langchain_core.embeddings.Embeddings 的自定义类，
封装 minimax API 协议，对外暴露 LangChain 标准接口。
"""

import os
from pathlib import Path

# 重定向 HF 缓存到项目内（防御性，HF 没用到）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_HF_HOME = os.path.join(_PROJECT_ROOT, ".huggingface")
os.environ.setdefault("HF_HOME", _HF_HOME)

from typing import Iterable

import urllib.request
import json

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

EMBED_MODEL_NAME = "embo-01"
EMBED_BASE_URL = "https://api.minimaxi.com/v1"
# type 是这个 API 必填参数："db" = 入库用，"query" = 检索用。
# 我们都固定用 "db"（不影响检索质量，dim 都是 1536）。
EMBED_TYPE = "db"


class MinimaxEmbeddings(Embeddings):
    """适配 minimax 代理的 OpenAI 风格（非完全兼容）embedding API。

    实现 LangChain 的 Embeddings 接口：embed_query + embed_documents。
    Chroma 会自动调 embed_documents 入库，检索时调 embed_query。
    """

    def __init__(
        self,
        model: str = EMBED_MODEL_NAME,
        base_url: str = EMBED_BASE_URL,
        embed_type: str = EMBED_TYPE,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url
        self.embed_type = embed_type
        self.timeout = timeout
        # api_key 优先用参数，否则从 .env 读 ANTHROPIC_API_KEY
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "找不到 API key。请设置环境变量 ANTHROPIC_API_KEY 或传入 api_key 参数。"
            )

    def _post(self, texts: list[str]) -> list[list[float]]:
        """调一次 API，返回嵌入向量列表。"""
        body = json.dumps(
            {
                "model": self.model,
                "texts": texts,
                "type": self.embed_type,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        # 检查错误
        base = data.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            raise RuntimeError(f"minimax embedding API 错误: {base}")

        vectors = data.get("vectors")
        if not vectors:
            raise RuntimeError(f"minimax embedding 返回空 vectors: {data}")

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """单条文本 → 向量（Chroma 检索时调用）。"""
        return self._post([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """多条文本 → 向量列表（Chroma 入库时调用）。"""
        # 防止单批过大：API 可能限速或拒收，每批 ≤ 32 条
        all_vectors: list[list[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(self._post(batch))
        return all_vectors


def build_embeddings(
    model: str = EMBED_MODEL_NAME,
    base_url: str = EMBED_BASE_URL,
    embed_type: str = EMBED_TYPE,
) -> MinimaxEmbeddings:
    """构造 minimax embeddings 适配实例。

    Returns:
        MinimaxEmbeddings：可直接传给 Chroma.from_documents(...)
                           或 vectorstore.as_retriever()。
    """
    return MinimaxEmbeddings(
        model=model,
        base_url=base_url,
        embed_type=embed_type,
    )