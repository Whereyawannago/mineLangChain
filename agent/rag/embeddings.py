"""本地 Embedding 适配层。

历史背景：早期版本是 ``MinimaxEmbeddings``，自己用 ``urllib`` 调
minimax 的 ``/v1/embeddings``（协议不是 OpenAI 标准的 ``data[*].embedding``，
所以不能直接用 ``OpenAIEmbeddings(base_url=...)``）。

现在改用 ``langchain_huggingface.HuggingFaceEmbeddings`` 跑本地 sentence-transformers
模型，对 LangChain 暴露标准 ``Embeddings`` 接口（``embed_query`` / ``embed_documents``）。

默认模型：``BAAI/bge-small-zh-v1.5``
  - dim = 512，对中文检索友好
  - ~93 MB，本地首次运行会下载到 ``.huggingface/`` 缓存目录

⚠️ 网络依赖：首次运行需要从 huggingface.co / hf-mirror.com 拉模型权重。
   如果机器无法出网（DNS / 防火墙），请提前手动放模型到本地缓存目录。
"""

import os

# 把 HF 缓存重定向到项目内 .huggingface/（已在 .gitignore 里），不污染用户目录
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_ROOT, ".huggingface"))

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

# bge-small-zh-v1.5：中文友好、dim 适中、模型体积小（~93MB）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 推理参数：bge 系列官方推荐把 prompt 设为 "为这个句子生成表示以用于检索中文文档："
# 但 langchain_huggingface 的 encode_kwargs 只接受传给 sentence-transformers 的参数，
# 我们不在这里改 query prompt，保持最小配置 —— 后续如果召回率不满意再加。
_DEFAULT_MODEL_KWARGS = {
    "device": "cpu",  # 容器里没有 GPU，先跑 CPU；要 GPU 改成 "cuda"
    "trust_remote_code": False,
}
_DEFAULT_ENCODE_KWARGS = {
    "normalize_embeddings": True,  # bge 官方推荐：归一化后再算余弦
    "batch_size": 32,  # 与原 batch 切分上限保持一致
}


def build_embeddings(model_name: str = EMBED_MODEL_NAME) -> HuggingFaceEmbeddings:
    """构造 HuggingFaceEmbeddings 实例。

    Args:
        model_name: HuggingFace 上的 sentence-transformers 模型仓库名。
                    默认 ``BAAI/bge-small-zh-v1.5``（中文友好，dim=512）。

    Returns:
        ``HuggingFaceEmbeddings``：可直接传给 ``Chroma.from_documents(...)``
        或 ``vectorstore.as_retriever()``。
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=_DEFAULT_MODEL_KWARGS,
        encode_kwargs=_DEFAULT_ENCODE_KWARGS,
        # cache_folder 默认走 HF_HOME，无需显式指定
    )