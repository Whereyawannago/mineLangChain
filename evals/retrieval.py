"""检索质量离线评测 —— 零模型成本，本机可直接跑。

衡量 build_agent 用的那套 HybridRetriever（dense Chroma + BM25 → RRF）召回准不准，
用 recall@k / MRR。粒度按文件：相关集合是文件名，预测序列是「检索结果去重成文件名」。

用法：
    uv run python -m evals.retrieval                # 全量 golden
    uv run python -m evals.retrieval --limit 5      # 只跑前 5 条
    uv run python -m evals.retrieval --top 10       # 每路候选/融合 top 改为 10

产物：evals/results/retrieval/<时间戳>.jsonl（每 query 一条）+ 终端汇总表。
前置：需先跑过 demos/ingest_obsidian_notes.py（data/chroma_db + bm25_index.pkl 存在）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from evals.lib import (  # noqa: E402
    dedup_files,
    ensure_results_dir,
    ensure_utf8,
    load_json,
    reciprocal_rank,
    recall_at_k,
    render_table,
    stamp,
    timer,
    write_jsonl,
)


def main() -> None:
    ensure_utf8()
    ap = argparse.ArgumentParser(description="检索质量离线评测（HybridRetriever recall@k / MRR）")
    ap.add_argument("--golden", default=str(_PROJECT / "evals" / "golden_retrieval.json"),
                    help="golden 标签 JSON 路径")
    ap.add_argument("--top", type=int, default=20,
                    help="每路候选数 & 融合返回 top_k（测 recall 曲线要大于工具实际用的 4）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = ap.parse_args()

    raw = load_json(args.golden)
    cases = raw["cases"] if isinstance(raw, dict) else raw
    if args.limit:
        cases = cases[: args.limit]
    print(f"加载 {len(cases)} 条 golden 查询（top={args.top}）")

    # 组装与 build_agent 相同的混合检索（只是把 top_k 调大以便算 recall 曲线）
    from agent.rag import (  # noqa: PLC0415
        HybridRetriever,
        build_embeddings,
        load_or_build_bm25_retriever,
        load_vectorstore,
    )

    print("构建检索资产（首次会加载本地 bge 模型，稍等）…")
    with timer("assets build"):
        embeddings = build_embeddings()
        vectorstore = load_vectorstore(embeddings)
        dense = vectorstore.as_retriever(search_kwargs={"k": args.top})
        sparse = load_or_build_bm25_retriever(vectorstore, k=args.top)
        retriever = HybridRetriever(retrievers=[dense, sparse], top_k=args.top)

    ks = (1, 3, 5)
    per_query: list[dict] = []
    table_rows: list[list] = []

    for case in cases:
        query: str = case["query"]
        relevant: set[str] = set(case["relevant"])
        docs = retriever.invoke(query)
        pred = dedup_files(docs)

        rec = {k: recall_at_k(pred, relevant, k) for k in ks}
        rr = reciprocal_rank(pred, relevant)
        hit = next((f for f in pred if f in relevant), None)
        # 首个命中的位次
        rank = (pred.index(hit) + 1) if hit else None

        per_query.append({
            "query": query,
            "relevant": sorted(relevant),
            "pred_top": pred[: args.top],
            "recall@1": rec[1], "recall@3": rec[3], "recall@5": rec[5],
            "mrr": rr, "first_hit_rank": rank,
        })
        table_rows.append([
            query[:34],
            "/".join(sorted(relevant)),
            f"{rec[1]:.2f}",
            f"{rec[3]:.2f}",
            f"{rec[5]:.2f}",
            f"{rr:.3f}",
        ])

    render_table(f"per-query recall@k / MRR  (top 融合 {args.top})",
                 ["query", "relevant", "r@1", "r@3", "r@5", "MRR"], table_rows)

    # 汇总
    n = len(per_query)
    if n:
        agg = []
        for k in ks:
            vals = [q[f"recall@{k}"] for q in per_query if q[f"recall@{k}"] is not None]
            agg.append(f"recall@{k} = {sum(vals) / len(vals):.3f}" if vals else "recall@k: n/a")
        mrr_vals = [q["mrr"] for q in per_query]
        print(f"\n=== 汇总（{n} 条）===")
        print("  平均 " + " | ".join(agg))
        print(f"  平均 MRR = {sum(mrr_vals) / n:.3f}")
        miss = [q for q in per_query if q["first_hit_rank"] is None]
        print(f"  完全没召回到相关文件：{len(miss)}/{n}"
              + (f"  ← {[q['query'][:24] for q in miss]}" if miss else ""))

    out_dir = ensure_results_dir("retrieval")
    out = out_dir / f"{stamp()}.jsonl"
    write_jsonl(out, per_query)
    print(f"\n结果已写：{out}")


if __name__ == "__main__":
    main()
