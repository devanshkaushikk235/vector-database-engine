import os
import numpy as np
import matplotlib.pyplot as plt

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.query_engine import QueryEngine, FilterStrategy, IndexType
from benchmarks.dataset_loader import get_sift_128_dataset


def run_experiment(output_dir: str = "benchmarks/charts") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    print("Running Experiment 4: Pre-filter vs Post-filter Degradation...")

    base_vecs, query_vecs, payloads = get_sift_128_dataset(num_vectors=1000, num_queries=20, dimension=128)

    store = VectorStore(dimension=128)
    engine = QueryEngine(store, metric=DistanceMetric.EUCLIDEAN, hnsw_kwargs={"M": 16, "ef_construction": 64})

    for i, vec in enumerate(base_vecs):
        vid = f"v_{i}"
        engine.insert(vid, vec, payloads[i])

    k = 10
    selectivities = [0.10, 0.25, 0.50, 0.75, 0.90]
    pre_recalls = []
    post_recalls = []
    results_table = []

    for sel in selectivities:
        filter_q = {"selectivity_score": {"lte": sel}}

        # Ground truth: Exact Flat search with pre-filtering
        ground_truth = []
        for q in query_vecs:
            gt_res = engine.search(q, k=k, filter_query=filter_q, filter_strategy=FilterStrategy.PRE_FILTER, index_type=IndexType.FLAT)
            ground_truth.append(set(r["id"] for r in gt_res))

        # 1. HNSW Pre-filter Search
        hits_pre = 0
        for idx, q in enumerate(query_vecs):
            res_pre = engine.search(q, k=k, filter_query=filter_q, filter_strategy=FilterStrategy.PRE_FILTER, index_type=IndexType.HNSW, ef_search=64)
            found_ids = set(r["id"] for r in res_pre)
            hits_pre += len(found_ids.intersection(ground_truth[idx]))
        avg_recall_pre = hits_pre / (len(query_vecs) * k)

        # 2. HNSW Post-filter Search
        hits_post = 0
        for idx, q in enumerate(query_vecs):
            res_post = engine.search(q, k=k, filter_query=filter_q, filter_strategy=FilterStrategy.POST_FILTER, index_type=IndexType.HNSW, ef_search=64, oversample_factor=3.0)
            found_ids = set(r["id"] for r in res_post)
            hits_post += len(found_ids.intersection(ground_truth[idx]))
        avg_recall_post = hits_post / (len(query_vecs) * k)

        pre_recalls.append(avg_recall_pre)
        post_recalls.append(avg_recall_post)

        results_table.append({
            "selectivity_percent": int(sel * 100),
            "pre_filter_recall": round(avg_recall_pre, 4),
            "post_filter_recall": round(avg_recall_post, 4)
        })

    # Plot Chart
    plt.figure(figsize=(8, 5))
    x_percentages = [int(s * 100) for s in selectivities]
    plt.plot(x_percentages, pre_recalls, marker="o", linewidth=2, color="tab:blue", label="Pre-Filter (HNSW)")
    plt.plot(x_percentages, post_recalls, marker="s", linestyle="--", linewidth=2, color="tab:orange", label="Post-Filter (HNSW oversample 3x)")

    plt.xlabel("Filter Selectivity (% Pass Rate)")
    plt.ylabel("Recall@10")
    plt.ylim([0.0, 1.05])
    plt.title("Experiment 4: Pre-filter vs Post-filter Recall Degradation (D=128)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    chart_path = os.path.join(output_dir, "exp4_pre_vs_post_filter.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Experiment 4 completed. Saved chart to {chart_path}")
    return results_table


if __name__ == "__main__":
    run_experiment()
