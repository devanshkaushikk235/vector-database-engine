import os
import time
import numpy as np
import matplotlib.pyplot as plt

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.hnsw import HNSWIndex
from benchmarks.dataset_loader import get_sift_128_dataset


def run_experiment(output_dir: str = "benchmarks/charts") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    print("Running Experiment 1: ef_search vs Recall@10 vs p95 Latency...")

    base_vecs, query_vecs, _ = get_sift_128_dataset(num_vectors=1000, num_queries=20, dimension=128)

    store = VectorStore(dimension=128)
    flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)
    hnsw = HNSWIndex(store, metric=DistanceMetric.EUCLIDEAN, M=16, ef_construction=64)

    for i, vec in enumerate(base_vecs):
        vid = f"v_{i}"
        store.add(vid, vec)
        hnsw.insert(vid)

    # Compute ground truth using FlatIndex
    k = 10
    ground_truth = []
    for q in query_vecs:
        gt_res = flat.search(q, k=k)
        ground_truth.append(set(vid for vid, _ in gt_res))

    ef_search_values = [16, 32, 64, 128, 256, 512]
    recalls = []
    p95_latencies = []
    results_table = []

    for ef in ef_search_values:
        latencies = []
        hits = 0

        for idx, q in enumerate(query_vecs):
            t0 = time.perf_counter()
            res = hnsw.search(q, k=k, ef_search=ef)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms
            found_ids = set(vid for vid, _ in res)
            hits += len(found_ids.intersection(ground_truth[idx]))

        avg_recall = hits / (len(query_vecs) * k)
        p95_lat = float(np.percentile(latencies, 95))

        recalls.append(avg_recall)
        p95_latencies.append(p95_lat)
        results_table.append({
            "ef_search": ef,
            "recall": round(avg_recall, 4),
            "p95_latency_ms": round(p95_lat, 3)
        })

    # Plot Chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("ef_search")
    ax1.set_ylabel("Recall@10", color=color)
    ax1.plot(ef_search_values, recalls, color=color, marker="o", linewidth=2, label="Recall@10")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim([0.7, 1.02])
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("p95 Latency (ms)", color=color)
    ax2.plot(ef_search_values, p95_latencies, color=color, marker="s", linestyle="--", linewidth=2, label="p95 Latency")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Experiment 1: ef_search vs Recall@10 vs p95 Latency (M=16, D=128)")
    fig.tight_layout()
    chart_path = os.path.join(output_dir, "exp1_ef_search_vs_recall_latency.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Experiment 1 completed. Saved chart to {chart_path}")
    return results_table


if __name__ == "__main__":
    run_experiment()
