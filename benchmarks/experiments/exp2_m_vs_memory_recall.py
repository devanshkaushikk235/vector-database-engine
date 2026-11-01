import os
import sys
import psutil
import matplotlib.pyplot as plt

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.hnsw import HNSWIndex
from benchmarks.dataset_loader import get_sift_128_dataset


def run_experiment(output_dir: str = "benchmarks/charts") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    print("Running Experiment 2: M vs Memory Consumption vs Recall@10...")

    base_vecs, query_vecs, _ = get_sift_128_dataset(num_vectors=1000, num_queries=20, dimension=128)

    k = 10
    # Ground truth
    temp_store = VectorStore(dimension=128)
    flat = FlatIndex(temp_store, metric=DistanceMetric.EUCLIDEAN)
    for i, vec in enumerate(base_vecs):
        temp_store.add(f"v_{i}", vec)

    ground_truth = []
    for q in query_vecs:
        gt_res = flat.search(q, k=k)
        ground_truth.append(set(vid for vid, _ in gt_res))

    M_values = [4, 8, 16, 32, 64]
    memory_footprints_mb = []
    recalls = []
    results_table = []

    process = psutil.Process(os.getpid())

    for M in M_values:
        store = VectorStore(dimension=128)
        hnsw = HNSWIndex(store, metric=DistanceMetric.EUCLIDEAN, M=M, ef_construction=64)

        mem_before = process.memory_info().rss / (1024 * 1024)
        for i, vec in enumerate(base_vecs):
            vid = f"v_{i}"
            store.add(vid, vec)
            hnsw.insert(vid)
        mem_after = process.memory_info().rss / (1024 * 1024)

        # Estimate memory usage for graph structure
        # Total edges across all layers
        total_edges = sum(len(neighbors) for layer in hnsw.graph for neighbors in layer.values())
        # Pointer & graph overhead estimate: base vector store + edges
        graph_mem_estimate_mb = (total_edges * 8) / (1024 * 1024) + (len(base_vecs) * 128 * 4) / (1024 * 1024)

        hits = 0
        for idx, q in enumerate(query_vecs):
            res = hnsw.search(q, k=k, ef_search=64)
            found_ids = set(vid for vid, _ in res)
            hits += len(found_ids.intersection(ground_truth[idx]))

        avg_recall = hits / (len(query_vecs) * k)

        recalls.append(avg_recall)
        memory_footprints_mb.append(round(graph_mem_estimate_mb, 2))

        results_table.append({
            "M": M,
            "memory_mb": round(graph_mem_estimate_mb, 2),
            "recall": round(avg_recall, 4)
        })

    # Plot Chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:green"
    ax1.set_xlabel("M (Max connections per node)")
    ax1.set_ylabel("Memory Footprint (MB)", color=color)
    ax1.plot(M_values, memory_footprints_mb, color=color, marker="s", linewidth=2, label="Memory (MB)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:purple"
    ax2.set_ylabel("Recall@10", color=color)
    ax2.plot(M_values, recalls, color=color, marker="o", linestyle="--", linewidth=2, label="Recall@10")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim([0.7, 1.02])

    plt.title("Experiment 2: M vs Memory Footprint vs Recall@10 (ef_search=64, D=128)")
    fig.tight_layout()
    chart_path = os.path.join(output_dir, "exp2_m_vs_memory_recall.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Experiment 2 completed. Saved chart to {chart_path}")
    return results_table


if __name__ == "__main__":
    run_experiment()
