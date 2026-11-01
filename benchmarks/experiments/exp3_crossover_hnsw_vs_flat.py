import os
import time
import numpy as np
import matplotlib.pyplot as plt

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.ivf_index import IVFIndex
from vecdb.core.hnsw import HNSWIndex
from benchmarks.dataset_loader import get_sift_128_dataset


def run_experiment(output_dir: str = "benchmarks/charts") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    print("Running Experiment 3: Crossover Point (Flat vs IVF vs HNSW QPS)...")

    dataset_sizes = [200, 500, 1000, 2000, 4000]
    flat_qps_list = []
    ivf_qps_list = []
    hnsw_qps_list = []
    results_table = []

    for n in dataset_sizes:
        base_vecs, query_vecs, _ = get_sift_128_dataset(num_vectors=n, num_queries=20, dimension=128)

        store = VectorStore(dimension=128)
        flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)
        ivf = IVFIndex(store, metric=DistanceMetric.EUCLIDEAN, n_list=32, n_probe=4)
        hnsw = HNSWIndex(store, metric=DistanceMetric.EUCLIDEAN, M=16, ef_construction=64)

        for i, vec in enumerate(base_vecs):
            vid = f"v_{i}"
            store.add(vid, vec)
            hnsw.insert(vid)

        ivf.train()

        # Benchmark Flat Search QPS
        t0 = time.perf_counter()
        for q in query_vecs:
            flat.search(q, k=10)
        t1 = time.perf_counter()
        flat_qps = len(query_vecs) / (t1 - t0)

        # Benchmark IVF Search QPS
        t0 = time.perf_counter()
        for q in query_vecs:
            ivf.search(q, k=10, n_probe=4)
        t1 = time.perf_counter()
        ivf_qps = len(query_vecs) / (t1 - t0)

        # Benchmark HNSW Search QPS
        t0 = time.perf_counter()
        for q in query_vecs:
            hnsw.search(q, k=10, ef_search=64)
        t1 = time.perf_counter()
        hnsw_qps = len(query_vecs) / (t1 - t0)

        flat_qps_list.append(round(flat_qps, 1))
        ivf_qps_list.append(round(ivf_qps, 1))
        hnsw_qps_list.append(round(hnsw_qps, 1))

        results_table.append({
            "n": n,
            "flat_qps": round(flat_qps, 1),
            "ivf_qps": round(ivf_qps, 1),
            "hnsw_qps": round(hnsw_qps, 1)
        })

    # Plot Chart
    plt.figure(figsize=(8, 5))
    plt.plot(dataset_sizes, flat_qps_list, marker="o", linewidth=2, label="Flat Index (Brute Force)")
    plt.plot(dataset_sizes, ivf_qps_list, marker="^", linewidth=2, label="IVF Index (n_probe=4)")
    plt.plot(dataset_sizes, hnsw_qps_list, marker="s", linewidth=2, label="HNSW Index (M=16, ef=64)")

    plt.xlabel("Dataset Size (N vectors)")
    plt.ylabel("Queries Per Second (QPS)")
    plt.yscale("log")
    plt.title("Experiment 3: Crossover Point — QPS vs Dataset Size N (D=128)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    chart_path = os.path.join(output_dir, "exp3_crossover_hnsw_vs_flat.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Experiment 3 completed. Saved chart to {chart_path}")
    return results_table


if __name__ == "__main__":
    run_experiment()
