import os
import time
import numpy as np
import matplotlib.pyplot as plt

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.ivf_index import IVFIndex
from vecdb.core.hnsw import HNSWIndex
from benchmarks.dataset_loader import get_sift_128_dataset


def run_experiment(output_dir: str = "benchmarks/charts") -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    print("Running Experiment 5: Index Build Time vs Dataset Size N...")

    dataset_sizes = [200, 500, 1000, 2000, 4000]
    hnsw_build_times = []
    ivf_build_times = []
    results_table = []

    for n in dataset_sizes:
        base_vecs, _, _ = get_sift_128_dataset(num_vectors=n, num_queries=10, dimension=128)

        # HNSW Index Build Time
        store_hnsw = VectorStore(dimension=128)
        hnsw = HNSWIndex(store_hnsw, metric=DistanceMetric.EUCLIDEAN, M=16, ef_construction=64)

        t0 = time.perf_counter()
        for i, vec in enumerate(base_vecs):
            vid = f"v_{i}"
            store_hnsw.add(vid, vec)
            hnsw.insert(vid)
        t1 = time.perf_counter()
        hnsw_time = t1 - t0

        # IVF Index Build Time
        store_ivf = VectorStore(dimension=128)
        ivf = IVFIndex(store_ivf, metric=DistanceMetric.EUCLIDEAN, n_list=32)
        for i, vec in enumerate(base_vecs):
            store_ivf.add(f"v_{i}", vec)
        
        t0 = time.perf_counter()
        ivf.train()
        t1 = time.perf_counter()
        ivf_time = t1 - t0

        hnsw_build_times.append(round(hnsw_time, 3))
        ivf_build_times.append(round(ivf_time, 3))

        results_table.append({
            "n": n,
            "hnsw_build_time_sec": round(hnsw_time, 3),
            "ivf_build_time_sec": round(ivf_time, 3)
        })

    # Plot Chart
    plt.figure(figsize=(8, 5))
    plt.plot(dataset_sizes, hnsw_build_times, marker="o", linewidth=2, color="tab:blue", label="HNSW Index Build")
    plt.plot(dataset_sizes, ivf_build_times, marker="^", linewidth=2, color="tab:green", label="IVF Index Train")

    plt.xlabel("Dataset Size (N vectors)")
    plt.ylabel("Index Build Time (seconds)")
    plt.title("Experiment 5: Index Build Time vs Dataset Size N (D=128)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    chart_path = os.path.join(output_dir, "exp5_build_time.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Experiment 5 completed. Saved chart to {chart_path}")
    return results_table


if __name__ == "__main__":
    run_experiment()
