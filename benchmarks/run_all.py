import json
import os
import sys

from benchmarks.experiments import (
    exp1_ef_search_vs_recall_latency,
    exp2_m_vs_memory_recall,
    exp3_crossover_hnsw_vs_flat,
    exp4_pre_vs_post_filter,
    exp5_build_time
)


def run_all_benchmarks():
    print("=" * 70)
    print("      VecDB Comprehensive Benchmark Suite Execution (SIFT-128)")
    print("=" * 70)

    output_charts_dir = os.path.join("benchmarks", "charts")
    os.makedirs(output_charts_dir, exist_ok=True)

    exp1_data = exp1_ef_search_vs_recall_latency.run_experiment(output_charts_dir)
    exp2_data = exp2_m_vs_memory_recall.run_experiment(output_charts_dir)
    exp3_data = exp3_crossover_hnsw_vs_flat.run_experiment(output_charts_dir)
    exp4_data = exp4_pre_vs_post_filter.run_experiment(output_charts_dir)
    exp5_data = exp5_build_time.run_experiment(output_charts_dir)

    all_results = {
        "exp1_ef_search": exp1_data,
        "exp2_m_params": exp2_data,
        "exp3_crossover": exp3_data,
        "exp4_filtering": exp4_data,
        "exp5_build_time": exp5_data,
    }

    # Save to JSON
    json_path = os.path.join("benchmarks", "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("                      BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    # Print markdown tables
    print("\n### Experiment 1: ef_search vs Recall@10 vs p95 Latency (M=16, D=128)")
    print("| ef_search | Recall@10 | p95 Latency (ms) |")
    print("|---|---|---|")
    for row in exp1_data:
        print(f"| {row['ef_search']} | {row['recall']:.4f} | {row['p95_latency_ms']:.3f} |")

    print("\n### Experiment 2: M vs Memory Consumption vs Recall@10 (ef_search=64, D=128)")
    print("| M | Memory (MB) | Recall@10 |")
    print("|---|---|---|")
    for row in exp2_data:
        print(f"| {row['M']} | {row['memory_mb']:.2f} | {row['recall']:.4f} |")

    print("\n### Experiment 3: Crossover Point — QPS vs Dataset Size N (D=128)")
    print("| Dataset Size (N) | Flat QPS | IVF QPS | HNSW QPS |")
    print("|---|---|---|---|")
    for row in exp3_data:
        print(f"| {row['n']} | {row['flat_qps']:.1f} | {row['ivf_qps']:.1f} | {row['hnsw_qps']:.1f} |")

    print("\n### Experiment 4: Pre-filter vs Post-filter Recall Degradation (D=128)")
    print("| Selectivity (%) | Pre-Filter Recall@10 | Post-Filter Recall@10 |")
    print("|---|---|---|")
    for row in exp4_data:
        print(f"| {row['selectivity_percent']}% | {row['pre_filter_recall']:.4f} | {row['post_filter_recall']:.4f} |")

    print("\n### Experiment 5: Index Build Time vs Dataset Size N (D=128)")
    print("| Dataset Size (N) | HNSW Build Time (s) | IVF Build Time (s) |")
    print("|---|---|---|")
    for row in exp5_data:
        print(f"| {row['n']} | {row['hnsw_build_time_sec']:.3f} | {row['ivf_build_time_sec']:.3f} |")

    print("\n" + "=" * 70)
    print(f"Benchmark execution complete. Saved charts to '{output_charts_dir}'")
    print(f"Saved JSON results to '{json_path}'")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
