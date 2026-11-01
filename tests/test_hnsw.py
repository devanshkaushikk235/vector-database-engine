import numpy as np
import pytest

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.hnsw import HNSWIndex


def test_hnsw_construction_and_recall():
    np.random.seed(42)
    dim = 16
    num_vectors = 200

    store = VectorStore(dimension=dim)
    hnsw = HNSWIndex(store, metric=DistanceMetric.EUCLIDEAN, M=8, ef_construction=32, ef_search=32)
    flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)

    # Generate random vectors
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    for i in range(num_vectors):
        vid = f"vec_{i}"
        store.add(vid, vectors[i])
        hnsw.insert(vid)

    # Query recall test
    query = np.random.randn(dim).astype(np.float32)
    k = 10

    flat_results = flat.search(query, k=k)
    hnsw_results = hnsw.search(query, k=k)

    flat_ids = set(vid for vid, _ in flat_results)
    hnsw_ids = set(vid for vid, _ in hnsw_results)

    recall = len(flat_ids.intersection(hnsw_ids)) / float(k)
    print(f"HNSW Recall@{k}: {recall:.2f}")
    assert recall >= 0.80  # For 200 16-d vectors, recall should be very high


def test_hnsw_tombstone_exclusion():
    store = VectorStore(dimension=4)
    hnsw = HNSWIndex(store, metric=DistanceMetric.COSINE)

    store.add("v1", [1.0, 0.0, 0.0, 0.0])
    store.add("v2", [0.9, 0.1, 0.0, 0.0])
    store.add("v3", [0.0, 1.0, 0.0, 0.0])

    for vid in ["v1", "v2", "v3"]:
        hnsw.insert(vid)

    res1 = hnsw.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert res1[0][0] == "v1"

    # Soft delete v1
    store.delete("v1", soft=True)

    res2 = hnsw.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert "v1" not in [vid for vid, _ in res2]
    assert res2[0][0] == "v2"
