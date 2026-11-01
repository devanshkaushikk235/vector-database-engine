import numpy as np
import pytest

from vecdb.core.distance import DistanceMetric, compute_distance, compute_distances, compute_batch_distances
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex


def test_distance_metrics():
    u = np.array([1.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0], dtype=np.float32)
    
    # Cosine distance between orthogonal vectors = 1.0
    assert pytest.approx(compute_distance(u, v, DistanceMetric.COSINE), 1e-5) == 1.0
    # Euclidean distance between (1,0) and (0,1) = sqrt(2)
    assert pytest.approx(compute_distance(u, v, DistanceMetric.EUCLIDEAN), 1e-5) == np.sqrt(2)
    # Manhattan distance = 2.0
    assert pytest.approx(compute_distance(u, v, DistanceMetric.MANHATTAN), 1e-5) == 2.0
    # Dot product distance = -0.0
    assert pytest.approx(compute_distance(u, v, DistanceMetric.DOT_PRODUCT), 1e-5) == 0.0


def test_vector_store_crud():
    store = VectorStore(dimension=3)
    store.add("v1", [1.0, 2.0, 3.0], {"category": "A"})
    store.add("v2", [4.0, 5.0, 6.0], {"category": "B"})

    assert len(store) == 2
    assert "v1" in store
    assert "v2" in store

    vec, payload = store.get("v1")
    np.testing.assert_array_almost_equal(vec, [1.0, 2.0, 3.0])
    assert payload["category"] == "A"

    # Soft delete
    assert store.delete("v1", soft=True)
    assert "v1" not in store
    assert len(store) == 1
    assert store.total_count == 2
    assert store.tombstone_count == 1

    # Compact
    purged = store.compact()
    assert purged == ["v1"]
    assert store.total_count == 1
    assert store.tombstone_count == 0


def test_flat_index_knn_range_batch():
    store = VectorStore(dimension=2)
    store.add("v1", [1.0, 0.0], {"name": "east"})
    store.add("v2", [0.0, 1.0], {"name": "north"})
    store.add("v3", [-1.0, 0.0], {"name": "west"})
    store.add("v4", [0.8, 0.2], {"name": "east-northeast"})

    index = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)

    # kNN Search
    query = [1.0, 0.1]
    results = index.search(query, k=2)
    assert len(results) == 2
    assert results[0][0] in ["v1", "v4"]

    # Filtered search
    filtered_results = index.search(query, k=2, filter_ids={"v2", "v3"})
    assert len(filtered_results) == 2
    assert set(res[0] for res in filtered_results) == {"v2", "v3"}

    # Range Search
    range_res = index.range_search([1.0, 0.0], radius=0.3)
    assert len(range_res) >= 1
    assert range_res[0][0] == "v1"

    # Batch Search
    batch_queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    batch_res = index.batch_search(batch_queries, k=1)
    assert len(batch_res) == 2
    assert batch_res[0][0][0] == "v1"
    assert batch_res[1][0][0] == "v2"
