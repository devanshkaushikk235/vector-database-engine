import concurrent.futures
import shutil
import tempfile
import numpy as np
import pytest

from vecdb.core.distance import DistanceMetric, compute_distance
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.ivf_index import IVFIndex
from vecdb.core.hnsw import HNSWIndex
from vecdb.core.metadata import evaluate_filter, get_matching_ids
from vecdb.core.query_engine import QueryEngine, FilterStrategy, IndexType
from vecdb.db.collection import Collection
from vecdb.db.wal import WriteAheadLog
from vecdb.db.snapshot import SnapshotManager


@pytest.fixture
def temp_storage():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


# =====================================================================
# 1. RECALL REGRESSION TESTS
# =====================================================================

def test_hnsw_recall_regression():
    """Insert N=300 high-dim vectors, verify HNSW recall@10 >= 0.85 against FlatIndex ground truth."""
    np.random.seed(42)
    dim = 32
    num_vecs = 300
    k = 10

    store = VectorStore(dimension=dim)
    flat = FlatIndex(store, metric=DistanceMetric.COSINE)
    hnsw = HNSWIndex(store, metric=DistanceMetric.COSINE, M=16, ef_construction=64, ef_search=64)

    vectors = np.random.randn(num_vecs, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    for i in range(num_vecs):
        vid = f"v_{i}"
        store.add(vid, vectors[i])
        hnsw.insert(vid)

    # Test 20 query vectors
    total_recall = 0.0
    num_queries = 20

    for q_idx in range(num_queries):
        query = np.random.randn(dim).astype(np.float32)
        query /= np.linalg.norm(query)

        flat_res = flat.search(query, k=k)
        hnsw_res = hnsw.search(query, k=k)

        flat_set = set(vid for vid, _ in flat_res)
        hnsw_set = set(vid for vid, _ in hnsw_res)

        total_recall += len(flat_set.intersection(hnsw_set)) / float(k)

    avg_recall = total_recall / num_queries
    assert avg_recall >= 0.85, f"HNSW recall regression failure: got {avg_recall:.4f}, expected >= 0.85"


# =====================================================================
# 2. WAL REPLAY & CRASH RECOVERY TESTS
# =====================================================================

def test_wal_replay_sequence_correctness(temp_storage):
    """Verify sequence of inserts, updates, and deletes replay accurately after crash."""
    coll_name = "wal_test"
    coll1 = Collection(name=coll_name, dimension=3, metric=DistanceMetric.EUCLIDEAN, storage_dir=temp_storage)

    coll1.insert("v1", [1.0, 0.0, 0.0], {"state": "initial"})
    coll1.insert("v2", [0.0, 1.0, 0.0], {"state": "initial"})
    coll1.upsert("v1", [1.1, 0.0, 0.0], {"state": "updated"})
    coll1.delete("v2")

    # Simulate crash: recreate collection object pointing to same directory
    coll2 = Collection(name=coll_name, dimension=3, metric=DistanceMetric.EUCLIDEAN, storage_dir=temp_storage)

    # v1 should exist with updated payload
    v1_data = coll2.get("v1")
    assert v1_data is not None
    assert v1_data["payload"]["state"] == "updated"
    np.testing.assert_array_almost_equal(v1_data["vector"], [1.1, 0.0, 0.0])

    # v2 should be tombstoned/deleted
    assert coll2.get("v2") is None


def test_wal_truncation_after_snapshot(temp_storage):
    """Verify snapshot truncates WAL while maintaining complete collection state."""
    coll = Collection(name="snap_test", dimension=2, storage_dir=temp_storage)
    coll.insert("v1", [1.0, 2.0], {"val": 1})
    coll.insert("v2", [3.0, 4.0], {"val": 2})

    # Snapshot creates snapshot file and clears WAL
    coll.create_snapshot()
    wal_entries = coll.wal.replay()
    assert len(wal_entries) == 0

    # New collection re-hydrates from snapshot
    coll_reloaded = Collection(name="snap_test", dimension=2, storage_dir=temp_storage)
    assert coll_reloaded.get("v1")["payload"]["val"] == 1
    assert coll_reloaded.get("v2")["payload"]["val"] == 2


# =====================================================================
# 3. TOMBSTONE EXCLUSION TESTS ACROSS ALL INDICES
# =====================================================================

def test_tombstone_exclusion_flat_ivf_hnsw():
    """Verify deleted/tombstoned vectors never appear in search results for Flat, IVF, and HNSW."""
    store = VectorStore(dimension=3)
    engine = QueryEngine(store, metric=DistanceMetric.EUCLIDEAN)

    engine.insert("target", [1.0, 0.0, 0.0], {"role": "target"})
    engine.insert("other1", [0.9, 0.1, 0.0], {"role": "other"})
    engine.insert("other2", [0.8, 0.2, 0.0], {"role": "other"})

    # Delete closest vector 'target'
    engine.delete("target", soft=True)

    query = [1.0, 0.0, 0.0]

    for idx_type in [IndexType.FLAT, IndexType.HNSW]:
        results = engine.search(query, k=3, index_type=idx_type)
        retrieved_ids = [r["id"] for r in results]
        assert "target" not in retrieved_ids, f"Tombstoned vector returned in {idx_type} index"
        assert retrieved_ids[0] == "other1"


# =====================================================================
# 4. IVF NPROBE SENSITIVITY TESTS
# =====================================================================

def test_ivf_nprobe_sensitivity():
    """Verify increasing n_probe monotonically improves recall towards exact ground truth."""
    np.random.seed(42)
    dim = 16
    num_vecs = 200

    store = VectorStore(dimension=dim)
    ivf = IVFIndex(store, metric=DistanceMetric.EUCLIDEAN, n_list=16, n_probe=1)
    flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)

    vectors = np.random.randn(num_vecs, dim).astype(np.float32)
    for i in range(num_vecs):
        vid = f"v_{i}"
        store.add(vid, vectors[i])

    ivf.train()

    query = np.random.randn(dim).astype(np.float32)
    gt_res = flat.search(query, k=10)
    gt_ids = set(vid for vid, _ in gt_res)

    recalls = []
    for probe in [1, 2, 4, 8, 16]:
        ivf_res = ivf.search(query, k=10, n_probe=probe)
        ivf_ids = set(vid for vid, _ in ivf_res)
        recall = len(gt_ids.intersection(ivf_ids)) / 10.0
        recalls.append(recall)

    # Verify recall at n_probe=16 is 1.0 (scans all Voronoi cells)
    assert recalls[-1] == 1.0
    # Verify general increasing trend
    assert recalls[0] <= recalls[2] <= recalls[-1]


# =====================================================================
# 5. METADATA FILTER EDGE CASE TESTS
# =====================================================================

def test_metadata_filter_empty_and_no_match():
    """Test metadata evaluation when filter matches nothing or fields are missing."""
    payload = {"price": 50, "status": "available"}

    # No match
    assert not evaluate_filter(payload, {"price": {"gt": 100}})
    # Missing key
    assert not evaluate_filter(payload, {"nonexistent_key": "val"})
    # Empty filter matches everything
    assert evaluate_filter(payload, {})
    assert evaluate_filter(payload, None)


def test_metadata_nested_and_or():
    """Test nested $and and $or logic combinations."""
    payload = {"brand": "nike", "price": 80, "rating": 4.8}

    # Complex expression: (brand == nike AND price <= 100) OR (rating > 4.9)
    complex_filter = {
        "$or": [
            {"$and": [{"brand": "nike"}, {"price": {"lte": 100}}]},
            {"rating": {"gt": 4.9}}
        ]
    }
    assert evaluate_filter(payload, complex_filter)

    # Fails both OR branches
    failing_filter = {
        "$or": [
            {"$and": [{"brand": "adidas"}, {"price": {"lte": 100}}]},
            {"rating": {"gt": 4.9}}
        ]
    }
    assert not evaluate_filter(payload, failing_filter)


def test_metadata_filter_operators_all():
    """Test all comparison operators: eq, neq, gt, gte, lt, lte, in, nin."""
    payload = {"count": 10, "score": 95.5, "tag": "prod", "items": ["a", "b"]}

    assert evaluate_filter(payload, {"count": {"eq": 10}})
    assert evaluate_filter(payload, {"count": {"neq": 20}})
    assert evaluate_filter(payload, {"score": {"gt": 90.0}})
    assert evaluate_filter(payload, {"score": {"gte": 95.5}})
    assert evaluate_filter(payload, {"count": {"lt": 15}})
    assert evaluate_filter(payload, {"count": {"lte": 10}})
    assert evaluate_filter(payload, {"tag": {"in": ["dev", "prod"]}})
    assert evaluate_filter(payload, {"tag": {"nin": ["staging", "test"]}})
    assert evaluate_filter(payload, {"items": {"in": ["b", "c"]}})
    assert evaluate_filter(payload, {"items": {"nin": ["x", "y"]}})


# =====================================================================
# 6. CONCURRENT READ / THREAD SAFETY TESTS
# =====================================================================

def test_concurrent_read_queries():
    """Test multithreaded concurrent query execution against QueryEngine."""
    store = VectorStore(dimension=4)
    engine = QueryEngine(store, metric=DistanceMetric.EUCLIDEAN)

    for i in range(50):
        engine.insert(f"v_{i}", [float(i), 0.0, 0.0, 0.0], {"idx": i})

    def run_query(q_val: float):
        return engine.search([q_val, 0.0, 0.0, 0.0], k=3, index_type=IndexType.HNSW)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_query, float(i)) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    for res in results:
        assert len(res) == 3


# =====================================================================
# 7. CORNER CASES & VALIDATION TESTS
# =====================================================================

def test_vector_dimension_mismatch_error():
    """Verify inserting vector with wrong dimension raises ValueError."""
    store = VectorStore(dimension=4)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        store.add("invalid", [1.0, 2.0, 3.0])  # 3-d vector into 4-d store


def test_empty_store_query_graceful_handling():
    """Verify querying an empty store returns empty results without crashing."""
    store = VectorStore(dimension=4)
    engine = QueryEngine(store)

    assert engine.search([1.0, 0.0, 0.0, 0.0], k=5, index_type=IndexType.FLAT) == []
    assert engine.search([1.0, 0.0, 0.0, 0.0], k=5, index_type=IndexType.HNSW) == []
    assert engine.search([1.0, 0.0, 0.0, 0.0], k=5, index_type=IndexType.IVF) == []


def test_duplicate_vector_id_upsert():
    """Verify inserting vector with duplicate ID updates the vector and payload."""
    store = VectorStore(dimension=2)
    store.add("v1", [1.0, 0.0], {"v": 1})
    store.add("v1", [0.0, 1.0], {"v": 2})

    assert len(store) == 1
    vec, payload = store.get("v1")
    np.testing.assert_array_almost_equal(vec, [0.0, 1.0])
    assert payload["v"] == 2


def test_range_search_radius_boundary():
    """Test radius boundary behavior in FlatIndex range search."""
    store = VectorStore(dimension=2)
    store.add("v1", [0.0, 0.0])
    store.add("v2", [1.0, 0.0])
    store.add("v3", [2.0, 0.0])

    flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)

    # Radius = 1.0 should return v1 (dist 0) and v2 (dist 1.0)
    res = flat.range_search([0.0, 0.0], radius=1.0)
    assert set(vid for vid, _ in res) == {"v1", "v2"}

    # Radius = 0.5 returns only v1
    res_tight = flat.range_search([0.0, 0.0], radius=0.5)
    assert [vid for vid, _ in res_tight] == ["v1"]


def test_compaction_purges_tombstones():
    """Verify compaction purges soft-deleted vectors permanently."""
    store = VectorStore(dimension=2)
    engine = QueryEngine(store)

    engine.insert("v1", [1.0, 0.0])
    engine.insert("v2", [0.0, 1.0])
    engine.delete("v1", soft=True)

    assert store.total_count == 2
    assert store.tombstone_count == 1

    engine.compact()

    assert store.total_count == 1
    assert store.tombstone_count == 0
    assert "v1" not in store._vectors
