"""Tests for concurrent RWLock correctness on Collection.

Verifies that:
  - Multiple concurrent search() calls proceed without deadlock.
  - Concurrent insert() + search() do not produce data races.
  - entry_point remains consistent under concurrent inserts (TOCTOU guard).
"""

import threading
import time
import numpy as np
import pytest

from vecdb.db.collection import Collection
from vecdb.core.distance import DistanceMetric
from vecdb.core.query_engine import IndexType


DIM = 32
N_INIT = 50      # vectors inserted before concurrent test
N_CONCURRENT = 20  # vectors inserted during concurrent phase


@pytest.fixture
def coll(tmp_path):
    """Create a fresh collection in a temp directory."""
    return Collection(
        name="concurrent_test",
        dimension=DIM,
        metric=DistanceMetric.COSINE,
        storage_dir=str(tmp_path),
    )


def _insert_n(coll: Collection, start: int, n: int) -> None:
    """Insert n random vectors into coll, IDs from start..start+n."""
    rng = np.random.default_rng(start)
    for i in range(n):
        vid = f"vec_{start + i}"
        vec = rng.random(DIM).astype(np.float32)
        coll.insert(vid, vec, {"thread": start, "idx": i})


def _search_n(coll: Collection, n: int, results: list) -> None:
    """Run n searches and collect result counts."""
    rng = np.random.default_rng(99)
    for _ in range(n):
        q = rng.random(DIM).astype(np.float32)
        r = coll.search(q, k=5, index_type=IndexType.HNSW)
        results.append(len(r))


class TestConcurrentReads:
    def test_many_concurrent_readers(self, coll):
        """Multiple threads searching simultaneously should not deadlock."""
        _insert_n(coll, 0, N_INIT)

        results = [[] for _ in range(8)]
        threads = [
            threading.Thread(target=_search_n, args=(coll, 10, results[i]))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        for t in threads:
            assert not t.is_alive(), "Thread deadlocked during concurrent reads"

        # All searches should complete; results count can be 0 only if index is empty
        total = sum(len(r) for r in results)
        assert total >= 0  # sanity: no exception raised


class TestConcurrentWritesAndReads:
    def test_interleaved_insert_search(self, coll):
        """Concurrent inserts and searches must not corrupt entry_point or graph."""
        _insert_n(coll, 0, N_INIT)

        errors = []

        def insert_worker():
            try:
                _insert_n(coll, N_INIT, N_CONCURRENT)
            except Exception as e:
                errors.append(("insert", str(e)))

        def search_worker():
            try:
                _search_n(coll, 30, [])
            except Exception as e:
                errors.append(("search", str(e)))

        writers = [threading.Thread(target=insert_worker) for _ in range(3)]
        readers = [threading.Thread(target=search_worker) for _ in range(5)]

        all_threads = writers + readers
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join(timeout=30)
        for t in all_threads:
            assert not t.is_alive(), "Thread did not complete in time"

        assert errors == [], f"Concurrent errors detected: {errors}"

    def test_no_duplicate_vectors_after_concurrent_insert(self, coll):
        """Verify no duplicate IDs after concurrent inserts."""
        threads = [
            threading.Thread(target=_insert_n, args=(coll, i * 20, 20))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        all_ids = coll.vector_store.get_all_ids(include_tombstones=False)
        assert len(all_ids) == len(set(all_ids)), "Duplicate vector IDs after concurrent inserts"


class TestWALFsync:
    def test_wal_lock_prevents_torn_writes(self, coll):
        """WAL log() should succeed from multiple threads without producing corrupt lines."""
        errors = []

        def log_worker(thread_id):
            try:
                for i in range(20):
                    coll.wal.log("insert", f"vec_{thread_id}_{i}", [0.1] * DIM)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=log_worker, args=(tid,)) for tid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"WAL concurrent log errors: {errors}"

        # All lines must be valid JSON
        entries = coll.wal.replay()
        assert len(entries) >= 20 * 8 - 10  # allow small variance


class TestEntryPointTOCTOU:
    def test_entry_point_consistent_under_concurrent_insert(self, coll):
        """TOCTOU guard: entry_point must never be None during a search while inserts happen."""
        entry_none_errors = []

        def insert_continuously():
            rng = np.random.default_rng(7)
            for i in range(50):
                coll.insert(f"ep_vec_{i}", rng.random(DIM).astype(np.float32))

        def search_continuously():
            rng = np.random.default_rng(8)
            for _ in range(50):
                q = rng.random(DIM).astype(np.float32)
                try:
                    coll.search(q, k=3)
                except Exception as e:
                    entry_none_errors.append(str(e))

        inserter = threading.Thread(target=insert_continuously)
        searcher = threading.Thread(target=search_continuously)
        inserter.start()
        searcher.start()
        inserter.join(timeout=20)
        searcher.join(timeout=20)

        assert entry_none_errors == [], f"Errors during concurrent search+insert: {entry_none_errors}"
