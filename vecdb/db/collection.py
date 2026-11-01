import os
import time
import threading
from typing import Any
import numpy as np

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.query_engine import QueryEngine, FilterStrategy, IndexType
from vecdb.db.wal import WriteAheadLog
from vecdb.db.snapshot import SnapshotManager


class Collection:
    """Named collection namespace encapsulating index, metadata, WAL, snapshot, and disk file lifecycle.
    
    Thread-safety model (readers-writer pattern at collection level):
      • search(), get(), get_stats()   → read lock  (concurrent reads allowed)
      • insert(), upsert(), delete()   → write lock (exclusive)
      • compact(), rebuild_index()     → write lock (exclusive)
      • create_snapshot()              → write lock (exclusive)
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        storage_dir: str = "./vdb_data",
        m_subspaces: int | None = None,
    ):
        """Initialize Collection.
        
        Args:
            name: Collection namespace name.
            dimension: Vector dimension size.
            metric: Distance metric strategy.
            storage_dir: Directory for storing WAL and snapshot files.
            m_subspaces: Number of PQ sub-spaces.  Must evenly divide dimension.
                         Defaults to min(8, dimension // 16) if not specified.
                         Raise ValueError if dimension % m_subspaces != 0.
        """
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.storage_dir = os.path.join(storage_dir, name)
        os.makedirs(self.storage_dir, exist_ok=True)

        # ── PQ subspace validation ─────────────────────────────────────────
        if m_subspaces is None:
            target = min(8, max(1, dimension // 16))
            m_subspaces = 1
            for m in range(target, 0, -1):
                if dimension % m == 0:
                    m_subspaces = m
                    break
        if dimension % m_subspaces != 0:
            raise ValueError(
                f"m_subspaces={m_subspaces} does not evenly divide dimension={dimension}. "
                f"Choose m_subspaces such that {dimension} % m_subspaces == 0. "
                f"Valid values: {[d for d in range(1, dimension + 1) if dimension % d == 0]}"
            )
        self.m_subspaces = m_subspaces

        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.last_rebuild_time = "Never"
        self.last_build_duration_sec = 0.0

        self.wal_path = os.path.join(self.storage_dir, "wal.jsonl")
        self.snapshot_path = os.path.join(self.storage_dir, "snapshot.pkl")

        self.vector_store = VectorStore(dimension=dimension)
        self.query_engine = QueryEngine(
            self.vector_store,
            metric=metric,
            m_subspaces=m_subspaces,
        )
        self.wal = WriteAheadLog(self.wal_path)

        # ── Collection-level readers-writer lock ───────────────────────────
        self._rw_lock = threading.Lock()       # guards _reader_count
        self._reader_count: int = 0
        self._write_event = threading.Lock()   # held by readers (collectively); writers wait for it

        # Automatic state recovery upon instantiation (single-threaded at startup)
        self.recover()

    # ── RW-lock helpers ──────────────────────────────────────────────────

    def _acquire_read(self) -> None:
        with self._rw_lock:
            self._reader_count += 1
            if self._reader_count == 1:
                self._write_event.acquire()

    def _release_read(self) -> None:
        with self._rw_lock:
            self._reader_count -= 1
            if self._reader_count == 0:
                self._write_event.release()

    # ── Public write operations ──────────────────────────────────────────

    def insert(
        self,
        vector_id: str,
        vector: np.ndarray | list[float],
        payload: dict[str, Any] | None = None
    ) -> None:
        """Insert vector with payload into collection, logging to WAL first."""
        vec_list = [float(x) for x in vector]
        # WAL write happens outside the collection RW lock because the WAL
        # has its own internal lock + fsync guarantee.
        self.wal.log("insert", vector_id, vec_list, payload)
        with self._write_event:
            self.query_engine.insert(vector_id, vector, payload)

    def upsert(
        self,
        vector_id: str,
        vector: np.ndarray | list[float],
        payload: dict[str, Any] | None = None
    ) -> None:
        """Upsert vector entry."""
        vec_list = [float(x) for x in vector]
        self.wal.log("upsert", vector_id, vec_list, payload)
        with self._write_event:
            self.query_engine.insert(vector_id, vector, payload)

    def delete(self, vector_id: str, soft: bool = True) -> bool:
        """Delete vector entry by ID."""
        self.wal.log("delete", vector_id)
        with self._write_event:
            return self.query_engine.delete(vector_id, soft=soft)

    def create_snapshot(self) -> None:
        """Persist full collection state to binary snapshot and truncate WAL."""
        with self._write_event:
            SnapshotManager.create_snapshot(
                self.snapshot_path, self.vector_store, self.query_engine.hnsw_index
            )
            self.wal.truncate()

    def compact(self) -> None:
        """Purge tombstoned items, rebuild indices, and create snapshot."""
        t0 = time.time()
        with self._write_event:
            self.query_engine.compact()
        self.create_snapshot()
        self.last_build_duration_sec = round(time.time() - t0, 3)
        self.last_rebuild_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def rebuild_index(self) -> None:
        """Trigger index rebuild."""
        t0 = time.time()
        with self._write_event:
            self.query_engine.hnsw_index.rebuild()
            self.query_engine.ivf_index.rebuild()
        self.last_build_duration_sec = round(time.time() - t0, 3)
        self.last_rebuild_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── Public read operations ────────────────────────────────────────────

    def get(self, vector_id: str) -> dict[str, Any] | None:
        """Retrieve vector payload by ID."""
        self._acquire_read()
        try:
            data = self.vector_store.get(vector_id)
        finally:
            self._release_read()
        if data is None:
            return None
        vec, payload = data
        return {"id": vector_id, "vector": vec.tolist(), "payload": payload}

    def search(
        self,
        query: np.ndarray | list[float],
        k: int = 10,
        filter_query: dict[str, Any] | None = None,
        filter_strategy: FilterStrategy = FilterStrategy.PRE_FILTER,
        index_type: IndexType = IndexType.HNSW,
        oversample_factor: float = 3.0,
        ef_search: int | None = None,
        n_probe: int | None = None
    ) -> list[dict[str, Any]]:
        """Execute similarity search against the collection."""
        self._acquire_read()
        try:
            return self.query_engine.search(
                query=query,
                k=k,
                filter_query=filter_query,
                filter_strategy=filter_strategy,
                index_type=index_type,
                oversample_factor=oversample_factor,
                ef_search=ef_search,
                n_probe=n_probe,
            )
        finally:
            self._release_read()

    # ── Recall Verification ───────────────────────────────────────────────

    def verify_recall(
        self,
        query: np.ndarray | list[float],
        k: int = 10,
        index_type: IndexType = IndexType.HNSW,
        filter_query: dict[str, Any] | None = None,
        filter_strategy: FilterStrategy = FilterStrategy.PRE_FILTER,
        ef_search: int | None = None,
        n_probe: int | None = None,
    ) -> dict[str, Any]:
        """Compare ANN index results against exact Flat search to compute Recall@K.
        
        Runs two searches under the read lock to ensure a consistent snapshot:
          1. Exact Flat (brute-force) search → ground truth set G
          2. ANN search (HNSW / IVF / PQ)  → approximate set A
        
        Recall@K = |A ∩ G| / min(K, |G|)
        
        Returns:
            dict with keys:
                recall_at_k     float  — fraction of ground-truth top-k found by ANN
                k               int    — requested k
                exact_count     int    — actual ground-truth results
                ann_count       int    — actual ANN results
                exact_latency_ms float
                ann_latency_ms  float
                exact_results   list   — ground-truth ranked results
                ann_results     list   — ANN ranked results
                index_type      str
        """
        self._acquire_read()
        try:
            q = np.asarray(query, dtype=np.float32)

            # 1. Exact ground-truth search
            t0 = time.perf_counter()
            exact_results = self.query_engine.search(
                query=q,
                k=k,
                filter_query=filter_query,
                filter_strategy=filter_strategy,
                index_type=IndexType.FLAT,
            )
            exact_latency_ms = (time.perf_counter() - t0) * 1000.0

            # 2. ANN search
            t0 = time.perf_counter()
            ann_results = self.query_engine.search(
                query=q,
                k=k,
                filter_query=filter_query,
                filter_strategy=filter_strategy,
                index_type=index_type,
                ef_search=ef_search,
                n_probe=n_probe,
            )
            ann_latency_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            self._release_read()

        # Compute Recall@K
        exact_ids = {r["id"] for r in exact_results}
        ann_ids   = {r["id"] for r in ann_results}
        denom = min(k, len(exact_ids))
        recall = len(ann_ids & exact_ids) / denom if denom > 0 else 0.0

        return {
            "recall_at_k": round(recall, 4),
            "k": k,
            "exact_count": len(exact_results),
            "ann_count": len(ann_results),
            "exact_latency_ms": round(exact_latency_ms, 3),
            "ann_latency_ms": round(ann_latency_ms, 3),
            "exact_results": exact_results,
            "ann_results": ann_results,
            "index_type": index_type.value,
        }

    # ── Recovery ─────────────────────────────────────────────────────────

    def recover(self) -> None:
        """Recover state: load latest snapshot then replay remaining WAL entries."""
        SnapshotManager.load_snapshot(
            self.snapshot_path, self.vector_store, self.query_engine.hnsw_index
        )
        entries = self.wal.replay()
        for entry in entries:
            op  = entry.get("op")
            vid = entry.get("id")
            if op in ("insert", "upsert"):
                vec     = entry.get("vector")
                payload = entry.get("payload")
                if vec is not None and vid is not None:
                    self.query_engine.insert(vid, vec, payload)
            elif op == "delete":
                if vid is not None:
                    self.query_engine.delete(vid, soft=True)

    # ── Metadata & stats ─────────────────────────────────────────────────

    def infer_payload_schema(self) -> dict[str, str]:
        """Infer JSON payload schema types from active vectors."""
        schema: dict[str, str] = {}
        active_ids = self.vector_store.get_all_ids(include_tombstones=False)[:50]
        for vid in active_ids:
            payload = self.vector_store.get_payload(vid)
            if payload:
                for k, v in payload.items():
                    if k not in schema:
                        schema[k] = type(v).__name__
        return schema

    def get_storage_details(self) -> dict[str, Any]:
        """Get disk storage details for WAL, snapshots, and segments."""
        wal_bytes = os.path.getsize(self.wal_path) if os.path.exists(self.wal_path) else 0
        snapshot_bytes = os.path.getsize(self.snapshot_path) if os.path.exists(self.snapshot_path) else 0
        total_bytes = wal_bytes + snapshot_bytes

        snapshot_mtime = "None"
        if os.path.exists(self.snapshot_path):
            snapshot_mtime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(self.snapshot_path))
            )

        wal_entries = self.wal.replay()

        return {
            "wal_bytes": wal_bytes,
            "wal_entries_pending": len(wal_entries),
            "snapshot_bytes": snapshot_bytes,
            "snapshot_mtime": snapshot_mtime,
            "total_bytes": total_bytes,
            "segment_count": 1,
            "segments": [
                {
                    "id": f"seg_001_{self.name}",
                    "vector_count": len(self.vector_store),
                    "size_bytes": total_bytes,
                    "status": "compacting" if self.vector_store.tombstone_count > 0 else "active"
                }
            ]
        }

    def get_stats(self) -> dict[str, Any]:
        """Return collection metadata and usage metrics."""
        self._acquire_read()
        try:
            storage = self.get_storage_details()
            pq_status = {
                "m_subspaces": self.m_subspaces,
                "subspace_dim": self.dimension // self.m_subspaces,
                "is_trained": self.query_engine.pq_index.is_trained,
                "vectors_until_train": max(
                    0, self.query_engine.pq_index.min_train_size - len(self.vector_store)
                ),
            }
            return {
                "name": self.name,
                "dimension": self.dimension,
                "metric": self.metric.value,
                "created_at": self.created_at,
                "total_vectors": len(self.vector_store),
                "tombstoned_vectors": self.vector_store.tombstone_count,
                "storage_dir": self.storage_dir,
                "hnsw_config": {
                    "M": self.query_engine.hnsw_index.M,
                    "ef_construction": self.query_engine.hnsw_index.ef_construction,
                    "ef_search": self.query_engine.hnsw_index.ef_search,
                    "max_layer": self.query_engine.hnsw_index.max_layer
                },
                "ivf_config": {
                    "n_list": self.query_engine.ivf_index.n_list,
                    "n_probe": self.query_engine.ivf_index.n_probe,
                    "is_trained": self.query_engine.ivf_index.is_trained
                },
                "pq_config": pq_status,
                "payload_schema": self.infer_payload_schema(),
                "storage": storage,
                "last_rebuild_time": self.last_rebuild_time,
                "last_build_duration_sec": self.last_build_duration_sec
            }
        finally:
            self._release_read()
