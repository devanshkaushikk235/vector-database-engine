from enum import Enum
from typing import Any, Collection
import numpy as np

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.ivf_index import IVFIndex
from vecdb.core.hnsw import HNSWIndex
from vecdb.core.pq_index import ProductQuantizer
from vecdb.core.metadata import evaluate_filter, get_matching_ids


class FilterStrategy(str, Enum):
    """Metadata filtering strategy."""
    PRE_FILTER = "pre_filter"
    POST_FILTER = "post_filter"


class IndexType(str, Enum):
    """Index algorithm selection."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"


class QueryEngine:
    """Query execution engine supporting exact kNN, IVF, HNSW, and PQ search with metadata filtering."""

    def __init__(
        self,
        vector_store: VectorStore,
        metric: DistanceMetric = DistanceMetric.COSINE,
        hnsw_kwargs: dict[str, Any] | None = None,
        ivf_kwargs: dict[str, Any] | None = None,
        m_subspaces: int | None = None,
        pq_k_centroids: int = 256,
    ):
        """Initialize QueryEngine.
        
        Args:
            vector_store:   VectorStore instance.
            metric:         Distance metric strategy.
            hnsw_kwargs:    Custom parameters for HNSW index.
            ivf_kwargs:     Custom parameters for IVF index.
            m_subspaces:    Number of PQ sub-spaces. Must divide vector_store.dimension evenly.
            pq_k_centroids: Number of PQ centroids per sub-space (max 256 for uint8 codes).
        """
        self.vector_store = vector_store
        self.metric = metric

        dim = vector_store.dimension
        if m_subspaces is None:
            target = min(8, max(1, dim // 16))
            m_subspaces = 1
            for m in range(target, 0, -1):
                if dim % m == 0:
                    m_subspaces = m
                    break

        self.flat_index = FlatIndex(vector_store, metric=metric)
        self.hnsw_index = HNSWIndex(vector_store, metric=metric, **(hnsw_kwargs or {}))
        self.ivf_index  = IVFIndex(vector_store, metric=metric,  **(ivf_kwargs  or {}))
        self.pq_index   = ProductQuantizer(
            M=m_subspaces,
            K=pq_k_centroids,
            dimension=dim,
            vector_store=vector_store,
            metric=metric.value if hasattr(metric, "value") else str(metric),
        )

    # ── Lazy PQ training trigger ──────────────────────────────────────────────

    def _maybe_train_pq(self) -> None:
        """Trigger PQ training once enough vectors have been inserted.
        
        Training is deferred until min_train_size vectors are present to avoid
        a train-on-empty-index crash.  After the first training, subsequent
        inserts are encoded individually via encode_and_add().
        """
        if self.pq_index.is_trained:
            return
        n_active = len(self.vector_store)
        if n_active < self.pq_index.min_train_size:
            return

        active_ids, matrix = self.vector_store.get_active_matrix()
        if matrix.shape[0] == 0:
            return

        # Train codebooks — this is the expensive step (K-Means per sub-space)
        self.pq_index.train(matrix)

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(
        self,
        vector_id: str,
        vector: np.ndarray | list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a vector entry in the vector store and all indices."""
        self.vector_store.add(vector_id, vector, payload)
        self.hnsw_index.insert(vector_id)

        if self.ivf_index.is_trained:
            self.ivf_index.insert(vector_id)

        # PQ: buffer or encode depending on training state
        self.pq_index.encode_and_add(vector_id)
        # Check if we've crossed the training threshold
        self._maybe_train_pq()

    def delete(self, vector_id: str, soft: bool = True) -> bool:
        """Delete a vector entry by ID."""
        return self.vector_store.delete(vector_id, soft=soft)

    def compact(self) -> None:
        """Purge tombstoned vectors and rebuild all indices."""
        self.vector_store.compact()
        self.hnsw_index.rebuild()
        self.ivf_index.rebuild()
        # Rebuild PQ code matrix after compaction
        if self.pq_index.is_trained:
            active_ids, matrix = self.vector_store.get_active_matrix()
            if matrix.shape[0] > 0:
                self.pq_index._ids = list(active_ids)
                self.pq_index._codes = self.pq_index._encode_matrix(matrix)

    def search(
        self,
        query: np.ndarray | list[float],
        k: int = 10,
        filter_query: dict[str, Any] | None = None,
        filter_strategy: FilterStrategy = FilterStrategy.PRE_FILTER,
        index_type: IndexType = IndexType.HNSW,
        oversample_factor: float = 3.0,
        ef_search: int | None = None,
        n_probe: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform similarity search with optional metadata filtering.
        
        Args:
            query:             1D query vector.
            k:                 Top-k nearest neighbours to return.
            filter_query:      JSON query filter payload.
            filter_strategy:   PRE_FILTER or POST_FILTER.
            index_type:        FLAT, IVF, HNSW, or PQ.
            oversample_factor: Oversampling multiplier for post-filtering.
            ef_search:         HNSW ef_search parameter override.
            n_probe:           IVF n_probe parameter override.

        Returns:
            list[dict]: [{id, score, payload, vector}, ...].
        """
        q = np.asarray(query, dtype=np.float32)

        def raw_search(target_k: int, allowed_ids: Collection[str] | None) -> list[tuple[str, float]]:
            if index_type == IndexType.FLAT:
                return self.flat_index.search(q, k=target_k, filter_ids=allowed_ids)
            elif index_type == IndexType.IVF:
                return self.ivf_index.search(q, k=target_k, n_probe=n_probe, filter_ids=allowed_ids)
            elif index_type == IndexType.HNSW:
                return self.hnsw_index.search(q, k=target_k, ef_search=ef_search, filter_ids=allowed_ids)
            elif index_type == IndexType.PQ:
                if not self.pq_index.is_trained:
                    # Fall back to flat search if PQ is not yet trained
                    return self.flat_index.search(q, k=target_k, filter_ids=allowed_ids)
                return self.pq_index.search(q, k=target_k, filter_ids=allowed_ids)
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        raw_results: list[tuple[str, float]] = []

        if not filter_query:
            raw_results = raw_search(k, allowed_ids=None)
        elif filter_strategy == FilterStrategy.PRE_FILTER:
            matching_ids = get_matching_ids(self.vector_store, filter_query)
            if not matching_ids:
                return []
            raw_results = raw_search(k, allowed_ids=matching_ids)
        elif filter_strategy == FilterStrategy.POST_FILTER:
            fetch_k = int(k * oversample_factor)
            candidates = raw_search(fetch_k, allowed_ids=None)
            filtered: list[tuple[str, float]] = []
            for vid, score in candidates:
                payload = self.vector_store.get_payload(vid)
                if payload is not None and evaluate_filter(payload, filter_query):
                    filtered.append((vid, score))
                    if len(filtered) == k:
                        break
            raw_results = filtered
        else:
            raise ValueError(f"Unknown filter strategy: {filter_strategy}")

        # Assemble full result response dicts
        response = []
        for vid, score in raw_results:
            vec_payload = self.vector_store.get(vid)
            if vec_payload is not None:
                vec, payload = vec_payload
                response.append({
                    "id": vid,
                    "score": float(score),
                    "payload": payload,
                    "vector": vec.tolist()
                })

        return response
