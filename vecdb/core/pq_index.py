"""Product Quantization (PQ) index for compressed approximate nearest-neighbor search.

Theory
------
PQ splits a D-dimensional vector into M sub-spaces of D/M dimensions each.
Each sub-space is independently quantized to K centroids (the "codebook").
A vector is represented as M uint8 codes — one centroid index per sub-space.
This compresses D*4 bytes → M bytes (up to 512× compression for D=128, M=8, K=256).

Search uses Asymmetric Distance Computation (ADC):
  1. Pre-compute a distance table: for each of the M sub-spaces, compute the distance
     from the query sub-vector to every one of the K centroids.  O(M * K * D/M) = O(K*D).
  2. For each encoded database vector (M uint8 codes), look up M distances and sum them.
     This is O(N*M) with no floating-point multiplications — extremely fast.

Usage
-----
    pq = ProductQuantizer(M=8, K=256, dimension=128)
    pq.train(training_vectors)        # K-Means per sub-space
    codes = pq.encode(vectors)        # (N, M) uint8
    results = pq.search(query, k=10)  # ADC nearest-neighbour search
"""

from __future__ import annotations

import threading
from typing import Any, Collection
import numpy as np


class ProductQuantizer:
    """Product Quantization index with Asymmetric Distance Computation (ADC).
    
    Thread-safety: a threading.RLock serialises train() and encode_and_add().
    search() acquires a read-snapshot of the codebooks and code matrix, so
    concurrent searches do not block each other after training.
    """

    # Minimum number of vectors required before training is triggered
    MIN_TRAIN_SIZE_MULTIPLIER = 2  # at least 2 * K vectors per sub-space centroid

    def __init__(
        self,
        M: int,
        K: int,
        dimension: int,
        vector_store: Any,  # VectorStore reference for ID lookup
        metric: str = "cosine",
        seed: int = 42,
    ):
        """
        Args:
            M:            Number of sub-spaces.  Must satisfy dimension % M == 0.
            K:            Number of centroids per sub-space (≤ 256 to fit in uint8).
            dimension:    Full vector dimension.
            vector_store: VectorStore instance for ID-to-vector lookups.
            metric:       Distance metric ('cosine', 'euclidean', 'dot_product').
            seed:         NumPy random seed for K-Means initialisation.
        """
        if dimension % M != 0:
            raise ValueError(f"dimension {dimension} must be divisible by M={M}")
        if K > 256:
            raise ValueError("K must be ≤ 256 to store codes as uint8")

        self.M = M
        self.K = K
        self.dimension = dimension
        self.sub_dim = dimension // M
        self.metric = metric
        self._rng = np.random.default_rng(seed)
        self.vector_store = vector_store

        # Codebook: (M, K, sub_dim) — one set of K centroids per sub-space
        self.codebooks: np.ndarray | None = None
        self.is_trained: bool = False

        # Encoded database: list of (vector_id, codes_array)
        # codes_array shape: (M,) uint8
        self._ids: list[str] = []
        self._codes: np.ndarray | None = None   # (N, M) uint8 — rebuilt on encode_and_add

        # Minimum number of vectors before we trigger training
        self.min_train_size: int = K * self.MIN_TRAIN_SIZE_MULTIPLIER

        self._lock = threading.RLock()

    # ── Internals ────────────────────────────────────────────────────────

    def _sub_vectors(self, matrix: np.ndarray, m: int) -> np.ndarray:
        """Extract sub-vectors for sub-space m. Returns (N, sub_dim) slice."""
        start = m * self.sub_dim
        return matrix[:, start: start + self.sub_dim]

    def _kmeans_train(self, data: np.ndarray, K: int, max_iter: int = 25) -> np.ndarray:
        """K-Means++ with Lloyd's iterations for one sub-space.
        
        Args:
            data:    (N, D_sub) float32 matrix.
            K:       Number of centroids.
            max_iter: Maximum Lloyd iterations.
            
        Returns:
            centroids: (K, D_sub) float32 array.
        """
        N = data.shape[0]
        if N < K:
            repeats = (K + N - 1) // N
            tiled = np.tile(data, (repeats, 1))[:K]
            noise = self._rng.normal(0, 1e-4, size=tiled.shape).astype(np.float32)
            data = tiled + noise
            N = data.shape[0]

        # Vectorized K-Means++ initialisation
        centroids = np.empty((K, data.shape[1]), dtype=np.float32)
        centroids[0] = data[self._rng.integers(0, N)]
        
        diff0 = data - centroids[0]
        min_dists_sq = np.sum(diff0 * diff0, axis=1)

        for k in range(1, K):
            total_dist = min_dists_sq.sum()
            if total_dist <= 1e-12:
                probs = None
            else:
                probs = min_dists_sq / total_dist

            idx = self._rng.choice(N, p=probs)
            centroids[k] = data[idx]

            diff_k = data - centroids[k]
            dists_sq_k = np.sum(diff_k * diff_k, axis=1)
            min_dists_sq = np.minimum(min_dists_sq, dists_sq_k)

        # Lloyd's iterations
        for _ in range(max_iter):
            # Assignment step: closest centroid for each vector
            # Using broadcasting: (N, 1, D) - (1, K, D) → (N, K, D) → (N, K)
            diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dists_sq = np.einsum("nkd,nkd->nk", diff, diff)
            labels = np.argmin(dists_sq, axis=1)  # (N,)

            # Update step
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                members = data[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]  # keep empty centroid in place

            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids  # (K, D_sub)

    def _encode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Encode (N, D) float32 matrix → (N, M) uint8 codes.
        
        Must be called while holding self._lock (codebooks must be trained).
        """
        assert self.codebooks is not None
        N = matrix.shape[0]
        codes = np.zeros((N, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub = self._sub_vectors(matrix, m)           # (N, sub_dim)
            cb  = self.codebooks[m]                      # (K, sub_dim)
            # Squared distances via broadcasting: (N, K)
            diff = sub[:, np.newaxis, :] - cb[np.newaxis, :, :]  # (N, K, sub_dim)
            dists_sq = np.einsum("nkd,nkd->nk", diff, diff)
            codes[:, m] = np.argmin(dists_sq, axis=1).astype(np.uint8)

        return codes

    def _adc_table(self, query: np.ndarray) -> np.ndarray:
        """Compute Asymmetric Distance Computation (ADC) look-up table.
        
        For each sub-space m and each centroid k, compute the distance from
        the query's sub-vector to centroid k.
        
        Returns:
            table: (M, K) float32 — pre-computed distances.
        """
        assert self.codebooks is not None
        table = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            q_sub = query[m * self.sub_dim: (m + 1) * self.sub_dim]  # (sub_dim,)
            cb = self.codebooks[m]                                     # (K, sub_dim)
            diff = cb - q_sub[np.newaxis, :]                          # (K, sub_dim)
            table[m] = np.einsum("kd,kd->k", diff, diff)              # (K,) sq dists
        return table

    # ── Public API ────────────────────────────────────────────────────────

    def train(self, training_vectors: np.ndarray) -> None:
        """Train codebooks from a set of vectors using K-Means per sub-space.
        
        Args:
            training_vectors: (N, D) float32 matrix with N >= min_train_size.
        """
        with self._lock:
            self.codebooks = np.zeros((self.M, self.K, self.sub_dim), dtype=np.float32)
            for m in range(self.M):
                sub = self._sub_vectors(training_vectors, m)
                self.codebooks[m] = self._kmeans_train(sub, self.K)

            # Re-encode all previously pending vectors
            if self._ids:
                all_vecs = np.array([
                    self.vector_store._vectors[vid]
                    for vid in self._ids
                    if vid in self.vector_store._vectors
                ], dtype=np.float32)
                if len(all_vecs) > 0:
                    self._codes = self._encode_matrix(all_vecs)

            self.is_trained = True

    def encode_and_add(self, vector_id: str) -> None:
        """Encode a single vector from VectorStore and append to the code matrix.
        
        If training has not yet occurred, the ID is buffered.  Once train() is
        called, all buffered IDs are batch-encoded.
        
        Args:
            vector_id: Must already exist in self.vector_store.
        """
        with self._lock:
            if vector_id in self._ids:
                return  # already encoded (upsert guard)
            self._ids.append(vector_id)

            if not self.is_trained:
                return  # buffered; will be encoded after training

            vec = self.vector_store._vectors.get(vector_id)
            if vec is None:
                return
            code = self._encode_matrix(vec[np.newaxis, :])[0]  # (M,)
            if self._codes is None:
                self._codes = code[np.newaxis, :]
            else:
                self._codes = np.vstack([self._codes, code[np.newaxis, :]])

    def search(
        self,
        query: np.ndarray,
        k: int,
        filter_ids: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Approximate kNN search via Asymmetric Distance Computation (ADC).
        
        Complexity: O(M * K) ADC-table build + O(N * M) table lookups.
        
        Args:
            query:      1D float32 query vector.
            k:          Number of nearest neighbours.
            filter_ids: Optional set of allowed IDs (pre-filter support).
            
        Returns:
            list[tuple[str, float]]: (vector_id, approx_distance) sorted ascending.
        """
        with self._lock:
            if not self.is_trained or self._codes is None or len(self._ids) == 0:
                return []

            # Snapshot to avoid mutation during lookup
            ids_snap   = list(self._ids)
            codes_snap = self._codes.copy()

        # Build ADC table outside the lock (read-only on codebooks)
        table = self._adc_table(query)  # (M, K)

        # Compute approximate distances for all encoded vectors
        # codes_snap: (N, M) uint8
        # table[m, codes_snap[:, m]] → distance for sub-space m for all N vectors
        N = len(ids_snap)
        approx_dists = np.zeros(N, dtype=np.float32)
        for m in range(self.M):
            approx_dists += table[m][codes_snap[:, m]]  # vectorised table lookup

        # Respect tombstones and filter set
        filter_set = set(filter_ids) if filter_ids is not None else None
        pairs: list[tuple[float, str]] = []
        for i, vid in enumerate(ids_snap):
            if self.vector_store.is_tombstoned(vid):
                continue
            if filter_set is not None and vid not in filter_set:
                continue
            pairs.append((float(approx_dists[i]), vid))

        pairs.sort(key=lambda x: x[0])
        return [(vid, dist) for dist, vid in pairs[:k]]
