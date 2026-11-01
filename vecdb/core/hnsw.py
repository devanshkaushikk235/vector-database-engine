import heapq
import math
import random
import threading
from typing import Collection
import numpy as np

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore


class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) graph index built from first principles.
    
    Thread-safe via a readers-writer lock:
      • Multiple concurrent search() calls proceed in parallel (read lock).
      • insert() and rebuild() are serialised with a write lock.
      • Critically, entry_point and max_layer are read under the lock to prevent
        a TOCTOU race where a concurrent insert() promotes the entry point between
        the moment search() reads it and the moment it starts graph traversal.
    
    References:
        Malkov & Yashunin (2018): "Efficient and robust approximate nearest neighbor
        search using Hierarchical Navigable Small World graphs"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        metric: DistanceMetric = DistanceMetric.COSINE,
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        seed: int | None = 42
    ):
        """Initialize HNSW Index.
        
        Args:
            vector_store: Reference to VectorStore for vector data retrieval.
            metric: Distance metric strategy.
            M: Max number of connections per node for layers > 0.
            ef_construction: Size of dynamic candidate list during graph construction.
            ef_search: Size of dynamic candidate list during query search.
            seed: Random seed for level generation reproducibility.
        """
        self.vector_store = vector_store
        self.metric = metric
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(M) if M > 1 else 1.0

        if seed is not None:
            random.seed(seed)

        # Graph structure: graph[layer][node_id] = set of neighbor_ids
        self.graph: list[dict[str, set[str]]] = []
        # Store individual node max level: node_levels[node_id] = level
        self.node_levels: dict[str, int] = {}

        self.entry_point: str | None = None
        self.max_layer: int = -1
        self._norms: dict[str, float] = {}

        # ── Readers-Writer Lock ────────────────────────────────────────────
        # _write_lock: held exclusively by insert() and rebuild().
        # _read_lock + _readers: allows multiple concurrent search() calls.
        # We use a simple counter-based RW lock so that multiple readers can
        # proceed in parallel while writers get exclusive access.
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()   # guards _readers count
        self._readers: int = 0
        self._write_event = threading.Lock()  # acquired by writer; readers block on it

    # ── RW Lock helpers ───────────────────────────────────────────────────

    def _acquire_read(self) -> None:
        """Increment reader count; block if a writer holds the write event."""
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                # First reader acquires the write-event lock to block writers.
                self._write_event.acquire()

    def _release_read(self) -> None:
        """Decrement reader count; release write event when last reader exits."""
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_event.release()

    # ── Private graph accessors ───────────────────────────────────────────

    @property
    def _vectors(self) -> dict[str, np.ndarray]:
        """Dynamically reference VectorStore internal vector dictionary."""
        return self.vector_store._vectors

    def _get_norm(self, vid: str, vec: np.ndarray) -> float:
        """Get or cache precomputed vector squared norm."""
        norm_val = self._norms.get(vid)
        if norm_val is None:
            norm_val = float(np.dot(vec, vec))
            self._norms[vid] = norm_val
        return norm_val

    def _get_distance_with_sq(self, q: np.ndarray, q_sq: float, vid: str, vec: np.ndarray) -> float:
        """Accelerated distance calculation using precomputed query squared norm and vector norm."""
        if self.metric == DistanceMetric.EUCLIDEAN:
            v_sq = self._get_norm(vid, vec)
            dot = float(np.dot(q, vec))
            return math.sqrt(max(0.0, q_sq + v_sq - 2.0 * dot))
        elif self.metric == DistanceMetric.COSINE:
            v_sq = self._get_norm(vid, vec)
            denom = math.sqrt(q_sq * v_sq)
            if denom < 1e-12:
                return 1.0
            dot = float(np.dot(q, vec))
            sim = max(-1.0, min(1.0, dot / denom))
            return 1.0 - sim
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            return -float(np.dot(q, vec))
        elif self.metric == DistanceMetric.MANHATTAN:
            return float(np.sum(np.abs(q - vec)))
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def _get_distance_between_nodes(self, id1: str, vec1: np.ndarray, id2: str, vec2: np.ndarray) -> float:
        """Fast distance between two nodes using cached norms."""
        if self.metric == DistanceMetric.EUCLIDEAN:
            sq1 = self._get_norm(id1, vec1)
            sq2 = self._get_norm(id2, vec2)
            dot = float(np.dot(vec1, vec2))
            return math.sqrt(max(0.0, sq1 + sq2 - 2.0 * dot))
        elif self.metric == DistanceMetric.COSINE:
            sq1 = self._get_norm(id1, vec1)
            sq2 = self._get_norm(id2, vec2)
            denom = math.sqrt(sq1 * sq2)
            if denom < 1e-12:
                return 1.0
            dot = float(np.dot(vec1, vec2))
            sim = max(-1.0, min(1.0, dot / denom))
            return 1.0 - sim
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            return -float(np.dot(vec1, vec2))
        elif self.metric == DistanceMetric.MANHATTAN:
            return float(np.sum(np.abs(vec1 - vec2)))
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def _random_level(self) -> int:
        """Probabilistic level assignment formula from Malkov & Yashunin (2018):
        l = floor(-ln(unif(0, 1)) * m_L)
        """
        unif = max(1e-12, random.random())
        return int(math.floor(-math.log(unif) * self.m_L))

    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: set[str],
        ef: int,
        layer: int
    ) -> list[tuple[float, str]]:
        """Greedy search in a single layer.
        
        Must be called while the caller holds at least a read lock so that
        the graph snapshot is consistent for the duration of traversal.
        
        Args:
            q: Query vector.
            entry_points: Set of node IDs to start searching from.
            ef: Beam search capacity.
            layer: Layer level to search within.

        Returns:
            list[tuple[float, str]]: Nearest neighbors (dist, node_id) found in layer.
        """
        if layer >= len(self.graph):
            return []

        layer_graph = self.graph[layer]
        v = set(entry_points)

        q_sq = float(np.dot(q, q))

        C: list[tuple[float, str]] = []   # min-heap: candidates
        W: list[tuple[float, str]] = []   # max-heap: found nearest neighbours

        is_euclidean = (self.metric == DistanceMetric.EUCLIDEAN)

        for ep in entry_points:
            vec = self._vectors.get(ep)
            if vec is None:
                continue
            if is_euclidean:
                v_sq = self._get_norm(ep, vec)
                dist = math.sqrt(max(0.0, q_sq + v_sq - 2.0 * float(np.dot(q, vec))))
            else:
                dist = self._get_distance_with_sq(q, q_sq, ep, vec)
            heapq.heappush(C, (dist, ep))
            heapq.heappush(W, (-dist, ep))

        while C:
            dist_c, c = heapq.heappop(C)
            if not W:
                break
            worst_dist = -W[0][0]

            if dist_c > worst_dist and len(W) >= ef:
                break

            neighbors = layer_graph.get(c, set())
            for neighbor in neighbors:
                if neighbor in v:
                    continue
                v.add(neighbor)

                n_vec = self._vectors.get(neighbor)
                if n_vec is None:
                    continue

                if is_euclidean:
                    v_sq = self._get_norm(neighbor, n_vec)
                    dist_n = math.sqrt(max(0.0, q_sq + v_sq - 2.0 * float(np.dot(q, n_vec))))
                else:
                    dist_n = self._get_distance_with_sq(q, q_sq, neighbor, n_vec)

                worst_dist = -W[0][0]

                if dist_n < worst_dist or len(W) < ef:
                    heapq.heappush(C, (dist_n, neighbor))
                    heapq.heappush(W, (-dist_n, neighbor))
                    if len(W) > ef:
                        heapq.heappop(W)

        result = [(-neg_d, node_id) for neg_d, node_id in W]
        result.sort(key=lambda x: x[0])
        return result

    def _select_neighbors_heuristic(
        self,
        q: np.ndarray,
        candidates: list[tuple[float, str]],
        M_max: int
    ) -> list[str]:
        """Select neighbors heuristic (SELECT-NEIGHBORS-HEURISTIC).
        
        Evaluates candidates in order of distance, adding a candidate to the output
        set only if it is closer to the query than to any neighbour already selected.
        """
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        result: list[str] = []
        result_nodes: list[tuple[str, np.ndarray]] = []

        for dist_e, e in sorted_candidates:
            if len(result) >= M_max:
                break

            e_vec = self._vectors.get(e)
            if e_vec is None:
                continue

            keep = True
            for r_id, r_vec in result_nodes:
                dist_e_r = self._get_distance_between_nodes(e, e_vec, r_id, r_vec)
                if dist_e_r < dist_e:
                    keep = False
                    break

            if keep:
                result.append(e)
                result_nodes.append((e, e_vec))

        # Fallback: if heuristic returned fewer than M_max, fill with remaining candidates
        if len(result) < M_max:
            for _, e in sorted_candidates:
                if e not in result:
                    result.append(e)
                    if len(result) >= M_max:
                        break

        return result

    # ── Public API ────────────────────────────────────────────────────────

    def insert(self, vector_id: str) -> None:
        """Insert a vector ID into the HNSW graph (exclusive write lock)."""
        vec = self._vectors.get(vector_id)
        if vec is None:
            raise ValueError(f"Vector ID {vector_id} not found in VectorStore.")

        # Writers acquire write_lock (mutual exclusion with other writers)
        # and then wait for all current readers to finish via write_event.
        with self._write_lock:
            self._write_event.acquire()
            try:
                self._insert_unlocked(vector_id, vec)
            finally:
                self._write_event.release()

    def _insert_unlocked(self, vector_id: str, vec: np.ndarray) -> None:
        """Internal insert — must be called while holding write_lock + write_event."""
        self._get_norm(vector_id, vec)
        l = self._random_level()

        while len(self.graph) <= l:
            self.graph.append({})

        self.node_levels[vector_id] = l

        if self.entry_point is None:
            for lc in range(l + 1):
                self.graph[lc][vector_id] = set()
            self.entry_point = vector_id
            self.max_layer = l
            return

        ep_set = {self.entry_point}
        L = self.max_layer

        for lc in range(L, l, -1):
            W = self._search_layer(vec, ep_set, ef=1, layer=lc)
            if W:
                ep_set = {W[0][1]}

        for lc in range(min(L, l), -1, -1):
            W = self._search_layer(vec, ep_set, ef=self.ef_construction, layer=lc)
            ep_set = {node_id for _, node_id in W}

            M_max = self.M_max0 if lc == 0 else self.M
            neighbors = self._select_neighbors_heuristic(vec, W, M_max)

            if vector_id not in self.graph[lc]:
                self.graph[lc][vector_id] = set()

            for nbr in neighbors:
                self.graph[lc][vector_id].add(nbr)
                if nbr not in self.graph[lc]:
                    self.graph[lc][nbr] = set()
                self.graph[lc][nbr].add(vector_id)

                nbr_links = self.graph[lc][nbr]
                if len(nbr_links) > M_max:
                    nbr_vec = self._vectors.get(nbr)
                    if nbr_vec is not None:
                        nbr_sq = self._get_norm(nbr, nbr_vec)
                        sorted_links = sorted(
                            nbr_links,
                            key=lambda link: self._get_distance_with_sq(nbr_vec, nbr_sq, link, self._vectors[link])
                            if link in self._vectors else float('inf')
                        )
                        self.graph[lc][nbr] = set(sorted_links[:M_max])

        if l > L:
            self.entry_point = vector_id
            self.max_layer = l

    def search(
        self,
        query: np.ndarray | list[float],
        k: int,
        ef_search: int | None = None,
        filter_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        """Search for top-k nearest neighbours using HNSW graph traversal.
        
        TOCTOU safety: entry_point and max_layer are read inside the read lock,
        before graph traversal begins.  This prevents a race where a concurrent
        insert() promotes the entry point between the snapshot read and traversal.
        
        Args:
            query: 1D query vector.
            k: Number of nearest neighbours to return.
            ef_search: Overriding beam width for search.
            filter_ids: Optional set of vector IDs to restrict search results.

        Returns:
            list[tuple[str, float]]: List of (vector_id, distance_score) sorted ascending.
        """
        if k <= 0:
            return []

        q = np.asarray(query, dtype=np.float32)
        ef = max(ef_search or self.ef_search, k)

        self._acquire_read()
        try:
            # ── Snapshot mutable state under read lock ─────────────────────
            # Reading entry_point and max_layer here (inside the lock) prevents
            # the TOCTOU race: a concurrent insert() cannot promote entry_point
            # between this read and the subsequent _search_layer() calls.
            entry_point = self.entry_point
            max_layer = self.max_layer

            if entry_point is None:
                return []

            ep_set = {entry_point}
            L = max_layer

            # Traverse upper layers (greedy search with ef=1)
            for lc in range(L, 0, -1):
                W = self._search_layer(q, ep_set, ef=1, layer=lc)
                if W:
                    ep_set = {W[0][1]}

            # Layer 0 search with full beam width ef
            W = self._search_layer(q, ep_set, ef=ef, layer=0)
        finally:
            self._release_read()

        # Post-lock: filter tombstoned / disallowed IDs (VectorStore reads are fine here)
        filter_set = set(filter_ids) if filter_ids is not None else None
        results: list[tuple[str, float]] = []

        for dist, vid in W:
            if self.vector_store.is_tombstoned(vid):
                continue
            if filter_set is not None and vid not in filter_set:
                continue
            results.append((vid, dist))
            if len(results) == k:
                break

        if filter_set is not None and len(results) < k:
            results = []
            for dist, vid in W:
                if not self.vector_store.is_tombstoned(vid) and vid in filter_set:
                    results.append((vid, dist))

        return results[:k]

    def rebuild(self) -> None:
        """Rebuild index graph from active non-tombstoned entries (exclusive write lock)."""
        with self._write_lock:
            self._write_event.acquire()
            try:
                self.graph = []
                self.node_levels = {}
                self.entry_point = None
                self.max_layer = -1
                self._norms.clear()

                active_ids = self.vector_store.get_all_ids(include_tombstones=False)
                for vid in active_ids:
                    vec = self._vectors.get(vid)
                    if vec is not None:
                        self._insert_unlocked(vid, vec)
            finally:
                self._write_event.release()
