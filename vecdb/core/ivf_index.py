from typing import Collection
import numpy as np

from vecdb.core.distance import DistanceMetric, compute_distance, compute_distances
from vecdb.core.vector_store import VectorStore


class IVFIndex:
    """Inverted File (IVF) Index built from first principles with custom k-means clustering.
    
    Partitions the vector space into Voronoi cells based on n_list centroids.
    Queries search only vectors belonging to the top n_probe closest centroids.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        metric: DistanceMetric = DistanceMetric.COSINE,
        n_list: int = 16,
        n_probe: int = 4,
        seed: int | None = 42
    ):
        """Initialize IVFIndex.
        
        Args:
            vector_store: Reference to VectorStore for vector storage.
            metric: Distance metric strategy.
            n_list: Number of clusters (Voronoi cells).
            n_probe: Number of centroids to inspect during query search.
            seed: Random seed for k-means centroid initialization.
        """
        self.vector_store = vector_store
        self.metric = metric
        self.n_list = n_list
        self.n_probe = max(1, min(n_probe, n_list))
        self.rng = np.random.default_rng(seed)

        self.centroids: np.ndarray | None = None  # (n_list, D)
        self.inv_lists: dict[int, list[str]] = {i: [] for i in range(n_list)}
        self.is_trained: bool = False

    def train(self, max_iters: int = 20) -> None:
        """Train k-means centroids on current active vectors in VectorStore.
        
        Args:
            max_iters: Maximum Lloyd's algorithm iterations.
        """
        active_ids, matrix = self.vector_store.get_active_matrix()
        N, D = matrix.shape

        if N == 0:
            return

        actual_n_list = min(self.n_list, N)
        self.inv_lists = {i: [] for i in range(actual_n_list)}

        # K-Means++ initialization
        init_indices = [self.rng.choice(N)]
        centroids = [matrix[init_indices[0]]]

        for _ in range(1, actual_n_list):
            current_c = np.array(centroids)
            # Compute min distance of each point to nearest chosen centroid
            dists = np.min([compute_distances(c, matrix, self.metric) for c in current_c], axis=0)
            dists_sq = dists ** 2
            prob = dists_sq / (np.sum(dists_sq) + 1e-12)
            next_idx = self.rng.choice(N, p=prob)
            centroids.append(matrix[next_idx])

        centroids_arr = np.array(centroids, dtype=np.float32)

        # Lloyd's algorithm iterations
        for _ in range(max_iters):
            # Assign points to closest centroid
            all_dists = np.vstack([compute_distances(c, matrix, self.metric) for c in centroids_arr])  # (actual_n_list, N)
            assignments = np.argmin(all_dists, axis=0)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids_arr)
            for c_idx in range(actual_n_list):
                members = matrix[assignments == c_idx]
                if members.shape[0] > 0:
                    new_centroids[c_idx] = np.mean(members, axis=0)
                else:
                    new_centroids[c_idx] = centroids_arr[c_idx]

            # Check convergence
            if np.allclose(centroids_arr, new_centroids, atol=1e-4):
                break
            centroids_arr = new_centroids

        self.centroids = centroids_arr
        self.is_trained = True

        # Populate inverted lists
        all_dists = np.vstack([compute_distances(c, matrix, self.metric) for c in self.centroids])
        assignments = np.argmin(all_dists, axis=0)

        for i, vid in enumerate(active_ids):
            c_idx = int(assignments[i])
            self.inv_lists[c_idx].append(vid)

    def insert(self, vector_id: str) -> None:
        """Insert a single vector into nearest centroid's inverted list. Train if not trained."""
        if not self.is_trained or self.centroids is None:
            self.train()
            return

        vec = self.vector_store.get_raw_vector(vector_id)
        if vec is None:
            return

        dists = compute_distances(vec, self.centroids, self.metric)
        best_c = int(np.argmin(dists))
        if best_c not in self.inv_lists:
            self.inv_lists[best_c] = []
        if vector_id not in self.inv_lists[best_c]:
            self.inv_lists[best_c].append(vector_id)

    def search(
        self,
        query: np.ndarray | list[float],
        k: int,
        n_probe: int | None = None,
        filter_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        """Search top-k nearest neighbors by checking top n_probe Voronoi cells.
        
        Args:
            query: 1D query vector.
            k: Number of nearest neighbors.
            n_probe: Overriding n_probe parameter.
            filter_ids: Optional set of allowed vector IDs.

        Returns:
            list[tuple[str, float]]: List of (id, distance) sorted ascending.
        """
        if not self.is_trained or self.centroids is None or k <= 0:
            return []

        q = np.asarray(query, dtype=np.float32)
        probe = max(1, min(n_probe or self.n_probe, len(self.centroids)))

        # Find top n_probe centroids
        c_dists = compute_distances(q, self.centroids, self.metric)
        probe_centroids = np.argsort(c_dists)[:probe]

        candidate_ids: set[str] = set()
        filter_set = set(filter_ids) if filter_ids is not None else None

        for c_idx in probe_centroids:
            for vid in self.inv_lists.get(int(c_idx), []):
                if self.vector_store.is_tombstoned(vid):
                    continue
                if filter_set is not None and vid not in filter_set:
                    continue
                candidate_ids.add(vid)

        if not candidate_ids:
            return []

        # Exact distance calculation over candidates
        cand_list = list(candidate_ids)
        matrix = np.vstack([self.vector_store.get_raw_vector(vid) for vid in cand_list])
        distances = compute_distances(q, matrix, self.metric)

        actual_k = min(k, len(distances))
        top_indices = np.argsort(distances)[:actual_k]

        return [(cand_list[idx], float(distances[idx])) for idx in top_indices]

    def rebuild(self) -> None:
        """Re-train centroids and rebuild inverted lists."""
        self.is_trained = False
        self.train()
