from typing import Collection
import numpy as np

from vecdb.core.distance import DistanceMetric, compute_distances, compute_batch_distances
from vecdb.core.vector_store import VectorStore


class FlatIndex:
    """Exact brute-force kNN index.
    
    Serves as the ground-truth baseline for exact search and recall evaluation.
    """

    def __init__(self, vector_store: VectorStore, metric: DistanceMetric = DistanceMetric.COSINE):
        """Initialize FlatIndex.
        
        Args:
            vector_store: Reference to VectorStore instance containing vectors.
            metric: DistanceMetric enum value.
        """
        self.vector_store = vector_store
        self.metric = metric

    def search(
        self,
        query: np.ndarray | list[float],
        k: int,
        filter_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        """Perform exact kNN search for a single query vector.
        
        Args:
            query: 1D query vector.
            k: Number of nearest neighbors to return.
            filter_ids: Optional set of allowed vector IDs (pre-filtered).

        Returns:
            list[tuple[str, float]]: List of (id, distance_score) sorted ascending by distance.
        """
        if k <= 0:
            return []

        q = np.asarray(query, dtype=np.float32)
        active_ids, matrix = self.vector_store.get_active_matrix()

        if matrix.shape[0] == 0:
            return []

        # If candidate filter provided, restrict indices
        if filter_ids is not None:
            filter_set = set(filter_ids)
            indices = [i for i, vid in enumerate(active_ids) if vid in filter_set]
            if not indices:
                return []
            active_ids = [active_ids[i] for i in indices]
            matrix = matrix[indices]

        distances = compute_distances(q, matrix, self.metric)
        
        # Select top k
        actual_k = min(k, len(distances))
        if actual_k == len(distances):
            top_indices = np.argsort(distances)
        else:
            # Partition for top k, then sort top k
            top_indices = np.argpartition(distances, actual_k - 1)[:actual_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]

        return [(active_ids[idx], float(distances[idx])) for idx in top_indices]

    def range_search(
        self,
        query: np.ndarray | list[float],
        radius: float,
        filter_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        """Perform range search returning all vectors within distance <= radius.
        
        Args:
            query: 1D query vector.
            radius: Maximum distance threshold (inclusive).
            filter_ids: Optional set of allowed vector IDs.

        Returns:
            list[tuple[str, float]]: List of (id, distance_score) within radius, sorted ascending.
        """
        q = np.asarray(query, dtype=np.float32)
        active_ids, matrix = self.vector_store.get_active_matrix()

        if matrix.shape[0] == 0:
            return []

        if filter_ids is not None:
            filter_set = set(filter_ids)
            indices = [i for i, vid in enumerate(active_ids) if vid in filter_set]
            if not indices:
                return []
            active_ids = [active_ids[i] for i in indices]
            matrix = matrix[indices]

        distances = compute_distances(q, matrix, self.metric)
        matching_mask = distances <= radius
        matching_indices = np.where(matching_mask)[0]

        if matching_indices.size == 0:
            return []

        sorted_order = matching_indices[np.argsort(distances[matching_indices])]
        return [(active_ids[idx], float(distances[idx])) for idx in sorted_order]

    def batch_search(
        self,
        queries: np.ndarray | list[list[float]],
        k: int,
        filter_ids: Collection[str] | None = None
    ) -> list[list[tuple[str, float]]]:
        """Perform exact kNN search for a batch of query vectors.
        
        Args:
            queries: 2D array of shape (Q, D).
            k: Number of nearest neighbors per query.
            filter_ids: Optional set of allowed vector IDs.

        Returns:
            list[list[tuple[str, float]]]: List of kNN search results for each query vector.
        """
        qs = np.asarray(queries, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)

        active_ids, matrix = self.vector_store.get_active_matrix()

        if matrix.shape[0] == 0 or k <= 0:
            return [[] for _ in range(qs.shape[0])]

        if filter_ids is not None:
            filter_set = set(filter_ids)
            indices = [i for i, vid in enumerate(active_ids) if vid in filter_set]
            if not indices:
                return [[] for _ in range(qs.shape[0])]
            active_ids = [active_ids[i] for i in indices]
            matrix = matrix[indices]

        dist_matrix = compute_batch_distances(qs, matrix, self.metric)  # (Q, N)
        results = []

        num_vectors = len(active_ids)
        actual_k = min(k, num_vectors)

        for i in range(qs.shape[0]):
            distances = dist_matrix[i]
            if actual_k == num_vectors:
                top_indices = np.argsort(distances)
            else:
                top_indices = np.argpartition(distances, actual_k - 1)[:actual_k]
                top_indices = top_indices[np.argsort(distances[top_indices])]
            
            results.append([(active_ids[idx], float(distances[idx])) for idx in top_indices])

        return results
