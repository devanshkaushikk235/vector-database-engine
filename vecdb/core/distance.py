from enum import Enum
import numpy as np


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity search.
    
    All metrics are implemented such that LOWER scores indicate GREATER similarity
    (closer in vector space).
    """
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


def compute_distance(u: np.ndarray, v: np.ndarray, metric: DistanceMetric) -> float:
    """Compute distance between two 1D vectors u and v.
    
    Args:
        u: 1D numpy array of float32/float64.
        v: 1D numpy array of float32/float64.
        metric: DistanceMetric enum value.

    Returns:
        float: Calculated distance (lower = closer).
    """
    if metric == DistanceMetric.EUCLIDEAN:
        return float(np.linalg.norm(u - v))
    elif metric == DistanceMetric.COSINE:
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < 1e-12 or norm_v < 1e-12:
            return 1.0
        similarity = float(np.dot(u, v) / (norm_u * norm_v))
        # Clip to prevent floating point out-of-bounds [-1.0, 1.0]
        similarity = max(-1.0, min(1.0, similarity))
        return 1.0 - similarity
    elif metric == DistanceMetric.DOT_PRODUCT:
        # Negative dot product so higher dot product -> smaller distance
        return float(-np.dot(u, v))
    elif metric == DistanceMetric.MANHATTAN:
        return float(np.sum(np.abs(u - v)))
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def compute_distances(query: np.ndarray, dataset: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Compute distance from a 1D query vector to a 2D matrix of vectors (N x D).
    
    Args:
        query: 1D numpy array of shape (D,).
        dataset: 2D numpy array of shape (N, D).
        metric: DistanceMetric enum value.

    Returns:
        np.ndarray: 1D array of shape (N,) containing distances.
    """
    if dataset.size == 0:
        return np.empty(0, dtype=np.float32)

    if metric == DistanceMetric.EUCLIDEAN:
        diff = dataset - query
        return np.linalg.norm(diff, axis=1)
    elif metric == DistanceMetric.COSINE:
        norm_query = np.linalg.norm(query)
        norm_dataset = np.linalg.norm(dataset, axis=1)
        
        # Avoid division by zero
        denom = norm_query * norm_dataset
        denom = np.where(denom < 1e-12, 1e-12, denom)
        
        dots = np.dot(dataset, query)
        similarity = dots / denom
        similarity = np.clip(similarity, -1.0, 1.0)
        return 1.0 - similarity
    elif metric == DistanceMetric.DOT_PRODUCT:
        return -np.dot(dataset, query)
    elif metric == DistanceMetric.MANHATTAN:
        return np.sum(np.abs(dataset - query), axis=1)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


def compute_batch_distances(queries: np.ndarray, dataset: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Compute distances from a 2D batch of queries (Q x D) to a 2D dataset (N x D).
    
    Args:
        queries: 2D numpy array of shape (Q, D).
        dataset: 2D numpy array of shape (N, D).
        metric: DistanceMetric enum value.

    Returns:
        np.ndarray: 2D array of shape (Q, N) where result[i, j] is dist(query_i, vector_j).
    """
    if queries.size == 0 or dataset.size == 0:
        return np.empty((queries.shape[0], dataset.shape[0]), dtype=np.float32)

    if metric == DistanceMetric.EUCLIDEAN:
        # ||q - d||^2 = ||q||^2 + ||d||^2 - 2 (q . d)
        q_sq = np.sum(queries ** 2, axis=1, keepdims=True)  # (Q, 1)
        d_sq = np.sum(dataset ** 2, axis=1, keepdims=True).T  # (1, N)
        qd_dot = np.dot(queries, dataset.T)  # (Q, N)
        sq_dist = np.maximum(0.0, q_sq + d_sq - 2 * qd_dot)
        return np.sqrt(sq_dist)
    elif metric == DistanceMetric.COSINE:
        norm_q = np.linalg.norm(queries, axis=1, keepdims=True)  # (Q, 1)
        norm_d = np.linalg.norm(dataset, axis=1, keepdims=True).T  # (1, N)
        denom = norm_q * norm_d
        denom = np.where(denom < 1e-12, 1e-12, denom)
        dots = np.dot(queries, dataset.T)
        similarity = np.clip(dots / denom, -1.0, 1.0)
        return 1.0 - similarity
    elif metric == DistanceMetric.DOT_PRODUCT:
        return -np.dot(queries, dataset.T)
    elif metric == DistanceMetric.MANHATTAN:
        # Broadcasting Manhattan dist: (Q, 1, D) - (1, N, D)
        diff = np.abs(queries[:, np.newaxis, :] - dataset[np.newaxis, :, :])
        return np.sum(diff, axis=2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
