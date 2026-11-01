from typing import Any
import numpy as np


class VectorStore:
    """In-memory storage for (id, vector, payload) triples with soft-delete support."""

    def __init__(self, dimension: int):
        """Initialize VectorStore for a given vector dimension.
        
        Args:
            dimension: Dimensionality of vectors stored.
        """
        self.dimension = dimension
        self._vectors: dict[str, np.ndarray] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self._tombstones: set[str] = set()

    def add(
        self,
        vector_id: str,
        vector: np.ndarray | list[float],
        payload: dict[str, Any] | None = None
    ) -> None:
        """Add or update a vector and its payload.
        
        Args:
            vector_id: Unique string identifier.
            vector: 1D array or list of floats matching store dimension.
            payload: Optional metadata JSON dict.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected ({self.dimension},), got {arr.shape}"
            )

        self._vectors[vector_id] = arr
        self._payloads[vector_id] = payload if payload is not None else {}
        self._tombstones.discard(vector_id)

    def get(self, vector_id: str) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Retrieve a vector and payload by ID. Returns None if missing or soft-deleted."""
        if vector_id not in self._vectors or vector_id in self._tombstones:
            return None
        return self._vectors[vector_id].copy(), self._payloads[vector_id].copy()

    def get_vector(self, vector_id: str) -> np.ndarray | None:
        """Retrieve vector by ID if active."""
        if vector_id not in self._vectors or vector_id in self._tombstones:
            return None
        return self._vectors[vector_id]

    def get_raw_vector(self, vector_id: str) -> np.ndarray | None:
        """Retrieve vector by ID even if soft-deleted/tombstoned (for index routing)."""
        return self._vectors.get(vector_id)

    def get_payload(self, vector_id: str) -> dict[str, Any] | None:
        """Retrieve payload by ID if active."""
        if vector_id not in self._vectors or vector_id in self._tombstones:
            return None
        return self._payloads[vector_id]

    def delete(self, vector_id: str, soft: bool = True) -> bool:
        """Delete a vector.
        
        Args:
            vector_id: ID to delete.
            soft: If True, mark as tombstoned; if False, hard delete.
            
        Returns:
            bool: True if vector existed and was deleted, False otherwise.
        """
        if vector_id not in self._vectors or vector_id in self._tombstones:
            return False

        if soft:
            self._tombstones.add(vector_id)
        else:
            del self._vectors[vector_id]
            del self._payloads[vector_id]
            self._tombstones.discard(vector_id)
        return True

    def is_tombstoned(self, vector_id: str) -> bool:
        """Check if vector ID is marked as soft-deleted."""
        return vector_id in self._tombstones

    def __contains__(self, vector_id: str) -> bool:
        """Check if active (non-tombstoned) vector exists."""
        return vector_id in self._vectors and vector_id not in self._tombstones

    def __len__(self) -> int:
        """Return count of active (non-tombstoned) vectors."""
        return len(self._vectors) - len(self._tombstones)

    @property
    def total_count(self) -> int:
        """Return total count including tombstoned entries."""
        return len(self._vectors)

    @property
    def tombstone_count(self) -> int:
        """Return count of tombstoned entries."""
        return len(self._tombstones)

    def compact(self) -> list[str]:
        """Hard purge all soft-deleted tombstoned entries.
        
        Returns:
            list[str]: List of purged vector IDs.
        """
        purged = list(self._tombstones)
        for vid in purged:
            self._vectors.pop(vid, None)
            self._payloads.pop(vid, None)
        self._tombstones.clear()
        return purged

    def get_all_ids(self, include_tombstones: bool = False) -> list[str]:
        """Get list of stored vector IDs."""
        if include_tombstones:
            return list(self._vectors.keys())
        return [vid for vid in self._vectors if vid not in self._tombstones]

    def get_active_matrix(self) -> tuple[list[str], np.ndarray]:
        """Construct a contiguous 2D numpy array of all active vectors.
        
        Returns:
            tuple[list[str], np.ndarray]: Active vector IDs and corresponding (N x D) matrix.
        """
        active_ids = self.get_all_ids(include_tombstones=False)
        if not active_ids:
            return [], np.empty((0, self.dimension), dtype=np.float32)

        matrix = np.vstack([self._vectors[vid] for vid in active_ids])
        return active_ids, matrix
