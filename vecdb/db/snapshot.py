import os
import pickle
from typing import Any

from vecdb.core.vector_store import VectorStore
from vecdb.core.hnsw import HNSWIndex


class SnapshotManager:
    """Snapshot manager for binary serialization and recovery of collection state."""

    @staticmethod
    def create_snapshot(
        filepath: str,
        vector_store: VectorStore,
        hnsw_index: HNSWIndex | None = None
    ) -> None:
        """Serialize VectorStore and HNSW graph state to a binary snapshot file.
        
        Args:
            filepath: Target output file path.
            vector_store: VectorStore instance.
            hnsw_index: Optional HNSWIndex instance.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        state = {
            "dimension": vector_store.dimension,
            "vectors": vector_store._vectors,
            "payloads": vector_store._payloads,
            "tombstones": vector_store._tombstones,
            "hnsw_graph": hnsw_index.graph if hnsw_index else None,
            "hnsw_node_levels": hnsw_index.node_levels if hnsw_index else None,
            "hnsw_entry_point": hnsw_index.entry_point if hnsw_index else None,
            "hnsw_max_layer": hnsw_index.max_layer if hnsw_index else None,
        }

        tmp_path = filepath + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Atomic replace
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_path, filepath)

    @staticmethod
    def load_snapshot(filepath: str, vector_store: VectorStore, hnsw_index: HNSWIndex | None = None) -> bool:
        """Load state from a binary snapshot file into vector store and HNSW index.
        
        Args:
            filepath: Snapshot file path.
            vector_store: Target VectorStore instance to hydrate.
            hnsw_index: Target HNSWIndex instance to hydrate.

        Returns:
            bool: True if snapshot loaded successfully, False if snapshot file does not exist.
        """
        if not os.path.exists(filepath):
            return False

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        vector_store.dimension = state["dimension"]
        vector_store._vectors = state["vectors"]
        vector_store._payloads = state["payloads"]
        vector_store._tombstones = state["tombstones"]

        if hnsw_index and state.get("hnsw_graph") is not None:
            hnsw_index.graph = state["hnsw_graph"]
            hnsw_index.node_levels = state["hnsw_node_levels"]
            hnsw_index.entry_point = state["hnsw_entry_point"]
            hnsw_index.max_layer = state["hnsw_max_layer"]

        return True
