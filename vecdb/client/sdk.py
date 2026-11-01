import json
from typing import Any
import requests


class VecDBClient:
    """Python Client SDK for interacting with VecDB REST Server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client for a target VecDB server URL.
        
        Args:
            base_url: Server root URL.
        """
        self.base_url = base_url.rstrip("/")

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> dict[str, Any]:
        """Create a new collection on the server."""
        url = f"{self.base_url}/collections"
        resp = requests.post(url, json={"name": name, "dimension": dimension, "metric": metric})
        resp.raise_for_status()
        return resp.json()

    def delete_collection(self, name: str) -> dict[str, Any]:
        """Delete a collection by name."""
        url = f"{self.base_url}/collections/{name}"
        resp = requests.delete(url)
        resp.raise_for_status()
        return resp.json()

    def insert(
        self,
        collection_name: str,
        vector_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Insert a single vector into a collection."""
        url = f"{self.base_url}/collections/{collection_name}/vectors"
        body = {"id": vector_id, "vector": vector, "payload": payload}
        resp = requests.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def insert_batch(
        self,
        collection_name: str,
        vectors: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Batch insert multiple vectors into a collection."""
        url = f"{self.base_url}/collections/{collection_name}/vectors/batch"
        body = {"vectors": vectors}
        resp = requests.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def get(self, collection_name: str, vector_id: str) -> dict[str, Any]:
        """Retrieve vector entry by ID."""
        url = f"{self.base_url}/collections/{collection_name}/vectors/{vector_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def delete(self, collection_name: str, vector_id: str) -> dict[str, Any]:
        """Delete a vector entry by ID."""
        url = f"{self.base_url}/collections/{collection_name}/vectors/{vector_id}"
        resp = requests.delete(url)
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        collection_name: str,
        query: list[float],
        k: int = 10,
        filter_query: dict[str, Any] | None = None,
        filter_strategy: str = "pre_filter",
        index_type: str = "hnsw",
        ef_search: int | None = None,
        n_probe: int | None = None
    ) -> dict[str, Any]:
        """Execute vector search query."""
        url = f"{self.base_url}/collections/{collection_name}/search"
        body = {
            "query": query,
            "k": k,
            "filter": filter_query,
            "filter_strategy": filter_strategy,
            "index_type": index_type,
            "ef_search": ef_search,
            "n_probe": n_probe
        }
        resp = requests.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def get_stats(self, collection_name: str) -> dict[str, Any]:
        """Get collection metrics and statistics."""
        url = f"{self.base_url}/collections/{collection_name}/stats"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def snapshot(self, collection_name: str) -> dict[str, Any]:
        """Trigger snapshot creation for a collection."""
        url = f"{self.base_url}/collections/{collection_name}/snapshot"
        resp = requests.post(url)
        resp.raise_for_status()
        return resp.json()

    def get_metrics(self) -> str:
        """Retrieve Prometheus formatted metrics."""
        url = f"{self.base_url}/metrics"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.text
