import pytest
from fastapi.testclient import TestClient

from vecdb.server.api import app, collections
from vecdb.db.collection import Collection
from vecdb.core.distance import DistanceMetric


@pytest.fixture
def client():
    collections.clear()
    return TestClient(app)


def test_rest_api_endpoints(client):
    # Create collection
    res = client.post("/collections", json={"name": "test_app", "dimension": 3, "metric": "euclidean"})
    assert res.status_code == 201

    # Insert vector
    v_data = {"id": "item1", "vector": [1.0, 2.0, 3.0], "payload": {"tag": "A"}}
    res = client.post("/collections/test_app/vectors", json=v_data)
    assert res.status_code == 200

    # Batch insert
    batch_data = {
        "vectors": [
            {"id": "item2", "vector": [1.1, 2.1, 3.1], "payload": {"tag": "A"}},
            {"id": "item3", "vector": [10.0, 10.0, 10.0], "payload": {"tag": "B"}}
        ]
    }
    res = client.post("/collections/test_app/vectors/batch", json=batch_data)
    assert res.status_code == 200

    # Get vector
    res = client.get("/collections/test_app/vectors/item1")
    assert res.status_code == 200
    assert res.json()["payload"]["tag"] == "A"

    # Search
    search_payload = {
        "query": [1.0, 2.0, 3.0],
        "k": 2,
        "index_type": "hnsw"
    }
    res = client.post("/collections/test_app/search", json=search_payload)
    assert res.status_code == 200
    assert len(res.json()["results"]) == 2

    # Stats
    res = client.get("/collections/test_app/stats")
    assert res.status_code == 200
    assert res.json()["total_vectors"] == 3

    # Metrics
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "vecdb_query_count_total" in res.text
