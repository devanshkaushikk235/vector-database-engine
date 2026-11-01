import pytest
from fastapi.testclient import TestClient

from vecdb.server.api import app, collections
from vecdb.db.collection import Collection
from vecdb.core.distance import DistanceMetric


@pytest.fixture
def client():
    collections.clear()
    return TestClient(app)


def test_phase6_dashboard_endpoints(client):
    # 1. Test /ui HTML Dashboard
    res = client.get("/ui")
    assert res.status_code == 200
    assert "VecDB Monitoring & Operations Dashboard" in res.text
    assert "Chart.js" in res.text

    # 2. Create Collection
    res = client.post("/collections", json={"name": "dash_coll", "dimension": 4, "metric": "cosine"})
    assert res.status_code == 201

    # Insert vectors
    client.post("/collections/dash_coll/vectors", json={"id": "v1", "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"price": 50}})
    client.post("/collections/dash_coll/vectors", json={"id": "v2", "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"price": 150}})

    # Search with search by example (vector_id)
    res = client.post("/collections/dash_coll/search", json={"vector_id": "v1", "k": 2})
    assert res.status_code == 200
    assert len(res.json()["results"]) == 2

    # 3. Test /ui/stats JSON endpoint
    res = client.get("/ui/stats")
    assert res.status_code == 200
    stats = res.json()

    assert stats["total_collections"] == 1
    assert stats["total_vectors"] == 2
    assert stats["health"]["status"] in ["green", "yellow", "red"]
    assert "qps" in stats
    assert "latency_5m" in stats
    assert len(stats["recent_queries"]) >= 1
    assert stats["collections"][0]["name"] == "dash_coll"
    assert "payload_schema" in stats["collections"][0]
    assert stats["collections"][0]["payload_schema"]["price"] in ["int", "float"]

    # 4. Test Rebuild & Compaction endpoints
    res = client.post("/collections/dash_coll/rebuild")
    assert res.status_code == 200
    assert "started" in res.json()["message"]

    res = client.post("/collections/dash_coll/compact")
    assert res.status_code == 200
    assert "completed" in res.json()["message"]
