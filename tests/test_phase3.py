import numpy as np
import pytest

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.metadata import evaluate_filter, get_matching_ids
from vecdb.core.query_engine import QueryEngine, FilterStrategy, IndexType


def test_metadata_operator_evaluation():
    payload = {
        "brand": "nike",
        "price": 120,
        "rating": 4.5,
        "tags": ["running", "shoes"]
    }

    assert evaluate_filter(payload, {"price": {"gt": 100, "lte": 150}})
    assert evaluate_filter(payload, {"brand": "nike"})
    assert evaluate_filter(payload, {"brand": {"neq": "adidas"}})
    assert evaluate_filter(payload, {"rating": {"gte": 4.0}})
    assert evaluate_filter(payload, {"tags": {"in": ["shoes", "clothing"]}})
    assert evaluate_filter(payload, {"brand": {"nin": ["puma", "reebok"]}})
    assert not evaluate_filter(payload, {"price": {"lt": 50}})

    # Logical $and and $or
    assert evaluate_filter(payload, {"$or": [{"price": {"lt": 50}}, {"brand": "nike"}]})
    assert evaluate_filter(payload, {"$and": [{"price": {"gt": 100}}, {"rating": {"gt": 4.0}}]})


def test_query_engine_pre_and_post_filtering():
    store = VectorStore(dimension=2)
    engine = QueryEngine(store, metric=DistanceMetric.EUCLIDEAN)

    # Insert 10 vectors
    for i in range(10):
        vec = [float(i), 0.0]
        category = "even" if i % 2 == 0 else "odd"
        engine.insert(f"item_{i}", vec, {"idx": i, "category": category})

    # Filter query: category == "even"
    filter_q = {"category": "even"}

    # Pre-filtering
    pre_results = engine.search([0.0, 0.0], k=3, filter_query=filter_q, filter_strategy=FilterStrategy.PRE_FILTER, index_type=IndexType.FLAT)
    assert len(pre_results) == 3
    for res in pre_results:
        assert res["payload"]["category"] == "even"
    assert [r["id"] for r in pre_results] == ["item_0", "item_2", "item_4"]

    # Post-filtering
    post_results = engine.search([0.0, 0.0], k=3, filter_query=filter_q, filter_strategy=FilterStrategy.POST_FILTER, index_type=IndexType.FLAT)
    assert len(post_results) == 3
    for res in post_results:
        assert res["payload"]["category"] == "even"
