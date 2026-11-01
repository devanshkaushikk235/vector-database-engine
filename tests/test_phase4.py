import shutil
import tempfile
import pytest

from vecdb.core.distance import DistanceMetric
from vecdb.db.collection import Collection


@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


def test_wal_and_snapshot_recovery(temp_dir):
    coll_name = "test_coll"

    # Step 1: Create collection, insert data
    coll1 = Collection(name=coll_name, dimension=2, metric=DistanceMetric.EUCLIDEAN, storage_dir=temp_dir)
    coll1.insert("v1", [1.0, 0.0], {"name": "one"})
    coll1.insert("v2", [0.0, 1.0], {"name": "two"})
    coll1.insert("v3", [2.0, 2.0], {"name": "three"})

    # Delete v1
    coll1.delete("v1")

    # Step 2: Create snapshot
    coll1.create_snapshot()

    # Step 3: Insert more entries AFTER snapshot
    coll1.insert("v4", [3.0, 3.0], {"name": "four"})

    # Step 4: Simulate crash & instantiate new collection instance from same storage dir
    coll2 = Collection(name=coll_name, dimension=2, metric=DistanceMetric.EUCLIDEAN, storage_dir=temp_dir)

    # Verify state recovered from snapshot + replayed WAL entry
    assert coll2.get("v1") is None  # Deleted
    assert coll2.get("v2")["payload"]["name"] == "two"
    assert coll2.get("v3")["payload"]["name"] == "three"
    assert coll2.get("v4")["payload"]["name"] == "four"

    # Verify search on recovered collection
    results = coll2.search([0.0, 1.0], k=2)
    assert len(results) == 2
    assert results[0]["id"] == "v2"
