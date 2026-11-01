import numpy as np
import pytest

from vecdb.core.distance import DistanceMetric
from vecdb.core.vector_store import VectorStore
from vecdb.core.flat_index import FlatIndex
from vecdb.core.ivf_index import IVFIndex


def test_ivf_training_and_recall():
    np.random.seed(42)
    dim = 8
    num_vectors = 100

    store = VectorStore(dimension=dim)
    ivf = IVFIndex(store, metric=DistanceMetric.EUCLIDEAN, n_list=4, n_probe=2)
    flat = FlatIndex(store, metric=DistanceMetric.EUCLIDEAN)

    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    for i in range(num_vectors):
        vid = f"v_{i}"
        store.add(vid, vectors[i])

    ivf.train()
    assert ivf.is_trained
    assert len(ivf.centroids) == 4

    query = np.random.randn(dim).astype(np.float32)
    k = 5

    flat_res = flat.search(query, k=k)
    ivf_res = ivf.search(query, k=k)

    flat_ids = set(vid for vid, _ in flat_res)
    ivf_ids = set(vid for vid, _ in ivf_res)

    # Inverted file searching top 2 out of 4 probes should match majority of exact top 5
    intersection = flat_ids.intersection(ivf_ids)
    assert len(intersection) >= 3
