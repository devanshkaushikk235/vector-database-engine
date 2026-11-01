"""Tests for Product Quantization (PQ) index.

Verifies:
  - Training produces codebooks of correct shape.
  - Encoding returns (N, M) uint8 codes.
  - ADC search returns results sorted by approximate distance.
  - Lazy training trigger fires after min_train_size insertions.
  - m_subspaces validation at Collection creation time.
  - search falls back to Flat when PQ is not yet trained.
"""

import numpy as np
import pytest

from vecdb.core.pq_index import ProductQuantizer
from vecdb.core.vector_store import VectorStore
from vecdb.db.collection import Collection
from vecdb.core.distance import DistanceMetric
from vecdb.core.query_engine import IndexType


DIM = 32
M = 4
K = 16
N_TRAIN = K * 4  # enough to trigger training (min_train_size = K * 2)


@pytest.fixture
def vector_store():
    return VectorStore(dimension=DIM)


@pytest.fixture
def pq(vector_store):
    pq = ProductQuantizer(M=M, K=K, dimension=DIM, vector_store=vector_store, seed=42)
    pq.min_train_size = N_TRAIN  # override for speed
    return pq


@pytest.fixture
def training_matrix():
    rng = np.random.default_rng(0)
    return rng.random((N_TRAIN, DIM)).astype(np.float32)


class TestProductQuantizerInit:
    def test_dimension_not_divisible_raises(self, vector_store):
        with pytest.raises(ValueError, match="divisible"):
            ProductQuantizer(M=3, K=16, dimension=32, vector_store=vector_store)

    def test_k_exceeds_uint8_raises(self, vector_store):
        with pytest.raises(ValueError, match="256"):
            ProductQuantizer(M=4, K=512, dimension=32, vector_store=vector_store)

    def test_not_trained_by_default(self, pq):
        assert not pq.is_trained

    def test_sub_dim_correct(self, pq):
        assert pq.sub_dim == DIM // M


class TestCodebookTraining:
    def test_codebook_shape_after_train(self, pq, training_matrix):
        pq.train(training_matrix)
        assert pq.codebooks is not None
        assert pq.codebooks.shape == (M, K, DIM // M)

    def test_is_trained_flag_set(self, pq, training_matrix):
        pq.train(training_matrix)
        assert pq.is_trained

    def test_codebooks_dtype_float32(self, pq, training_matrix):
        pq.train(training_matrix)
        assert pq.codebooks.dtype == np.float32


class TestEncoding:
    def test_encode_returns_correct_shape(self, pq, training_matrix):
        pq.train(training_matrix)
        codes = pq._encode_matrix(training_matrix)
        assert codes.shape == (N_TRAIN, M)

    def test_codes_are_uint8(self, pq, training_matrix):
        pq.train(training_matrix)
        codes = pq._encode_matrix(training_matrix)
        assert codes.dtype == np.uint8

    def test_codes_in_valid_range(self, pq, training_matrix):
        pq.train(training_matrix)
        codes = pq._encode_matrix(training_matrix)
        assert codes.min() >= 0
        assert codes.max() < K


class TestADCSearch:
    def setup_method(self):
        self.vs = VectorStore(dimension=DIM)
        self.pq = ProductQuantizer(M=M, K=K, dimension=DIM, vector_store=self.vs, seed=42)
        self.pq.min_train_size = N_TRAIN
        rng = np.random.default_rng(1)
        vecs = rng.random((N_TRAIN, DIM)).astype(np.float32)
        for i, v in enumerate(vecs):
            self.vs.add(f"v{i}", v)
        self.pq.train(vecs)
        for i in range(N_TRAIN):
            self.pq.encode_and_add(f"v{i}")
        self.query = rng.random(DIM).astype(np.float32)

    def test_search_returns_list(self):
        results = self.pq.search(self.query, k=5)
        assert isinstance(results, list)

    def test_search_returns_k_results(self):
        results = self.pq.search(self.query, k=5)
        assert len(results) <= 5

    def test_search_results_sorted_ascending(self):
        results = self.pq.search(self.query, k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores), "Results not sorted by ascending distance"

    def test_search_result_ids_are_strings(self):
        results = self.pq.search(self.query, k=5)
        for vid, score in results:
            assert isinstance(vid, str)
            assert isinstance(score, float)

    def test_search_respects_filter_ids(self):
        allowed = {f"v{i}" for i in range(10)}
        results = self.pq.search(self.query, k=5, filter_ids=allowed)
        for vid, _ in results:
            assert vid in allowed

    def test_search_empty_when_not_trained(self):
        fresh_vs = VectorStore(dimension=DIM)
        fresh_pq = ProductQuantizer(M=M, K=K, dimension=DIM, vector_store=fresh_vs)
        result = fresh_pq.search(np.zeros(DIM, dtype=np.float32), k=5)
        assert result == []


class TestLazyTrainingTrigger:
    def test_pq_trains_after_threshold(self, tmp_path):
        """PQ index should train automatically after min_train_size vectors are inserted."""
        coll = Collection(
            name="pq_lazy",
            dimension=DIM,
            metric=DistanceMetric.COSINE,
            storage_dir=str(tmp_path),
            m_subspaces=M,
        )
        # Override min_train_size for speed
        coll.query_engine.pq_index.min_train_size = N_TRAIN

        rng = np.random.default_rng(77)
        for i in range(N_TRAIN - 1):
            coll.insert(f"lz_{i}", rng.random(DIM).astype(np.float32))
        assert not coll.query_engine.pq_index.is_trained, "Should not train before threshold"

        coll.insert(f"lz_{N_TRAIN}", rng.random(DIM).astype(np.float32))
        assert coll.query_engine.pq_index.is_trained, "Should train after threshold"

    def test_pq_search_fallback_before_training(self, tmp_path):
        """Searching with PQ before training should fall back to Flat without error."""
        coll = Collection("pq_fallback", DIM, storage_dir=str(tmp_path), m_subspaces=M)
        rng = np.random.default_rng(5)
        for i in range(5):
            coll.insert(f"fb_{i}", rng.random(DIM).astype(np.float32))

        q = rng.random(DIM).astype(np.float32)
        results = coll.search(q, k=3, index_type=IndexType.PQ)
        assert isinstance(results, list)  # no crash; returns Flat results


class TestCollectionMSubspacesValidation:
    def test_invalid_m_subspaces_raises(self, tmp_path):
        """Collection creation with dim % m_subspaces != 0 must raise ValueError."""
        with pytest.raises(ValueError, match="m_subspaces"):
            Collection("bad_pq", dimension=32, storage_dir=str(tmp_path), m_subspaces=7)

    def test_valid_m_subspaces_accepted(self, tmp_path):
        coll = Collection("good_pq", dimension=32, storage_dir=str(tmp_path), m_subspaces=4)
        assert coll.m_subspaces == 4

    def test_default_m_subspaces_auto_computed(self, tmp_path):
        """Default m_subspaces should be min(8, dim // 16)."""
        coll = Collection("auto_pq_128", dimension=128, storage_dir=str(tmp_path))
        assert coll.m_subspaces == min(8, 128 // 16)

    def test_default_m_subspaces_small_dim(self, tmp_path):
        """For dim=16, default should be 1 since 16 // 16 = 1."""
        coll = Collection("auto_pq_16", dimension=16, storage_dir=str(tmp_path))
        assert coll.m_subspaces == min(8, 16 // 16)

    def test_error_message_shows_valid_options(self, tmp_path):
        """Error message should list valid divisors to help the user."""
        with pytest.raises(ValueError, match=r"Valid values"):
            Collection("hint_pq", dimension=32, storage_dir=str(tmp_path), m_subspaces=7)
