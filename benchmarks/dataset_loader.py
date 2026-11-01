import numpy as np


def get_sift_128_dataset(num_vectors: int = 1000, num_queries: int = 20, dimension: int = 128, seed: int = 42):
    """Generate high-variance, challenging 128-dimensional benchmark dataset (SIFT-128 characteristics).
    
    Uses multi-centroid Gaussian mixture model with high noise variance and isotropic background
    components to produce realistic ANN recall curves without immediate saturation.
    """
    rng = np.random.default_rng(seed)
    
    # 50 Cluster centers across 128 dimensions
    num_clusters = 50
    centroids = rng.standard_normal((num_clusters, dimension)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # Assign vectors with high noise variance (0.45)
    assignments = rng.choice(num_clusters, size=num_vectors)
    noise = rng.standard_normal((num_vectors, dimension)).astype(np.float32) * 0.45
    background = rng.uniform(-0.2, 0.2, size=(num_vectors, dimension)).astype(np.float32)
    
    base_vectors = centroids[assignments] + noise + background
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)

    # Query vectors with offset and noise
    query_assignments = rng.choice(num_clusters, size=num_queries)
    query_noise = rng.standard_normal((num_queries, dimension)).astype(np.float32) * 0.50
    query_vectors = centroids[query_assignments] + query_noise
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Metadata payloads
    payloads = []
    for i in range(num_vectors):
        selectivity_val = float(rng.uniform(0.0, 1.0))
        payloads.append({
            "id": i,
            "selectivity_score": selectivity_val,
            "category": f"cat_{i % 10}"
        })

    return base_vectors.astype(np.float32), query_vectors.astype(np.float32), payloads
