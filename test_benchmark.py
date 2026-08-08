#!/usr/bin/env python3
"""
Tests for fiedler-universal benchmark functions.
Tests partition methods, graph generators, and metric computation.
"""

import numpy as np
import pytest
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import (
    fiedler_partition,
    kmeans_partition,
    spectral_partition,
    modularity_partition,
    random_partition,
    _normalize_laplacian,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_cluster_graph():
    """A simple graph with two obvious clusters."""
    n = 20
    adj = np.zeros((n, n))
    # Cluster 1: nodes 0-9, dense
    for i in range(10):
        for j in range(i + 1, 10):
            if np.random.random() > 0.3:
                adj[i, j] = 1
                adj[j, i] = 1
    # Cluster 2: nodes 10-19, dense
    for i in range(10, 20):
        for j in range(i + 1, 20):
            if np.random.random() > 0.3:
                adj[i, j] = 1
                adj[j, i] = 1
    # Bridge: 1 edge between clusters
    adj[0, 10] = 1
    adj[10, 0] = 1
    return adj


@pytest.fixture
def complete_graph():
    """A complete graph K_n."""
    n = 10
    adj = np.ones((n, n))
    np.fill_diagonal(adj, 0)
    return adj


@pytest.fixture
def empty_graph():
    """A graph with no edges."""
    return np.zeros((10, 10))


@pytest.fixture
def ground_truth_labels():
    """Ground truth for two_cluster_graph."""
    return np.array([0] * 10 + [1] * 10)


# ---------------------------------------------------------------------------
# _normalize_laplacian
# ---------------------------------------------------------------------------

class TestNormalizeLaplacian:
    def test_returns_square_matrix(self, two_cluster_graph):
        L = _normalize_laplacian(two_cluster_graph)
        assert L.shape == two_cluster_graph.shape

    def test_diagonal_entries(self, two_cluster_graph):
        L = _normalize_laplacian(two_cluster_graph)
        # Diagonal of L_norm should be 1 for connected nodes
        # (since L_norm = I - D^{-1/2} A D^{-1/2}, diagonal = 1 - A_ii/D_ii = 1)
        n = two_cluster_graph.shape[0]
        for i in range(n):
            if two_cluster_graph[i].sum() > 0:
                assert abs(L[i, i] - 1.0) < 1e-6

    def test_empty_graph_no_crash(self, empty_graph):
        # Empty graph has zero-degree nodes, should add epsilon
        L = _normalize_laplacian(empty_graph)
        assert L.shape == (10, 10)

    def test_symmetric(self, two_cluster_graph):
        L = _normalize_laplacian(two_cluster_graph)
        assert np.allclose(L, L.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Fiedler Partition
# ---------------------------------------------------------------------------

class TestFiedlerPartition:
    def test_returns_labels(self, two_cluster_graph):
        labels = fiedler_partition(two_cluster_graph)
        assert len(labels) == two_cluster_graph.shape[0]
        assert set(np.unique(labels)).issubset({0, 1})

    def test_finds_two_clusters(self, two_cluster_graph, ground_truth_labels):
        from sklearn.metrics import adjusted_rand_score
        labels = fiedler_partition(two_cluster_graph)
        ari = adjusted_rand_score(ground_truth_labels, labels)
        assert ari > 0.5  # Should do reasonably well on obvious clusters

    def test_multi_cluster(self, two_cluster_graph):
        labels = fiedler_partition(two_cluster_graph, n_clusters=3)
        assert len(labels) == two_cluster_graph.shape[0]
        assert len(np.unique(labels)) <= 3

    def test_handles_single_node(self):
        adj = np.array([[0]])
        labels = fiedler_partition(adj)
        assert len(labels) == 1


# ---------------------------------------------------------------------------
# K-Means Partition
# ---------------------------------------------------------------------------

class TestKMeansPartition:
    def test_returns_labels(self, two_cluster_graph):
        labels = kmeans_partition(two_cluster_graph)
        assert len(labels) == two_cluster_graph.shape[0]

    def test_finds_clusters(self, two_cluster_graph, ground_truth_labels):
        from sklearn.metrics import adjusted_rand_score
        labels = kmeans_partition(two_cluster_graph)
        ari = adjusted_rand_score(ground_truth_labels, labels)
        # K-means on adjacency might not do as well, but should be positive
        assert ari >= 0.0


# ---------------------------------------------------------------------------
# Spectral Partition
# ---------------------------------------------------------------------------

class TestSpectralPartition:
    def test_returns_labels(self, two_cluster_graph):
        labels = spectral_partition(two_cluster_graph)
        assert len(labels) == two_cluster_graph.shape[0]

    def test_finds_clusters(self, two_cluster_graph, ground_truth_labels):
        from sklearn.metrics import adjusted_rand_score
        labels = spectral_partition(two_cluster_graph)
        ari = adjusted_rand_score(ground_truth_labels, labels)
        assert ari > 0.3  # Spectral should find these clusters


# ---------------------------------------------------------------------------
# Modularity Partition
# ---------------------------------------------------------------------------

class TestModularityPartition:
    def test_returns_labels(self, two_cluster_graph):
        np.random.seed(42)
        labels = modularity_partition(two_cluster_graph)
        assert len(labels) == two_cluster_graph.shape[0]

    def test_finds_clusters(self, two_cluster_graph, ground_truth_labels):
        from sklearn.metrics import adjusted_rand_score
        np.random.seed(42)
        labels = modularity_partition(two_cluster_graph)
        ari = adjusted_rand_score(ground_truth_labels, labels)
        assert ari > 0.0  # Modularity should do at least okay


# ---------------------------------------------------------------------------
# Random Partition
# ---------------------------------------------------------------------------

class TestRandomPartition:
    def test_returns_labels(self, two_cluster_graph):
        labels = random_partition(two_cluster_graph, n_clusters=3)
        assert len(labels) == two_cluster_graph.shape[0]
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_label_range(self):
        adj = np.zeros((20, 20))
        labels = random_partition(adj, n_clusters=5)
        assert all(0 <= l < 5 for l in labels)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_complete_graph_fiedler(self, complete_graph):
        labels = fiedler_partition(complete_graph)
        assert len(labels) == 10

    def test_empty_graph_no_crash(self, empty_graph):
        labels = fiedler_partition(empty_graph)
        assert len(labels) == 10

    def test_all_methods_return_same_length(self, two_cluster_graph):
        n = two_cluster_graph.shape[0]
        np.random.seed(42)
        
        f_labels = fiedler_partition(two_cluster_graph)
        k_labels = kmeans_partition(two_cluster_graph)
        s_labels = spectral_partition(two_cluster_graph)
        m_labels = modularity_partition(two_cluster_graph)
        r_labels = random_partition(two_cluster_graph)
        
        assert all(len(l) == n for l in [f_labels, k_labels, s_labels, m_labels, r_labels])

    def test_reproducibility(self, two_cluster_graph):
        """Fiedler partition should be deterministic given the same input."""
        labels1 = fiedler_partition(two_cluster_graph)
        labels2 = fiedler_partition(two_cluster_graph)
        # Labels might be permuted but ARI should be 1.0
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(labels1, labels2) == 1.0
