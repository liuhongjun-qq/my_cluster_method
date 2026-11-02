"""Python multilevel Louvain port of GVE-Louvain's *algorithmic stages*.

Implements: local modularity optimization, aggregation (coarsening), repeated levels.
Uses NetworkX's Python Louvain engine for well-tested modularity optimization;
not a line-for-line translation of GVE's asynchronous OpenMP implementation.
"""
from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy import sparse

@dataclass
class CommunityResult:
    labels: np.ndarray
    modularity: float
    n_communities: int
    levels: int
    levels_modularity: list[float]


def detect_communities(graph: sparse.spmatrix, resolution: float = 1.0,
                       seed: int = 42, threshold: float = 1e-7,
                       max_levels: int = 30) -> CommunityResult:
    """Run modularity-based multilevel Louvain on an undirected sparse weighted graph."""
    if resolution <= 0 or max_levels < 1:
        raise ValueError('resolution and max_levels must be positive')
    matrix = sparse.csr_matrix(graph, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Adjacency must be square')
    if matrix.data.size and (not np.isfinite(matrix.data).all() or (matrix.data < 0).any()):
        raise ValueError('Edge weights must be finite, nonnegative')
    if (matrix - matrix.T).nnz and not np.allclose((matrix - matrix.T).data, 0, atol=1e-10):
        raise ValueError('Graph must be symmetric')
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    n = matrix.shape[0]
    if n == 0:
        return CommunityResult(np.empty(0, dtype=int), 0.0, 0, 0, [])
    G = nx.from_scipy_sparse_array(matrix, edge_attribute='weight')
    if G.number_of_edges() == 0:
        return CommunityResult(np.arange(n, dtype=int), 0.0, n, 0, [])
    partitions = []
    scores = []
    # NetworkX performs local moves and graph aggregation internally on each level.
    for partition in nx.community.louvain_partitions(
        G, weight='weight', resolution=resolution, threshold=threshold, seed=seed
    ):
        partitions.append(partition)
        scores.append(float(nx.community.modularity(G, partition, weight='weight', resolution=resolution)))
        if len(partitions) >= max_levels:
            break
    final = partitions[-1]
    ordered = sorted((sorted(c) for c in final), key=lambda c: c[0])
    labels = np.empty(n, dtype=int)
    for label, members in enumerate(ordered):
        labels[members] = label
    return CommunityResult(labels, scores[-1], len(ordered), len(partitions), scores)
