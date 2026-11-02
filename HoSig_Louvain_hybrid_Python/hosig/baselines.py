# 实现 HDBSCAN、Spectral、Louvain 等对比方法
import warnings

import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN, SpectralClustering

from .metrics import cluster_metrics


BASELINE_GRIDS = {
    "hdbscan": [{"min_cluster_size": size, "min_samples": samples} for size in (8, 16, 32) for samples in (4, 8)],
    "spectral": [{"n_clusters": count} for count in (2, 4, 8)],
    "louvain": [{"resolution": resolution} for resolution in (0.5, 1.0, 2.0)],
}


def fit_baseline(method, embeddings, graph, params=None, seed=42):
    params = params or {}
    count = len(embeddings)
    if method == "hdbscan":
        minimum = int(params.get("min_cluster_size", 16))
        if count < minimum:
            return np.full(count, -1, dtype=int), []
        labels = HDBSCAN(
            min_cluster_size=minimum, min_samples=int(params.get("min_samples", 8)),
            metric="euclidean", copy=True,
        ).fit_predict(embeddings)
        return labels, []
    if method == "spectral":
        clusters = min(int(params.get("n_clusters", 4)), count)
        if clusters < 2:
            return np.zeros(count, dtype=int), []
        if clusters == count or not graph.nnz:
            return np.arange(count), []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            labels = SpectralClustering(
                n_clusters=clusters, affinity="precomputed", assign_labels="kmeans",
                random_state=seed, n_init=10, eigen_solver="arpack",
            ).fit_predict(graph)
        return labels, [str(warning.message) for warning in captured]
    if method == "louvain":
        network = nx.from_scipy_sparse_array(graph)
        if not graph.nnz:
            return np.arange(count), []
        communities = nx.community.louvain_communities(
            network, weight="weight", resolution=float(params.get("resolution", 1.0)), seed=seed,
        )
        labels = np.empty(count, dtype=int)
        for cluster, nodes in enumerate(sorted(communities, key=lambda nodes: min(nodes))):
            labels[list(nodes)] = cluster
        return labels, []
    raise ValueError(f"Unknown baseline: {method}")


def tune_baseline(method, validation, seed=42, latency_budget=None):
    from time import perf_counter

    best_params = BASELINE_GRIDS[method][0]
    best_score = -np.inf
    trials = []
    for params in BASELINE_GRIDS[method]:
        start = perf_counter()
        labels, captured = fit_baseline(method, validation.embeddings, validation.graph, params, seed)
        duration = perf_counter() - start
        metrics = cluster_metrics(validation.embeddings, labels, validation.graph, seed=seed)
        quality = (metrics["silhouette"] or 0.0) + (metrics["modularity"] or 0.0) - metrics["noise_rate"]
        eligible = metrics["n_clusters"] >= 2 and (latency_budget is None or duration <= latency_budget)
        trials.append({
            "params": params, "quality": quality, "fit_seconds": duration,
            "eligible": eligible, "warnings": captured,
        })
        if eligible and quality > best_score:
            best_score, best_params = quality, params
    return best_params, {"trials": trials, "fallback": not np.isfinite(best_score)}
