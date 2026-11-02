"""Metrics for HoSig -> Louvain event communities (per micro-batch).

CH/silhouette: content embeddings; ARI/NMI: labeled events; Q: graph.
IIR values below describe community *centroids*, not HoSig RFF signatures.
"""
import math
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler


MISSING = {'', '-', '?', 'none', 'nan', 'null', 'n/a', 'unknown'}


def _is_known(label):
    return label is not None and str(label).strip().lower() not in MISSING


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def centroid_iir(embeddings, labels, proportion=0.05):
    """Descriptive nearest-centroid statistics; lower is NOT universally better.

    Uses HoSig's boundary_clarity convention: reciprocal and median/distance.
    Not equivalent to IIR of separately computed HoSig signature prototypes.
    """
    ids = np.unique(labels)
    k = len(ids)
    result = {'entity_type': 'event_community_centroid', 'entity_count': int(k),
              'neighbor_rank': None, 'iir_k': [], 'iir_norm': [],
              'iir_k_median': None, 'iir_norm_median': None,
              'duplicate_entity_count': 0}
    if k < 2:
        return result
    centers = np.vstack([embeddings[labels == group].mean(axis=0) for group in ids])
    scaled = StandardScaler().fit_transform(centers)
    dist = cdist(scaled, scaled)
    np.fill_diagonal(dist, np.inf)
    rank = min(k - 1, max(1, math.ceil(proportion * k)))
    nearest = np.partition(dist, rank - 1, axis=1)[:, rank - 1]
    median = float(np.median(nearest))
    a = [_finite(1 / d) if d > 0 else None for d in nearest]
    b = [_finite(median / d) if d > 0 and median > 0 else None for d in nearest]
    result.update(neighbor_rank=int(rank), iir_k=a, iir_norm=b,
                  iir_k_median=float(np.median([x for x in a if x is not None])) if any(x is not None for x in a) else None,
                  iir_norm_median=float(np.median([x for x in b if x is not None])) if any(x is not None for x in b) else None,
                  duplicate_entity_count=int(np.sum(nearest == 0)))
    return result


def evaluate_communities(embeddings, labels, truth=None, modularity=None,
                         seed=42, silhouette_sample=2000, compute_iir=True):
    x = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2 or x.shape[0] != len(labels):
        raise ValueError('Embeddings must have shape (events, features), matching labels')
    if not np.all(np.isfinite(x)):
        raise ValueError('Embeddings contain non-finite values')
    n, k = len(labels), len(np.unique(labels))
    metrics = {'n_events': int(n), 'n_communities': int(k),
               'ch': None, 'silhouette': None, 'modularity': _finite(modularity) if modularity is not None else None,
               'ari': None, 'nmi': None, 'labeled_events': 0,
               'truth_class_count': 0, 'truth_label_counts': {},
               'centroid_iir': None}
    if 1 < k < n:
        metrics['ch'] = _finite(calinski_harabasz_score(x, labels))
        sample_size = min(int(silhouette_sample), n)
        # Avoid sklearn's sampled-subset failure when small communities vanish from sample.
        # For small batches use all events, otherwise stratify at least one point/community.
        if sample_size == n:
            subset = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            mandatory = np.array([rng.choice(np.flatnonzero(labels == group)) for group in np.unique(labels)])
            target = max(sample_size, len(mandatory) + 1)
            remainder = np.setdiff1d(np.arange(n), mandatory, assume_unique=False)
            extra = rng.choice(remainder, size=min(target - len(mandatory), len(remainder)), replace=False)
            subset = np.concatenate((mandatory, extra))
        labs = labels[subset]
        if 1 < len(np.unique(labs)) < len(labs):
            metrics['silhouette'] = _finite(silhouette_score(x[subset], labs))
    if truth is not None:
        if len(truth) != n:
            raise ValueError('Ground-truth label count does not match events')
        known = np.array([_is_known(y) for y in truth], dtype=bool)
        true = [str(y).strip() for y, include in zip(truth, known) if include]
        counts = dict(sorted(Counter(true).items()))
        metrics['labeled_events'] = len(true)
        metrics['truth_class_count'] = len(counts)
        metrics['truth_label_counts'] = counts
        if len(true) >= 2:
            metrics['ari'] = _finite(adjusted_rand_score(true, labels[known]))
            metrics['nmi'] = _finite(normalized_mutual_info_score(true, labels[known]))
    if compute_iir:
        metrics['centroid_iir'] = centroid_iir(x, labels)
    return metrics
