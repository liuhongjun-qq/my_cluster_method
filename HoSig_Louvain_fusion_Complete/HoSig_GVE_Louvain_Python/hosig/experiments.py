from dataclasses import replace
from time import perf_counter

import numpy as np

from .baselines import fit_baseline, tune_baseline
from ._01data import micro_batches
from .metrics import cluster_metrics
from .pipeline import HoSigAlign


def benchmark(events, config, top_values=(5, 10, 20, 40), client_counts=(2, 4, 8), validation_fraction=0.2, entity_protocol="paper", latency_budget=None, progress=None):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not top_values or any(top_k < 1 for top_k in top_values):
        raise ValueError("top_values must contain positive integers")
    if not client_counts or any(count < 1 for count in client_counts):
        raise ValueError("client_counts must contain positive integers")
    if entity_protocol not in {"paper", "common-rff"}:
        raise ValueError("Unknown entity protocol")
    records, selection = [], []
    for client_count in client_counts:
        sites = [events[index::client_count] for index in range(client_count)]
        if any(len(site) < 4 for site in sites):
            raise ValueError("Each simulated site needs at least four events")
        for top_k in top_values:
            current = replace(config, top_k=top_k)
            model = HoSigAlign(current)
            for client_id, site in enumerate(sites):
                split = max(2, min(len(site) - 2, int(len(site) * validation_fraction)))
                validation = model.process_batch(site[:min(split, config.batch_size)], f"validation-{client_id}")
                selected = {}
                for method in ("hdbscan", "spectral", "louvain"):
                    selected[method], trials = tune_baseline(method, validation, current.seed, latency_budget)
                    selection.append({
                        "method": method, "top_k": top_k, "client_count": client_count,
                        "client_id": client_id, "selected": selected[method], **trials,
                        "validation_prefix_events": split, "validation_events_used": min(split, config.batch_size),
                        "test_events": len(site) - split,
                    })
                for batch_id, batch in enumerate(micro_batches(site[split:], current.batch_size)):
                    result = model.process_batch(batch, f"{client_count}-{top_k}-{client_id}-{batch_id}")
                    truth = [event.label for event in batch]
                    rff = model.rff.transform(result.embeddings)
                    for method in ("hosig", "hdbscan", "spectral", "louvain"):
                        if method == "hosig":
                            labels = result.labels
                            entities = np.asarray([signature.prototype for signature in result.signatures])
                            fit_time = sum(result.diagnostics["timings_seconds"][stage] for stage in ("spectral", "segmentation", "signatures"))
                            pipeline_time = result.diagnostics["timings_seconds"]["total"]
                            captured = result.diagnostics["warnings"]
                        else:
                            start = perf_counter()
                            labels, captured = fit_baseline(method, result.embeddings, result.graph, selected[method], current.seed)
                            fit_time = perf_counter() - start
                            preparation = result.diagnostics["timings_seconds"]["encoding"]
                            if method != "hdbscan":
                                preparation += sum(result.diagnostics["timings_seconds"][stage] for stage in ("forest", "graph"))
                            pipeline_time = preparation + fit_time
                            entity_features = rff if entity_protocol == "common-rff" else result.embeddings
                            entities = np.array([entity_features[labels == label].mean(axis=0) for label in np.unique(labels[labels >= 0])])
                        scores = cluster_metrics(result.embeddings, labels, result.graph, entities, truth, current.seed)
                        records.append({
                            "method": method, "top_k": top_k, "client_count": client_count,
                            "client_id": client_id, "batch_id": batch_id,
                            **scores, "fit_seconds": fit_time, "pipeline_seconds": pipeline_time,
                            "warnings": captured, "graph_mean_degree": result.diagnostics["mean_degree"],
                            "over_fit_budget": latency_budget is not None and fit_time > latency_budget,
                        })
                if progress:
                    progress(f"Finished Cn={client_count}, k={top_k}, client={client_id}")
    return {"records": records, "baseline_selection": selection, "entity_protocol": entity_protocol}
