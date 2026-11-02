"""HoSig content graph -> Python GVE-style multilevel Louvain -> event communities."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from hosig.config import Config
from hosig._01data import read_events, micro_batches
from hosig._02encoding import ContentEncoder
from hosig._02encoding_hybrid import HybridContentEncoder
from hosig._03forest import RandomPartitionForest
from hosig._04graph import mutual_topk_graph
from hosig.gve_louvain_python import detect_communities
from hosig.fusion_evaluation import evaluate_communities


def run(input_file, output, limit=None, batch_size=512, top_k=20,
        trees=64, resolution=1.0, seed=42, silhouette_sample=2000,
        encoder_type='original', semantic_model='sentence-transformers/all-MiniLM-L6-v2',
        bow_weight=1.0 / 3.0, numeric_weight=1.0 / 3.0,
        semantic_weight=1.0 / 3.0, semantic_device=None, semantic_aliases_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(batch_size=batch_size, top_k=top_k, trees=trees, seed=seed)

    # Keep the original encoder available as a reproducible baseline.
    encoder_fit_seconds = 0.0
    if encoder_type == 'original':
        encoder = ContentEncoder(config)
        encoder_info = {
            'name': 'Original HoSig hashed BOW + IP-prefix encoding',
            'output_dimension': config.embedding_dim,
        }
        events = read_events(input_file, data_format='csv', limit=limit)
    elif encoder_type == 'hybrid':
        semantic_aliases = None
        if semantic_aliases_path:
            with open(semantic_aliases_path, encoding='utf-8') as alias_file:
                semantic_aliases = json.load(alias_file)
            if not isinstance(semantic_aliases, dict):
                raise ValueError('--semantic-aliases must point to a JSON object')
        encoder = HybridContentEncoder(
            config,
            semantic_model_name=semantic_model,
            bow_weight=bow_weight,
            numeric_weight=numeric_weight,
            semantic_weight=semantic_weight,
            semantic_device=semantic_device,
            semantic_aliases=semantic_aliases,
        )
        print('Fitting hybrid encoder on event content (labels/timestamps are not used)...', flush=True)
        fit_start = perf_counter()
        # First pass: learn numeric scaling statistics and cache unique category
        # embeddings. Second pass below processes the actual micro-batches.
        encoder.fit(read_events(input_file, data_format='csv', limit=limit))
        encoder_fit_seconds = round(perf_counter() - fit_start, 4)
        encoder_info = encoder.info()
        print(
            f"Encoder fitted: numeric_fields={len(encoder.numeric_fields)} "
            f"semantic_fields={len(encoder.semantic_fields)} "
            f"input_dim={encoder.input_dim} output_dim={config.embedding_dim} "
            f"fit_seconds={encoder_fit_seconds}",
            flush=True,
        )
        events = read_events(input_file, data_format='csv', limit=limit)
    else:
        raise ValueError("encoder_type must be 'original' or 'hybrid'")
    results = []
    evaluation_rows = []
    local_dir = output / 'local'
    local_dir.mkdir(parents=True, exist_ok=True)
    global_row = 0
    with (output / 'event_communities.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['event_index', 'batch_id', 'event_id', 'timestamp', 'src_ip',
                         'dst_ip', 'ground_truth_label', 'community_id', 'local_community_id'])
        for batch_id, batch in enumerate(micro_batches(events, batch_size)):
            start = perf_counter()
            embeddings = encoder.transform(batch)
            forest = RandomPartitionForest(config.trees, config.leaf_size, config.seed).fit(embeddings)
            graph, half_lives = mutual_topk_graph(forest, np.array([e.timestamp for e in batch]), config)
            community = detect_communities(graph, resolution=resolution, seed=seed)
            # Save independent HoSig-style batch artifacts before proceeding.
            batch_dir = local_dir / f'batch_{batch_id:04d}'
            batch_dir.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(batch_dir / 'graph.npz', graph)
            np.savez_compressed(batch_dir / 'representations.npz', embeddings=embeddings)
            assignments = []
            # Batch-local groups only: a prefix makes IDs globally unique but DOES NOT link batches.
            for i, e in enumerate(batch):
                local = int(community.labels[i])
                writer.writerow([global_row, batch_id, e.event_id, e.timestamp, e.src_ip,
                                 e.dst_ip, e.label, f'b{batch_id}_c{local}', local])
                assignments.append({'batch_event_index': i, 'event_index': global_row,
                                    'event_id': e.event_id, 'timestamp': float(e.timestamp),
                                    'src_ip': e.src_ip, 'dst_ip': e.dst_ip,
                                    'ground_truth_label': e.label,
                                    'community_id': f'b{batch_id}_c{local}',
                                    'local_community_id': local})
                global_row += 1
            (batch_dir / 'assignments.json').write_text(
                json.dumps({'batch_id': batch_id, 'events': assignments},
                           indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            scores = evaluate_communities(
                embeddings, community.labels, [e.label for e in batch],
                modularity=community.modularity, seed=seed + batch_id,
                silhouette_sample=silhouette_sample,
            )
            evaluation_rows.append({'batch_id': batch_id, **{k: v for k, v in scores.items() if k != 'centroid_iir' and k != 'truth_label_counts'},
                                    'centroid_iir_k_median': scores['centroid_iir']['iir_k_median'],
                                    'centroid_iir_norm_median': scores['centroid_iir']['iir_norm_median']})
            info = {'batch_id': batch_id, 'events': len(batch),
                    'communities': community.n_communities,
                    'undirected_edges': int(graph.nnz // 2),
                    'modularity': community.modularity,
                    'louvain_levels': community.levels,
                    'levels_modularity': community.levels_modularity,
                    'half_lives': half_lives, 'evaluation': scores,
                    'artifact_dir': f'local/batch_{batch_id:04d}',
                    'seconds': round(perf_counter() - start, 4)}
            results.append(info)
            print(f"batch={batch_id:04d} events={len(batch)} communities={community.n_communities} "
                  f"Q={community.modularity:.5f} levels={community.levels}", flush=True)
    with (output / 'evaluation_batches.csv').open('w', newline='', encoding='utf-8-sig') as file:
        if evaluation_rows:
            writer = csv.DictWriter(file, fieldnames=list(evaluation_rows[0]))
            writer.writeheader()
            writer.writerows(evaluation_rows)
    summary = {'algorithm': 'HoSig graph + Python multilevel Louvain',
               'note': 'Independent micro-batches; no cross-batch community tracking',
               'total_events': global_row, 'batches': results, 'config': config.to_dict(),
               'encoder_type': encoder_type, 'encoder': encoder_info,
               'encoder_fit_seconds': encoder_fit_seconds,
               'resolution': resolution,
               'evaluation_note': 'Metrics are computed within each independent batch; centroid IIR is descriptive and not HoSig-signature IIR.',
               'evaluation_batches_csv': 'evaluation_batches.csv',
               'local_batch_dir': 'local',
               'saved_per_batch': ['assignments.json', 'graph.npz', 'representations.npz']}
    (output / 'report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Results:', output.resolve())
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='HoSig-Align I graph + Python Louvain fusion')
    p.add_argument('--input', default='data/iot23_ready.csv')
    p.add_argument('--output', default='outputs/fusion')
    p.add_argument('--limit', type=int, default=None, help='Number of events to read; default: all')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--trees', type=int, default=64)
    p.add_argument('--resolution', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--silhouette-sample', type=int, default=2000)
    p.add_argument('--encoder', choices=['original', 'hybrid'], default='original',
                   help='original=original HoSig BOW baseline; hybrid= BOW + numeric + semantic encoding')
    p.add_argument('--semantic-model', default='sentence-transformers/all-MiniLM-L6-v2',
                   help='SentenceTransformer model name or local model directory')
    p.add_argument('--bow-weight', type=float, default=1.0 / 3.0)
    p.add_argument('--numeric-weight', type=float, default=1.0 / 3.0)
    p.add_argument('--semantic-weight', type=float, default=1.0 / 3.0)
    p.add_argument('--semantic-device', default=None, help='Optional SentenceTransformer device, e.g. cpu or cuda')
    p.add_argument('--semantic-aliases', default=None, help='Optional JSON file with validated field-specific semantic aliases')
    a = p.parse_args()
    run(a.input, a.output, a.limit, a.batch_size, a.top_k, a.trees, a.resolution,
        a.seed, a.silhouette_sample, a.encoder, a.semantic_model,
        a.bow_weight, a.numeric_weight, a.semantic_weight, a.semantic_device,
        a.semantic_aliases)
