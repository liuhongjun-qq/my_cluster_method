# HoSig-Align I + Python Louvain (integrated source)

## What this does

`IoT-23 CSV -> HoSig ContentEncoder -> RandomPartitionForest -> dual temporal decay -> mutual Top-k weighted CSR graph -> Python multilevel Louvain -> event community labels`.

The original `hosig` modules from the uploaded ZIP are preserved. `run_fusion.py` is the new **single integrated pipeline**. `hosig/gve_louvain_python.py` adapts the **GVE-Louvain algorithmic stages** (local node moves, aggregation, modularity optimization) using NetworkX's Python Louvain engine; **this is not an exact line-by-line port of the C++ OpenMP/parallel implementation**, and does not claim equivalent speed or identical output.

The fusion deliberately does **not** call HoSig's spectral sorting, change-point segmentation or signature extraction. Instead it substitutes Louvain at the graph output, as requested. It currently processes each micro-batch independently: community IDs carry batch prefixes and do not indicate cross-batch matching. Sliding-window maintenance, confidence scores and budget adaptation are not implemented.

## Windows PowerShell

From extracted project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-fusion.txt
.\.venv\Scripts\python.exe run_fusion.py --input iot23_ready.csv --limit 256 --output outputs/smoke_256
```

To process all records (micro-batches of 512, no cross-batch linking):

```powershell
.\.venv\Scripts\python.exe run_fusion.py --input iot23_ready.csv --limit 23145 --batch-size 512 --output outputs/all_batches
```

Outputs:
- `event_communities.csv`: label of every event and its original ground-truth label for later evaluation
- `graph_batch_XXXX.npz`: the exact weighted sparse graph supplied to Louvain
- `report.json`: per-batch modularity Q, community counts, number of levels, durations

Do not interpret community IDs as ground-truth attack classes; they are unsupervised predictions.

## Complete evaluation edition (updated)

`run_fusion.py` now automatically writes per-batch CH, silhouette, ARI, NMI, weighted modularity, per-batch runtime and numbers of communities, plus **descriptive centroid IIR**. It is not HoSig RFF-signature IIR: in this fusion stage no HoSig block signatures are generated. Thus do not treat the IIR figures as directly comparable to HoSig paper signature-IIR results. Each batch is evaluated separately; rows cannot be assumed to share community identities across batches.

Install: `python -m pip install -r requirements-fusion.txt` (now includes scikit-learn).

For data placed in the `data` directory, from project root in Windows PowerShell:

```powershell
python run_fusion.py --input data/iot23_ready.csv --limit 256 --output outputs/smoke_256
```

For whole dataset, batch-local evaluation (no across-batch community tracking):

```powershell
python run_fusion.py --input data/iot23_ready.csv --limit 23145 --batch-size 512 --output outputs/all_batches
```

Outputs:
- `outputs/<name>/report.json`: full metrics nested in `batches[].evaluation`.
- `outputs/<name>/evaluation_batches.csv`: batch-level scalar metric summary.
- `outputs/<name>/event_communities.csv`: per-event community IDs and ground-truth labels.
- `outputs/<name>/graph_batch_XXXX.npz`: the exact weighted graph used by Louvain.

`CH`/`silhouette` measure embedding-space separation (not graph quality), `ARI`/`NMI` compare with supplied labels (IoT-23 may use broad labels), `modularity` measures graph community structure. Null values indicate metric undefined or unavailable. Do NOT average batch ARI/NMI and call it a single global partition score. To compare against original baselines, recompute all with the *same* events, labels, embeddings, batch and graph policies.


## Per-batch artifacts (HoSig-style layout; updated)

The default `--batch-size` is **512** events. The default `--limit` is now **all events**, rather than 256. For every micro-batch the program saves:

```text
outputs/fusion/
  report.json
  evaluation_batches.csv
  event_communities.csv
  local/
    batch_0000/
      assignments.json       # each event's assigned community and ground-truth label
      graph.npz              # exact SciPy CSR weighted event graph input to Louvain
      representations.npz    # content-encoder embeddings for batch events
    batch_0001/
      assignments.json
      graph.npz
      representations.npz
    ...
```

No `signatures.jsonl` or `first_batch_boundaries.png` is generated.

From the project root:

```powershell
python run_fusion.py --input data/iot23_ready.csv --batch-size 512 --output outputs/demo
```

For a quick test, add `--limit 512` (exactly one full micro-batch).

Load a graph later with:

```python
from scipy import sparse
G = sparse.load_npz('outputs/demo/local/batch_0000/graph.npz')
print(G.shape, G.nnz)
```

Batches are processed independently; the saved graphs are per-batch graphs, not a persistent cross-batch graph.
