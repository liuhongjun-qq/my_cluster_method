HoSig-Align I vs HoSig+Louvain comparison

Files:
- compare_hosig_fusion.py: reproducible standard-library Python comparison script
- comparison_46_batches.csv: batch-aligned metrics and deltas (fusion minus HoSig)
- comparison_summary.csv: all-batch and multi-truth-class summary
- comparison_summary.json: full summary including valid denominators and caveats

Run on Windows (from directory with both JSON inputs and Python script):
python compare_hosig_fusion.py --hosig "report(2).json" --fusion "report(1).json" --output comparison_results

Meaning:
- delta_* is fusion minus HoSig; positive is preferable for CH, Silhouette, ARI, NMI, Modularity
- empty CH/Silhouette means unavailable (e.g. original HoSig single-cluster case)
- winning counts use only batches for which BOTH values are finite
- CH=1.0 values need audit as potential fallback, and must not be silently assumed to be scientifically valid
- when truth_class_count=1, ARI/NMI do not measure discrimination between attack classes
- timing is displayed but not used to claim speed improvements: the measured scopes differ
- IIR is omitted due to differing signature versus centroid entities
