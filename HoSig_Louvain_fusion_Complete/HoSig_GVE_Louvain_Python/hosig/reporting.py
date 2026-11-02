# 输出 JSON、CSV 报告和图像
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)


def environment_info():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "networkx", "matplotlib")},
    }

'''
上图：候选切分得分：得分曲线 + 惩罚线 + 最终边界线
下图：同源块分配结果谱排序位置 + 每个位置的块编号
'''
def plot_boundaries(result, path):
    # 
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(result.segmentation.costs, label="C(b)")
    axes[0].axhline(result.segmentation.penalty, linestyle="--", color="red", label="penalty")
    for boundary in result.segmentation.boundaries:
        axes[0].axvline(boundary, color="grey", alpha=0.4)
    axes[0].set_ylabel("Contrastive score")
    axes[0].legend()
    axes[1].scatter(np.arange(len(result.order)), result.labels[result.order], s=8)
    axes[1].set_xlabel("Fiedler-ordered position (not chronological time)")
    axes[1].set_ylabel("Block ID")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_benchmark_csv(path, records):
    fields = (
        "method", "top_k", "client_count", "client_id", "batch_id", "n_events", "n_clusters",
        "noise_rate", "truth_class_count", "ch", "silhouette", "modularity", "ari", "nmi", "fit_seconds",
        "pipeline_seconds", "iir_k_mean", "iir_norm_mean",
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for key in ("iir_k", "iir_norm"):
                values = [value for value in record["boundary_clarity"][key] if value is not None]
                row[f"{key}_mean"] = float(np.mean(values)) if values else None
            writer.writerow(row)


def plot_benchmark(records, output):
    output = Path(output)
    methods = list(dict.fromkeys(record["method"] for record in records))
    top_values = sorted({record["top_k"] for record in records})
    client_counts = sorted({record["client_count"] for record in records})
    figure, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    for axis, method in zip(axes[0], methods):
        width = 0.8 / len(client_counts)
        positions = np.arange(len(top_values))
        for offset, client_count in enumerate(client_counts):
            values = []
            for top_k in top_values:
                group = [record["ch"] for record in records if record["method"] == method and record["top_k"] == top_k and record["client_count"] == client_count and record["ch"] is not None]
                values.append(float(np.mean(group)) if group else np.nan)
            axis.bar(positions + offset * width, values, width=width, label=f"Cn={client_count}")
        axis.set_xticks(positions + width * (len(client_counts) - 1) / 2, top_values)
        axis.set_xlabel("Top-k")
        axis.set_title(method)
        axis.set_ylabel("Mean per-batch CH")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ch_comparison.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(2, len(client_counts), figsize=(6 * len(client_counts), 8), squeeze=False)
    for column, client_count in enumerate(client_counts):
        for row, metric in enumerate(("iir_k", "iir_norm")):
            data, labels = [], []
            for top_k in top_values:
                for method in methods:
                    values = [
                        value for record in records
                        if record["method"] == method and record["client_count"] == client_count and record["top_k"] == top_k
                        for value in record["boundary_clarity"][metric] if value is not None
                    ]
                    if values:
                        data.append(values)
                        labels.append(f"{method}\nk={top_k}")
            axis = axes[row, column]
            if data:
                axis.boxplot(data, tick_labels=labels, showfliers=False)
                axis.tick_params(axis="x", labelrotation=70, labelsize=7)
            axis.set_title(f"{metric}, Cn={client_count}")
            axis.set_ylabel("Lower is better")
    figure.tight_layout()
    figure.savefig(output / "boundary_clarity.png", dpi=150)
    plt.close(figure)
