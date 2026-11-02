# -*- coding: utf-8 -*-
"""
多数据集：编码改进前后可视化比较
================================
支持：
1. 同时比较一个或多个数据集；
2. 每个数据集提供“改进前 JSON 报告”和“改进后 JSON 报告”；
3. 自动读取每个微批次的评价指标；
4. 同时统计全部批次和多类别批次（truth_class_count >= 2）；
5. 自动匹配相同 batch_id，进行配对比较；
6. 只输出多类别批次均值图、有效配对批次数图；
7. 为每个数据集输出全部批次和多类别批次的逐批次折线图及配对 CSV；
8. 改进后编码权重从 JSON 的 encoder.branch_weights 自动读取，不固定任何权重。

运行前安装：
    pip install pandas numpy matplotlib

示例（PowerShell）：
    python plot_encoding_comparison.py `
      --pair IoTID20 ".\\fusion_report(5).json" ".\\iotid20_semantic_015_085_fixed_report.json" `
      --output ".\\plot_comparison\\encoding_before_after"

添加第二个数据集时，再增加一个 --pair：
    python plot_encoding_comparison.py `
      --pair IoTID20 ".\\iotid_before.json" ".\\iotid_after.json" `
      --pair IoT-23 ".\\iot23_before.json" ".\\iot23_after.json" `
      --output ".\\plot_comparison\\encoding_before_after"

说明：
- ARI/NMI 在只有一个真实类别的批次上通常没有评价意义，因此代码会额外
  单独统计 truth_class_count >= 2 的多类别批次。
- 配对比较只使用改进前后都存在的 batch_id。
- 对 CH、Silhouette、Modularity、ARI、NMI，数值越高通常越好；
  对运行时间 seconds，数值越低越好。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 指标配置
# =========================

# 主要聚类质量指标：用于均值图和逐批次折线图
PLOT_METRICS = [
    "modularity",
    "ari",
    "nmi",
    "ch",
    "silhouette",
]

METRIC_TITLES = {
    "modularity": "Modularity（模块度）",
    "ari": "ARI",
    "nmi": "NMI",
    "ch": "Calinski-Harabasz（CH）",
    "silhouette": "Silhouette（轮廓系数）",
    "seconds": "Runtime（运行时间，秒）",
    "communities": "Community Count（社区数量）",
}

# 所有需要写入汇总 CSV 的指标
SUMMARY_METRICS = PLOT_METRICS + ["communities", "seconds"]

# 改进前橙色，改进后蓝色
COLOR_BEFORE = "#F28E2B"
COLOR_AFTER = "#4E79A7"

# 胜负图：改进前胜为橙色，平局为灰色，改进后胜为蓝色
COLOR_BEFORE_WIN = COLOR_BEFORE
COLOR_TIE = "#B8B8B8"
COLOR_AFTER_WIN = COLOR_AFTER


# =========================
# 2. 基础数据读取
# =========================

def to_float(value):
    """将数值转换为 float；无法转换或非有限值返回 NaN。"""
    try:
        if value is None:
            return np.nan
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_report(report_path):
    """读取一个实验 JSON，并整理成每行一个 batch 的 DataFrame。"""
    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"找不到报告文件：{report_path}")

    with report_path.open("r", encoding="utf-8-sig") as f:
        report = json.load(f)

    batches = report.get("batches")
    if not isinstance(batches, list):
        raise ValueError(f"{report_path.name} 中没有有效的 batches 列表。")

    rows = []

    for batch in batches:
        evaluation = batch.get("evaluation") or {}

        batch_id = batch.get("batch_id")
        if batch_id is None:
            continue

        truth_class_count = evaluation.get("truth_class_count")
        if truth_class_count is None:
            label_counts = evaluation.get("truth_label_counts")
            if isinstance(label_counts, dict):
                truth_class_count = len(label_counts)

        row = {
            "batch_id": batch_id,
            "events": to_float(
                batch.get("events", evaluation.get("n_events"))
            ),
            "truth_class_count": to_float(truth_class_count),
            "communities": to_float(
                evaluation.get("n_communities", batch.get("communities"))
            ),
            "modularity": to_float(
                evaluation.get("modularity", batch.get("modularity"))
            ),
            "ari": to_float(evaluation.get("ari")),
            "nmi": to_float(evaluation.get("nmi")),
            "ch": to_float(evaluation.get("ch")),
            "silhouette": to_float(evaluation.get("silhouette")),
            "seconds": to_float(batch.get("seconds")),
            "undirected_edges": to_float(batch.get("undirected_edges")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"{report_path.name} 中没有可用的批次记录。")

    if df["batch_id"].duplicated().any():
        duplicated = df.loc[df["batch_id"].duplicated(), "batch_id"].tolist()
        raise ValueError(
            f"{report_path.name} 存在重复 batch_id，例如：{duplicated[:5]}"
        )

    df["batch_id"] = pd.to_numeric(df["batch_id"], errors="coerce")
    df = df.dropna(subset=["batch_id"]).copy()
    df["batch_id"] = df["batch_id"].astype(int)
    df = df.sort_values("batch_id").reset_index(drop=True)

    return report, df


def infer_encoder_label(report, fallback):
    """
    从报告中自动识别编码方式。
    例如自动生成：BOW 0.15 + Semantic 0.85。
    不在代码中固定某一组权重。
    """
    encoder = report.get("encoder") or {}
    weights = encoder.get("branch_weights")

    if isinstance(weights, dict) and weights:
        parts = []
        for branch, weight in weights.items():
            weight = to_float(weight)
            if np.isfinite(weight) and weight > 0:
                parts.append(f"{branch.upper()} {weight:.2f}")

        if parts:
            return " + ".join(parts)

    encoder_type = report.get("encoder_type")
    if encoder_type:
        return str(encoder_type)

    return fallback


# =========================
# 3. 配对批次
# =========================

def prepare_pair(dataset_name, before_path, after_path,
                 before_label_override=None, after_label_override=None):

    before_report, before_df = read_report(before_path)
    after_report, after_df = read_report(after_path)

    before_label = (
        before_label_override
        if before_label_override
        else infer_encoder_label(before_report, "改进前（原始编码）")
    )
    after_label = (
        after_label_override
        if after_label_override
        else infer_encoder_label(after_report, "改进后")
    )

    # 只保留改进前后都存在的 batch_id，保证配对比较公平
    merged = before_df.merge(
        after_df,
        on="batch_id",
        how="inner",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    ).sort_values("batch_id").reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            f"{dataset_name}：改进前后没有相同的 batch_id，无法配对比较。"
        )

    # 检查前后批次事件数是否一致
    event_mismatch = (
        merged["events_before"].notna()
        & merged["events_after"].notna()
        & (merged["events_before"] != merged["events_after"])
    )

    # 多类别批次要求改进前、改进后都满足真实类别数 >= 2
    multi_mask = (
        (merged["truth_class_count_before"] >= 2)
        & (merged["truth_class_count_after"] >= 2)
    )

    class_count_mismatch = (
        merged["truth_class_count_before"].notna()
        & merged["truth_class_count_after"].notna()
        & (
            merged["truth_class_count_before"]
            != merged["truth_class_count_after"]
        )
    )

    info = {
        "dataset": dataset_name,
        "before_path": str(before_path),
        "after_path": str(after_path),
        "before_label": before_label,
        "after_label": after_label,
        "before_total_batches": len(before_df),
        "after_total_batches": len(after_df),
        "paired_batches": len(merged),
        "multi_class_paired_batches": int(multi_mask.sum()),
        "unmatched_before": len(before_df) - len(merged),
        "unmatched_after": len(after_df) - len(merged),
        "event_count_mismatch_batches": int(event_mismatch.sum()),
        "class_count_mismatch_batches": int(class_count_mismatch.sum()),
    }

    if info["unmatched_before"] or info["unmatched_after"]:
        print(
            f"[提示] {dataset_name}：前后报告批次数量不完全一致，"
            f"配对分析只使用共同 batch_id。"
        )

    if info["event_count_mismatch_batches"]:
        print(
            f"[警告] {dataset_name}：有 "
            f"{info['event_count_mismatch_batches']} 个配对批次的事件数不同。"
        )

    if info["class_count_mismatch_batches"]:
        print(
            f"[警告] {dataset_name}：有 "
            f"{info['class_count_mismatch_batches']} 个批次的真实类别数不同；"
            "多类别统计只保留前后都满足类别数 >= 2 的批次。"
        )

    return {
        "info": info,
        "data": merged,
        "multi_data": merged.loc[multi_mask].copy(),
    }


# =========================
# 4. 汇总统计
# =========================

def build_summary(pair_results):
    summary_rows = []
    count_rows = []

    for result in pair_results:
        info = result["info"]

        for subset_name, df in [
            ("全部配对批次", result["data"]),
            ("多类别配对批次", result["multi_data"]),
        ]:
            for metric in SUMMARY_METRICS:
                before_col = f"{metric}_before"
                after_col = f"{metric}_after"

                if before_col not in df.columns or after_col not in df.columns:
                    continue

                valid = df[[before_col, after_col]].dropna()
                n = len(valid)

                before_mean = valid[before_col].mean() if n else np.nan
                after_mean = valid[after_col].mean() if n else np.nan
                delta = after_mean - before_mean if n else np.nan

                if n and np.isfinite(before_mean) and abs(before_mean) > 1e-12:
                    relative_change = delta / abs(before_mean) * 100
                else:
                    relative_change = np.nan

                summary_rows.append({
                    "dataset": info["dataset"],
                    "subset": subset_name,
                    "metric": metric,
                    "before_label": info["before_label"],
                    "after_label": info["after_label"],
                    "paired_valid_n": n,
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "delta_after_minus_before": delta,
                    "relative_change_percent": relative_change,
                })

        # 批次数统计
        count_rows.append({
            "dataset": info["dataset"],
            "before_total_batches": info["before_total_batches"],
            "after_total_batches": info["after_total_batches"],
            "paired_batches": info["paired_batches"],
            "multi_class_paired_batches": info["multi_class_paired_batches"],
            "unmatched_before": info["unmatched_before"],
            "unmatched_after": info["unmatched_after"],
            "event_count_mismatch_batches": info["event_count_mismatch_batches"],
            "class_count_mismatch_batches": info["class_count_mismatch_batches"],
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(count_rows)


# =========================
# 5. 绘图通用函数
# =========================

def set_plot_style():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300


def add_bar_labels(ax, containers, fmt="{:.3f}"):
    """给柱状图添加数值标签。"""
    for container in containers:
        for bar in container:
            height = bar.get_height()
            if np.isfinite(height):
                ax.annotate(
                    fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )


def plot_metric_means(summary_df, dataset_names, subset_name,
                      output_path, title):
    """绘制五项指标的改进前后均值对比图。"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)

    for ax, metric in zip(axes, PLOT_METRICS):
        part = summary_df[
            (summary_df["subset"] == subset_name)
            & (summary_df["metric"] == metric)
        ]

        before_values = []
        after_values = []

        for dataset in dataset_names:
            row = part[part["dataset"] == dataset]
            if row.empty:
                before_values.append(np.nan)
                after_values.append(np.nan)
            else:
                before_values.append(row.iloc[0]["before_mean"])
                after_values.append(row.iloc[0]["after_mean"])

        x = np.arange(len(dataset_names))
        width = 0.36

        bars_before = ax.bar(
            x - width / 2,
            before_values,
            width,
            color=COLOR_BEFORE,
            label="改进前",
        )
        bars_after = ax.bar(
            x + width / 2,
            after_values,
            width,
            color=COLOR_AFTER,
            label="改进后",
        )

        ax.set_title(METRIC_TITLES[metric], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # 指标值标注，数据集较多时自动缩小字号
        if len(dataset_names) <= 5:
            add_bar_labels(ax, [bars_before, bars_after], fmt="{:.3f}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_batch_counts(count_df, output_path):
    """绘制共同配对批次和多类别配对批次数。"""
    fig, ax = plt.subplots(figsize=(max(8, len(count_df) * 1.5), 5))

    x = np.arange(len(count_df))
    width = 0.34

    b1 = ax.bar(
        x - width / 2,
        count_df["paired_batches"],
        width,
        color=COLOR_BEFORE,
        label="共同配对批次",
    )
    b2 = ax.bar(
        x + width / 2,
        count_df["multi_class_paired_batches"],
        width,
        color=COLOR_AFTER,
        label="多类别配对批次",
    )

    ax.set_title("各数据集有效配对批次数", fontsize=14, fontweight="bold")
    ax.set_ylabel("批次数")
    ax.set_xticks(x)
    ax.set_xticklabels(count_df["dataset"], rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    add_bar_labels(ax, [b1, b2], fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_batchwise(result, output_dir, subset_name, use_multi):
    """为单个数据集绘制五项指标的逐批次折线图。"""
    df = result["multi_data"] if use_multi else result["data"]
    info = result["info"]

    if df.empty:
        print(f"[提示] {info['dataset']} 没有符合条件的{subset_name}，跳过折线图。")
        return

    df = df.sort_values("batch_id")
    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(
        f"{info['dataset']}：{subset_name}逐批次指标比较",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    for ax, metric in zip(axes, PLOT_METRICS):
        before_col = f"{metric}_before"
        after_col = f"{metric}_after"

        ax.plot(
            df["batch_id"],
            df[before_col],
            color=COLOR_BEFORE,
            linewidth=1.0,
            alpha=0.85,
            label="改进前",
        )
        ax.plot(
            df["batch_id"],
            df[after_col],
            color=COLOR_AFTER,
            linewidth=1.0,
            alpha=0.85,
            label="改进后",
        )

        ax.set_ylabel(metric.upper())
        ax.set_title(METRIC_TITLES[metric], fontsize=10, loc="left")
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=8)

    axes[-1].set_xlabel("Batch ID")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    filename = "batchwise_multiclass.png" if use_multi else "batchwise_all_batches.png"
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6. 主程序
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="多数据集编码改进前后可视化比较"
    )

    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("DATASET", "BEFORE_JSON", "AFTER_JSON"),
        help="数据集名称、改进前报告、改进后报告；可以重复添加多个 --pair。",
    )
    parser.add_argument(
        "--output",
        default="plot_comparison/encoding_before_after",
        help="图片和 CSV 的输出目录。",
    )
    parser.add_argument(
        "--before-label",
        default=None,
        help="统一指定改进前的显示名称；不指定时优先自动识别。",
    )
    parser.add_argument(
        "--after-label",
        default=None,
        help="统一指定改进后的显示名称；不指定时优先从 JSON 自动读取分支权重。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_plot_style()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    pair_results = []
    seen_names = set()

    print("=" * 72)
    print("多数据集编码改进前后比较")
    print("=" * 72)

    for dataset_name, before_path, after_path in args.pair:
        if dataset_name in seen_names:
            raise ValueError(f"数据集名称重复：{dataset_name}")
        seen_names.add(dataset_name)

        print(f"\n正在读取数据集：{dataset_name}")
        result = prepare_pair(
            dataset_name=dataset_name,
            before_path=before_path,
            after_path=after_path,
            before_label_override=args.before_label,
            after_label_override=args.after_label,
        )
        pair_results.append(result)

        info = result["info"]
        print(f"  改进前：{info['before_label']}")
        print(f"  改进后：{info['after_label']}")
        print(f"  改进前批次：{info['before_total_batches']}")
        print(f"  改进后批次：{info['after_total_batches']}")
        print(f"  共同配对批次：{info['paired_batches']}")
        print(f"  多类别配对批次：{info['multi_class_paired_batches']}")

    summary_df, count_df = build_summary(pair_results)

    # 输出 CSV 汇总
    summary_df.to_csv(
        output_root / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    count_df.to_csv(
        output_root / "paired_batch_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 全局图片：只保留多类别批次均值比较和有效配对批次数
    dataset_names = [r["info"]["dataset"] for r in pair_results]

    plot_metric_means(
        summary_df,
        dataset_names,
        "多类别配对批次",
        output_root / "01_multiclass_mean_metrics.png",
        "多类别配对批次：编码改进前后五项指标均值比较",
    )

    plot_batch_counts(
        count_df,
        output_root / "02_paired_batch_counts.png",
    )

    # 每个数据集单独输出逐批次折线图和配对明细 CSV
    details_root = output_root / "paired_batch_details"
    details_root.mkdir(exist_ok=True)

    for result in pair_results:
        info = result["info"]
        dataset_folder = output_root / re.sub(
            r'[<>:"/\\\\|?*]+', "_", info["dataset"]
        )
        dataset_folder.mkdir(exist_ok=True)

        result["data"].to_csv(
            details_root / f"{dataset_folder.name}_all_paired_batches.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["multi_data"].to_csv(
            details_root / f"{dataset_folder.name}_multiclass_paired_batches.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plot_batchwise(
            result,
            dataset_folder,
            "全部配对批次",
            use_multi=False,
        )
        plot_batchwise(
            result,
            dataset_folder,
            "多类别配对批次",
            use_multi=True,
        )

    # 在终端输出主要汇总，便于快速查看
    print("\n" + "=" * 72)
    print("主要结果：多类别配对批次均值")
    print("=" * 72)

    multi_summary = summary_df[
        summary_df["subset"] == "多类别配对批次"
    ]

    if not multi_summary.empty:
        display_cols = [
            "dataset",
            "metric",
            "paired_valid_n",
            "before_mean",
            "after_mean",
            "delta_after_minus_before",
        ]
        print(
            multi_summary[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print("\n" + "=" * 72)
    print("完成！结果保存位置：")
    print(output_root.resolve())
    print("=" * 72)

    print("\n主要输出文件：")
    for name in [
        "01_multiclass_mean_metrics.png",
        "02_paired_batch_counts.png",
        "comparison_summary.csv",
        "paired_batch_counts.csv",
    ]:
        print(f"  {name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n运行失败：{exc}")
        sys.exit(1)
