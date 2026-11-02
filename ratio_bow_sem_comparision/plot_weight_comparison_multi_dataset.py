
# -*- coding: utf-8 -*-
"""
通用编码权重对比绘图工具

支持：
1. IoT-23、IoTID20 等不同数据集
2. 任意数量的编码权重实验组
3. Modularity、CH、Silhouette 三个指标
4. 自动分配橙色、蓝色、绿色、红色等颜色
5. 仅使用所有实验组共同有效的批次计算均值

输出：
figure1_three_metrics_mean.png
mean_metrics_summary.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


METRICS = [
    ("modularity", "Modularity", ".4f"),
    ("ch", "Calinski-Harabasz (CH)", ".2f"),
    ("silhouette", "Silhouette", ".4f"),
]

# 不同实验组使用不同颜色
COLORS = [
    "#F28E2B",  # 橙色
    "#4E79A7",  # 蓝色
    "#59A14F",  # 绿色
    "#E15759",  # 红色
    "#B07AA1",  # 紫色
    "#76B7B2",  # 青色
    "#EDC948",  # 黄色
    "#FF9DA7",  # 粉色
    "#9C755F",  # 棕色
    "#BAB0AC",  # 灰色
]


def is_valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_report(report_path):

    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到报告文件：{path}"
        )

    with path.open("r", encoding="utf-8-sig") as f:
        report = json.load(f)

    batches = report.get("batches")

    if not isinstance(batches, list):
        raise ValueError(
            f"{path.name} 中没有有效的 batches 列表"
        )

    batch_results = {}

    for index, batch in enumerate(batches):

        batch_id = batch.get("batch_id", index)

        evaluation = batch.get("evaluation") or {}

        batch_results[batch_id] = {
            "modularity": evaluation.get(
                "modularity",
                batch.get("modularity")
            ),
            "ch": evaluation.get(
                "ch",
                batch.get("ch")
            ),
            "silhouette": evaluation.get(
                "silhouette",
                batch.get("silhouette")
            ),
        }

    # 自动读取编码权重，生成图例名称
    encoder = report.get("encoder") or {}

    weights = encoder.get("branch_weights") or {}

    bow = weights.get("bow")
    semantic = weights.get("semantic")

    if bow is not None and semantic is not None:

        label = (
            f"BOW {float(bow):g} / "
            f"Semantic {float(semantic):g}"
        )

    else:
        label = path.stem

    return {
        "path": str(path),
        "label": label,
        "batches": batch_results,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Multi-dataset encoding weight comparison"
    )

    # 输入任意数量的实验报告
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="输入两个或更多 JSON 实验报告路径"
    )

    # 可以手动设置图例名称
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="手动指定各实验组名称"
    )

    # 数据集名称
    parser.add_argument(
        "--dataset",
        default="Dataset",
        help="数据集名称，例如 IoT-23 或 IoTID20"
    )

    # 输出目录
    parser.add_argument(
        "--output-dir",
        default="plot_comparison/weight_comparison",
        help="结果保存目录"
    )

    args = parser.parse_args()

    if len(args.reports) < 2:
        parser.error("至少需要两份实验报告")

    reports = [
        load_report(path)
        for path in args.reports
    ]

    if args.labels is not None:

        if len(args.labels) != len(reports):
            parser.error(
                "--labels 数量必须与 --reports 数量一致"
            )

        for report, label in zip(reports, args.labels):
            report["label"] = label

    labels = [
        report["label"]
        for report in reports
    ]

    # 字体设置
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans"
    ]

    plt.rcParams["axes.unicode_minus"] = False

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_rows = []

    metric_summaries = {}

    # 分别处理三个指标
    for metric_key, metric_title, value_format in METRICS:

        # 找出所有实验组共同拥有的批次
        common_ids = set(
            reports[0]["batches"].keys()
        )

        for report in reports[1:]:

            common_ids &= set(
                report["batches"].keys()
            )

        valid_ids = []

        # 只保留所有实验组该指标均有效的批次
        for batch_id in common_ids:

            values = [
                report["batches"][batch_id].get(
                    metric_key
                )
                for report in reports
            ]

            if all(
                is_valid_number(value)
                for value in values
            ):
                valid_ids.append(batch_id)

        if not valid_ids:

            raise ValueError(
                f"{metric_title} 没有共同有效批次"
            )

        means = []
        medians = []

        for report in reports:

            values = [
                float(
                    report["batches"][batch_id][metric_key]
                )
                for batch_id in valid_ids
            ]

            metric_mean = mean(values)

            metric_median = median(values)

            means.append(metric_mean)

            medians.append(metric_median)

            summary_rows.append({

                "metric": metric_key,

                "metric_name": metric_title,

                "setting": report["label"],

                "report_path": report["path"],

                "common_valid_batches": len(valid_ids),

                "mean": metric_mean,

                "median": metric_median

            })

        metric_summaries[metric_key] = {

            "title": metric_title,

            "format": value_format,

            "means": means,

            "medians": medians,

            "valid_count": len(valid_ids)

        }

    # ==================================
    # 第一张图：三个指标平均值比较
    # ==================================

    number_of_reports = len(reports)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6)
    )

    fig.suptitle(
        f"{args.dataset}: Mean Metric Comparison Across Encoding Weights",
        fontsize=16,
        fontweight="bold"
    )

    x_positions = list(
        range(number_of_reports)
    )

    colors = [
        COLORS[i % len(COLORS)]
        for i in range(number_of_reports)
    ]

    for ax, metric in zip(axes, METRICS):

        metric_key, metric_title, value_format = metric

        info = metric_summaries[metric_key]

        values = info["means"]

        bars = ax.bar(
            x_positions,
            values,
            width=0.68 if number_of_reports <= 4 else 0.82,
            color=colors,
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_title(
            metric_title,
            fontsize=13,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Mean across common valid batches"
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels(
            labels,
            rotation=25,
            ha="right",
            fontsize=9
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35
        )

        ax.set_axisbelow(True)

        low = min(0.0, min(values))

        high = max(values)

        span = high - low

        if span == 0:
            span = max(abs(high), 1.0)

        ax.set_ylim(
            low - span * 0.12,
            high + span * 0.24
        )

        offset = span * 0.025

        # 在柱状图上标注具体数值
        for bar, value in zip(bars, values):

            y = value + (
                offset if value >= 0 else -offset
            )

            va = (
                "bottom" if value >= 0 else "top"
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                format(value, value_format),
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold"
            )

        ax.text(
            0.5,
            -0.38,
            f"Common valid batches: {info['valid_count']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9
        )

    fig.text(
        0.5,
        0.015,
        "Each metric uses only batch IDs valid in every compared report.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(
        rect=[0, 0.11, 1, 0.91]
    )

    figure_path = (
        output_dir /
        "figure1_three_metrics_mean.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ==================================
    # 保存 CSV 汇总表
    # ==================================

    csv_path = (
        output_dir /
        "mean_metrics_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(summary_rows)

    # ==================================
    # 输出终端结果
    # ==================================

    print("\n========== Mean Metric Comparison ==========")

    for report in reports:

        print(
            f"Setting: {report['label']}"
        )

    print("---------------------------------------------")

    for metric_key, metric_title, value_format in METRICS:

        info = metric_summaries[metric_key]

        print(
            f"\n{metric_title}"
            f" (valid batches: {info['valid_count']})"
        )

        for label, value in zip(
            labels,
            info["means"]
        ):

            print(
                f"  {label}: {value:{value_format}}"
            )

    print("---------------------------------------------")

    print(
        f"Figure saved: {figure_path.resolve()}"
    )

    print(
        f"CSV saved: {csv_path.resolve()}"
    )


if __name__ == "__main__":

    main()
