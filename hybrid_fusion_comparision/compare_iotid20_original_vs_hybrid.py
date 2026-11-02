#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两份 HoSig / Hybrid 编码实验报告（相同数据、相同微批次划分）。

适配报告结构：
{
  "algorithm": "...",
  "total_events": ...,
  "batches": [
    {
      "batch_id": 0,
      "events": 512,
      "evaluation": {
        "n_events": 512,
        "n_communities": ...,
        "modularity": ...,
        "ari": ...,
        "nmi": ...,
        "ch": ...,
        "silhouette": ...,
        "truth_class_count": ...,
        "truth_label_counts": {...}
      }
    }
  ]
}

输出：
01_metric_means.png          各自有效均值 + 配对均值
02_valid_batch_counts.png    各指标有效批次数
03_batch_metric_lines.png    各指标逐批变化折线
04_paired_wins_ties.png      同批次胜负/持平统计
batch_metrics.csv            逐批指标数据
metric_summary.csv/json      汇总结果
README_结果说明.txt           结果说明
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_METRICS = ["modularity", "ari", "nmi", "ch", "silhouette"]

METRIC_NAMES = {
    "modularity": "模块度 Q",
    "ari": "ARI",
    "nmi": "NMI",
    "ch": "Calinski–Harabasz (CH)",
    "silhouette": "Silhouette",
}

COLORS = ["#3B6EAA", "#E38B42"]
EPS = 1e-10


def font_setup():
    """尽量设置中文字体；没有中文字体时使用 Matplotlib 默认字体。"""
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(fp).exists():
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 240


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def first_present(obj, paths, default=None):
    """按候选路径读取第一个存在且非 None 的字段。"""
    for path in paths:
        cur = obj
        ok = True
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return default


def metric_value(metrics, key):
    aliases = {
        "modularity": ["modularity", "Q", "q"],
        "ari": ["ari", "ARI"],
        "nmi": ["nmi", "NMI"],
        "ch": ["ch", "calinski_harabasz", "calinski_harabasz_score", "CH"],
        "silhouette": ["silhouette", "silhouette_score", "Silhouette"],
    }
    for alias in aliases.get(key, [key]):
        value = metrics.get(alias) if isinstance(metrics, dict) else None
        if is_number(value):
            return float(value)
    return np.nan


def infer_class_count(values, truth):
    """优先使用报告中的 truth_class_count，否则从标签计数字典推断。"""
    explicit = first_present(
        values,
        ["truth_class_count", "n_truth_classes"],
        None,
    )
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass

    if not isinstance(truth, dict) or not truth:
        return None

    count = 0
    for value in truth.values():
        try:
            if float(value) > 0:
                count += 1
        except (TypeError, ValueError):
            count += 1
    return count


def read_report(path, metrics):
    """读取一份 report.json，返回 {batch_id: batch_result}。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到报告文件：{path.resolve()}")

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    batches = data.get("batches")
    if not isinstance(batches, list):
        raise ValueError(f"{path} 中没有可识别的 batches 列表")

    result = {}
    for pos, batch in enumerate(batches):
        raw_id = first_present(batch, ["batch_id", "batch", "id"], pos)
        try:
            batch_id = int(raw_id)
        except (TypeError, ValueError):
            batch_id = pos

        # 这两份 IoTID20 报告的指标均位于 batches[].evaluation。
        values = first_present(batch, ["evaluation", "metrics", "results"], {})
        if not isinstance(values, dict):
            values = {}

        raw_events = first_present(
            values,
            ["n_events"],
            first_present(batch, ["events", "n_events", "event_count"], None),
        )
        try:
            events = int(raw_events) if raw_events is not None else None
        except (TypeError, ValueError):
            events = None

        truth = first_present(
            values,
            ["truth_label_counts"],
            first_present(batch, ["truth_label_counts"], {}),
        )
        if not isinstance(truth, dict):
            truth = {}

        row = {
            "events": events,
            "truth": truth,
            "truth_classes": infer_class_count(values, truth),
            "communities": first_present(
                values, ["n_communities", "communities"], None
            ),
        }

        for key in metrics:
            row[key] = metric_value(values, key)

        if batch_id in result:
            raise ValueError(f"{path} 中出现重复 batch_id：{batch_id}")
        result[batch_id] = row

    if not result:
        raise ValueError(f"{path} 中没有批次记录")

    return data, result


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def fmt(value):
    return "NA" if not np.isfinite(value) else f"{value:.6g}"


def validate_batches(report_a, report_b, ids, mode):
    issues = []
    for batch_id in ids:
        a, b = report_a[batch_id], report_b[batch_id]

        if (
            a["events"] is not None
            and b["events"] is not None
            and a["events"] != b["events"]
        ):
            issues.append(
                f"批次 {batch_id}: 事件数不同（A={a['events']}，B={b['events']}）"
            )

        if a["truth"] and b["truth"] and a["truth"] != b["truth"]:
            issues.append(f"批次 {batch_id}: 真实标签分布不同")

    if not issues:
        print("批次一致性检查通过：共同批次的事件数和真实标签分布一致。")
        return

    message = "输入报告的一致性检查发现问题：\n- " + "\n- ".join(issues[:20])
    if len(issues) > 20:
        message += f"\n……另有 {len(issues) - 20} 个问题"

    if mode == "strict":
        raise ValueError(
            message
            + "\n请确认两份报告来自同一份输入数据、相同 batch-size、相同批次划分。"
            + "\n如确认差异是有意的，可使用 --validation warn 检查后继续。"
        )
    elif mode == "warn":
        print("警告：" + message)
    else:
        print("已跳过批次一致性检查。")


def plot_mean_panels(
    output, metrics, names, mean_a, mean_b, pair_a, pair_b,
    pair_n, name_a, name_b, total_batches, scope_label
):
    fig, axes = plt.subplots(
        2, len(metrics),
        figsize=(4.0 * len(metrics), 8.0),
        squeeze=False,
        layout="constrained",
    )

    rows = [
        (
            0, mean_a, mean_b,
            f"(a) 各方法各自有效批次均值（{scope_label}，n={total_batches}）",
        ),
        (
            1, pair_a, pair_b,
            f"(b) 双方共同有效批次的配对均值（{scope_label}）",
        ),
    ]

    for row, vals_a, vals_b, row_title in rows:
        for j, ax in enumerate(axes[row]):
            a, b = vals_a[j], vals_b[j]
            vals = [v for v in (a, b) if np.isfinite(v)]

            bars = ax.bar(
                [0, 1],
                [a if np.isfinite(a) else 0, b if np.isfinite(b) else 0],
                color=COLORS,
                width=0.62,
            )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set_xticks([0, 1], [name_a, name_b], rotation=15, ha="right")
            ax.set_title(names[j], fontweight="bold")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)

            if vals:
                lo, hi = min(vals), max(vals)
                span = hi - lo
                pad = max(span * 0.18, max(abs(lo), abs(hi), 1.0) * 0.08)
                ax.set_ylim(min(0, lo - pad), max(0, hi + pad))

                for bar, value in zip(bars, (a, b)):
                    if np.isfinite(value):
                        y = value + (pad * 0.05 if value >= 0 else -pad * 0.05)
                        va = "bottom" if value >= 0 else "top"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            y,
                            fmt(value),
                            ha="center",
                            va=va,
                            fontsize=8,
                        )
                    else:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            0,
                            "NA",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
            else:
                ax.text(
                    0.5, 0.5, "没有有效值",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

            if row == 1:
                ax.set_xlabel(f"配对批次数 n={pair_n[j]}")

        axes[row, 0].text(
            -0.20, 0.5, row_title,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

    fig.suptitle(
        f"图1  指标均值与配对均值对比（{scope_label}，n={total_batches}）",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output / "01_metric_means.png", bbox_inches="tight")
    plt.close(fig)


def draw(
    report_a, report_b, output, metrics, name_a, name_b,
    validation="strict", paired_lines=True, scope="multiclass"
):
    font_setup()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [METRIC_NAMES.get(k, k) for k in metrics]

    all_common_ids = sorted(set(report_a) & set(report_b))
    if not all_common_ids:
        raise ValueError("两个报告没有共同 batch_id，无法进行逐批对比。")

    # 在筛选之前检查所有共同批次，避免漏掉批次错位问题。
    validate_batches(report_a, report_b, all_common_ids, validation)

    if scope == "multiclass":
        ids = [
            i for i in all_common_ids
            if report_a[i]["truth_classes"] is not None
            and report_b[i]["truth_classes"] is not None
            and report_a[i]["truth_classes"] >= 2
            and report_b[i]["truth_classes"] >= 2
        ]
        scope_label = "多类别批次"
        if not ids:
            raise ValueError(
                "没有筛选到多类别批次。请检查 evaluation.truth_class_count "
                "或 evaluation.truth_label_counts 字段。"
            )
        print("筛选范围：仅多类别批次（真实类别数 >= 2）")
        print(
            f"共同批次总数={len(all_common_ids)}；"
            f"筛选后多类别批次={len(ids)}；"
            f"排除批次={len(all_common_ids) - len(ids)}"
        )
    elif scope == "all":
        ids = all_common_ids
        scope_label = "全部共同批次"
        print("筛选范围：全部共同批次（包含单一真实类别批次）")
    else:
        raise ValueError(f"不支持的 scope：{scope}")

    excluded_a = sorted(set(report_a) - set(report_b))
    excluded_b = sorted(set(report_b) - set(report_a))
    print(
        f"A 报告批次={len(report_a)}；B 报告批次={len(report_b)}；"
        f"共同批次={len(all_common_ids)}；本次分析批次={len(ids)}"
    )
    if excluded_a or excluded_b:
        print(
            f"提示：未参与比较的 A 批次={len(excluded_a)}，"
            f"B 批次={len(excluded_b)}"
        )

    A = np.array(
        [[report_a[i].get(k, np.nan) for k in metrics] for i in ids],
        dtype=float,
    )
    B = np.array(
        [[report_b[i].get(k, np.nan) for k in metrics] for i in ids],
        dtype=float,
    )
    paired = np.isfinite(A) & np.isfinite(B)

    mean_a = np.array([safe_mean(A[:, j]) for j in range(len(metrics))])
    mean_b = np.array([safe_mean(B[:, j]) for j in range(len(metrics))])
    pair_a = np.array([
        safe_mean(A[paired[:, j], j]) for j in range(len(metrics))
    ])
    pair_b = np.array([
        safe_mean(B[paired[:, j], j]) for j in range(len(metrics))
    ])

    valid_a = np.isfinite(A).sum(axis=0)
    valid_b = np.isfinite(B).sum(axis=0)
    pair_n = paired.sum(axis=0)

    # 这里五项指标均按“数值越高越好”统计。
    wins_b = ((B > A + EPS) & paired).sum(axis=0)
    wins_a = ((A > B + EPS) & paired).sum(axis=0)
    ties = pair_n - wins_a - wins_b

    # 图1：总体均值与配对均值。
    plot_mean_panels(
        output, metrics, names, mean_a, mean_b, pair_a, pair_b,
        pair_n, name_a, name_b, len(ids), scope_label
    )

    # 图2：各指标有效批次数。
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(
        figsize=(max(9, 1.6 * len(metrics)), 5.6),
        layout="constrained",
    )
    b1 = ax.bar(x - width / 2, valid_a, width, color=COLORS[0], label=name_a)
    b2 = ax.bar(x + width / 2, valid_b, width, color=COLORS[1], label=name_b)
    ax.bar_label(b1, padding=3, fontsize=8)
    ax.bar_label(b2, padding=3, fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(
        0,
        max(len(ids), int(max(np.max(valid_a), np.max(valid_b)))) * 1.12 + 1,
    )
    ax.set_ylabel("有效批次数")
    ax.set_title(
        f"图2  各指标有效批次数（{scope_label}，n={len(ids)}）",
        fontweight="bold",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(output / "02_valid_batch_counts.png", bbox_inches="tight")
    plt.close(fig)

    # 图3：逐批折线；CH 和 Silhouette 默认只画双方共同有效的批次。
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(14, max(3.2 * len(metrics), 8)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes[:, 0]

    for j, ax in enumerate(axes):
        ya, yb = A[:, j].copy(), B[:, j].copy()

        if paired_lines and metrics[j] in ("ch", "silhouette"):
            ya[~paired[:, j]] = np.nan
            yb[~paired[:, j]] = np.nan

        ax.plot(
            ids, ya, color=COLORS[0], label=name_a,
            marker="o", markersize=2.5, linewidth=1.2,
        )
        ax.plot(
            ids, yb, color=COLORS[1], label=name_b,
            marker="s", markersize=2.5, linewidth=1.2,
        )
        ax.set_ylabel(names[j])

        suffix = (
            f"；配对有效 n={pair_n[j]}"
            if paired_lines and metrics[j] in ("ch", "silhouette")
            else ""
        )
        ax.set_title(
            f"({chr(97 + j)}) {names[j]}{suffix}",
            loc="left",
            fontweight="bold",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("批次编号")
    if len(ids) <= 30:
        axes[-1].set_xticks(ids)
    else:
        step = max(1, math.ceil(len(ids) / 20))
        axes[-1].set_xticks(ids[::step])

    fig.suptitle(
        f"图3  各指标逐批变化（{scope_label}，n={len(ids)}）",
        fontweight="bold",
        fontsize=16,
    )
    fig.savefig(output / "03_batch_metric_lines.png", bbox_inches="tight")
    plt.close(fig)

    # 图4：每项指标的同批次胜负/持平次数。
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(
        figsize=(12, max(5, 0.9 * len(metrics) + 2)),
        layout="constrained",
    )
    ax.barh(y, wins_b, color="#36936C", label=f"{name_b} 数值更高")
    ax.barh(
        y, wins_a, left=wins_b,
        color="#D37D6D", label=f"{name_a} 数值更高",
    )
    ax.barh(
        y, ties, left=wins_b + wins_a,
        color="#A6AAB2", label="持平",
    )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, int(np.max(pair_n))) * 1.16 + 1)
    ax.set_xlabel("配对批次数")
    ax.set_title(
        f"图4  有效配对批次中的数值高低统计（{scope_label}）",
        fontweight="bold",
        fontsize=14,
    )

    for j in range(len(metrics)):
        segments = [
            (wins_b[j], wins_b[j] / 2),
            (wins_a[j], wins_b[j] + wins_a[j] / 2),
            (ties[j], wins_b[j] + wins_a[j] + ties[j] / 2),
        ]
        for count, xpos in segments:
            if count:
                ax.text(
                    xpos, j, str(int(count)),
                    ha="center", va="center",
                    color="white" if count > 2 else "#202020",
                    fontsize=9, fontweight="bold",
                )
        ax.text(pair_n[j] + 0.3, j, f"n={int(pair_n[j])}", va="center", fontsize=9)

    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(output / "04_paired_wins_ties.png", bbox_inches="tight")
    plt.close(fig)

    # 导出逐批数据，便于后续检查。
    batch_csv = output / "batch_metrics.csv"
    with batch_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = [
            "batch_id",
            "events_a",
            "events_b",
            "truth_class_count",
            "truth_counts_match",
        ]
        for metric in metrics:
            header.extend([
                f"{metric}_a",
                f"{metric}_b",
                f"{metric}_paired_valid",
            ])
        writer.writerow(header)

        for pos, batch_id in enumerate(ids):
            row = [
                batch_id,
                report_a[batch_id]["events"],
                report_b[batch_id]["events"],
                report_a[batch_id]["truth_classes"],
                report_a[batch_id]["truth"] == report_b[batch_id]["truth"],
            ]
            for j, metric in enumerate(metrics):
                row.extend([
                    A[pos, j],
                    B[pos, j],
                    bool(paired[pos, j]),
                ])
            writer.writerow(row)

    # 导出指标汇总。
    summary_rows = []
    for j, metric in enumerate(metrics):
        summary_rows.append({
            "metric": metric,
            "metric_name": names[j],
            f"{name_a}_mean_all_valid": (
                mean_a[j] if np.isfinite(mean_a[j]) else None
            ),
            f"{name_b}_mean_all_valid": (
                mean_b[j] if np.isfinite(mean_b[j]) else None
            ),
            f"{name_a}_valid_batches": int(valid_a[j]),
            f"{name_b}_valid_batches": int(valid_b[j]),
            f"{name_a}_paired_mean": (
                pair_a[j] if np.isfinite(pair_a[j]) else None
            ),
            f"{name_b}_paired_mean": (
                pair_b[j] if np.isfinite(pair_b[j]) else None
            ),
            "paired_batches": int(pair_n[j]),
            f"{name_b}_higher_count": int(wins_b[j]),
            f"{name_a}_higher_count": int(wins_a[j]),
            "ties": int(ties[j]),
        })

    with (output / "metric_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "report_a_batches": len(report_a),
        "report_b_batches": len(report_b),
        "common_batches_before_filter": len(all_common_ids),
        "analyzed_batches": len(ids),
        "common_batch_ids": ids,
        "scope": scope,
        "validation_mode": validation,
        "name_a": name_a,
        "name_b": name_b,
        "metrics": summary_rows,
        "notes": [
            "各指标均值只使用该指标的有限数值。",
            "配对均值只使用同一批次上两种方法均有效的值。",
            "胜负统计只比较同批次双方均有效的值；数值更高不代表统计显著。",
            "图1每个指标使用独立坐标轴，避免不同量纲互相压缩。",
            "CH 和 Silhouette 的逐批折线默认仅绘制双方共同有效的批次。",
            "ARI/NMI 反映与真实标签的一致性；标签是否符合研究目标需单独论证。",
        ],
    }
    (output / "metric_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = (
        f"分析范围：{scope_label}\n"
        f"A 方法：{name_a}\n"
        f"B 方法：{name_b}\n"
        f"本次分析批次数：{len(ids)}\n"
        f"共同批次数（筛选前）：{len(all_common_ids)}\n"
        f"A 报告批次数：{len(report_a)}\n"
        f"B 报告批次数：{len(report_b)}\n\n"
        "01_metric_means.png：各自有效均值与配对均值；每项指标单独坐标轴。\n"
        "02_valid_batch_counts.png：各指标有效批次数。\n"
        "03_batch_metric_lines.png：逐批指标变化；CH、Silhouette 默认只画共同有效值。\n"
        "04_paired_wins_ties.png：同批配对数值高低统计，不代表统计显著性。\n"
        "batch_metrics.csv：逐批原始指标及有效性标记。\n"
        "metric_summary.csv/json：均值、有效批次数、配对均值和胜负统计。\n\n"
        "multiclass：只分析真实类别数至少为2的共同批次。\n"
        "all：分析全部共同批次，包括单一真实类别批次。\n"
        "五项指标的胜负图按数值越高越好统计；胜负次数不代表统计显著性。\n"
    )
    (output / "README_结果说明.txt").write_text(readme, encoding="utf-8")

    print(f"\n已生成 4 张对比图和统计文件：{output.resolve()}")
    print(f"共同批次数：{len(ids)}")
    print("各指标配对有效批次数：", dict(zip(metrics, map(int, pair_n))))
    print("\n配对均值：")
    for j, metric in enumerate(metrics):
        print(
            f"  {metric:12s}  "
            f"{name_a}={fmt(pair_a[j]):>12s}  "
            f"{name_b}={fmt(pair_b[j]):>12s}  "
            f"n={int(pair_n[j])}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="比较两份 HoSig / Hybrid 编码实验报告并绘制四类对比图。"
    )
    parser.add_argument(
        "--report-a",
        default="fusion_report(3).json",
        help="实验A报告 JSON（默认：fusion_report(3).json）",
    )
    parser.add_argument(
        "--report-b",
        default="iotid20_hybrid_full_equal_333_report.json",
        help="实验B报告 JSON（默认：iotid20_hybrid_full_equal_333_report.json）",
    )
    parser.add_argument(
        "--output",
        default="plot_comparison/iotid20_original_vs_hybrid_equal333_multiclass",
        help="图表和统计文件输出目录",
    )
    parser.add_argument(
        "--name-a",
        default="原始 HoSig",
        help="图例中的实验A名称",
    )
    parser.add_argument(
        "--name-b",
        default="Hybrid 均衡融合",
        help="图例中的实验B名称",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="指标列表，逗号分隔",
    )
    parser.add_argument(
        "--scope",
        choices=["multiclass", "all"],
        default="multiclass",
        help="multiclass=仅多类别批次（默认）；all=全部共同批次",
    )
    parser.add_argument(
        "--validation",
        choices=["strict", "warn", "off"],
        default="strict",
        help="批次一致性检查：strict=不一致时报错；warn=警告继续；off=跳过",
    )
    parser.add_argument(
        "--all-valid-lines",
        action="store_true",
        help="图3的CH/Silhouette也绘制双方各自所有有效值；默认只画配对有效值",
    )

    args = parser.parse_args()

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    unknown = [m for m in metrics if m not in METRIC_NAMES]
    if not metrics:
        parser.error("--metrics 至少需要一个指标")
    if unknown:
        parser.error(
            f"不支持的指标：{unknown}；支持：{sorted(METRIC_NAMES)}"
        )

    data_a, report_a = read_report(args.report_a, metrics)
    data_b, report_b = read_report(args.report_b, metrics)

    print("实验A报告：", Path(args.report_a).resolve())
    print("实验B报告：", Path(args.report_b).resolve())
    print("A 报告算法：", data_a.get("algorithm", "未记录"))
    print("B 报告算法：", data_b.get("algorithm", "未记录"))
    print("A 总事件数：", data_a.get("total_events", "未记录"))
    print("B 总事件数：", data_b.get("total_events", "未记录"))

    draw(
        report_a,
        report_b,
        args.output,
        metrics,
        args.name_a,
        args.name_b,
        validation=args.validation,
        paired_lines=not args.all_valid_lines,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
