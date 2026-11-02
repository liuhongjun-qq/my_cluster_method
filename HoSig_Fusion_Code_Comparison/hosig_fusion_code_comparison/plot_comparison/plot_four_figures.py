#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵活版 HoSig / 融合方法对比绘图脚本。

支持：
- 任意共同批次数（不再写死 46 批）
- 自定义报告路径、方法名称、输出目录和指标集合
- 每项指标使用独立坐标轴，避免 CH 等大数值压扁 ARI/NMI
- 正确处理负数、NaN、无有效值的指标
- 总体均值、配对均值、有效批次数、逐批折线、胜负统计
- 默认检查同批事件数和标签分布；可选择 warn 或 off
- 输出逐批 CSV、指标汇总 CSV/JSON、统计说明

报告结构默认兼容：
HoSig: batches[].metrics + batches[].diagnostics.n_events
Fusion: batches[].evaluation + batches[].events
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
    "iir_k": "IIR-k",
    "iir_norm": "IIR-normalized",
}
COLORS = ["#3B6EAA", "#E38B42"]
EPS = 1e-10


def font_setup():
    # Windows 通常可用 Microsoft YaHei；Linux 环境则尝试 Noto。
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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first_present(obj, paths, default=None):
    """按候选路径读取第一个存在的字段。"""
    for path in paths:
        cur = obj
        ok = True
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return default


def metric_value(metrics, key):
    aliases = {
        "modularity": ["modularity", "Q", "q"],
        "ari": ["ari", "ARI"],
        "nmi": ["nmi", "NMI"],
        "ch": ["ch", "calinski_harabasz", "calinski_harabasz_score", "CH"],
        "silhouette": ["silhouette", "silhouette_score", "Silhouette"],
        "iir_k": ["iir_k", "IIR_k"],
        "iir_norm": ["iir_norm", "IIR_norm"],
    }
    for alias in aliases.get(key, [key]):
        value = metrics.get(alias) if isinstance(metrics, dict) else None
        if is_number(value):
            return float(value)
    return np.nan


def read_report(path, alg, metrics):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到报告文件：{path}")
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

        if alg == "hosig":
            values = first_present(batch, ["metrics", "evaluation", "results"], {})
            events = first_present(batch, ["diagnostics.n_events", "n_events", "events", "event_count"])
        else:
            values = first_present(batch, ["evaluation", "metrics", "results"], {})
            events = first_present(batch, ["events", "n_events", "diagnostics.n_events", "event_count"])

        if not isinstance(values, dict):
            values = {}
        try:
            events = int(events) if events is not None else None
        except (TypeError, ValueError):
            events = None

        truth = first_present(values, ["truth_label_counts"], None)
        if truth is None:
            truth = first_present(batch, ["truth_label_counts", "diagnostics.truth_label_counts"], {})
        if not isinstance(truth, dict):
            truth = {}

        row = {"events": events, "truth": truth}
        for key in metrics:
            row[key] = metric_value(values, key)
        result[batch_id] = row

    if not result:
        raise ValueError(f"{path} 中没有批次记录")
    return result


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def fmt(value):
    return "NA" if not np.isfinite(value) else f"{value:.6g}"


def validate_batches(hosig, fusion, ids, mode):
    issues = []
    for batch_id in ids:
        h, f = hosig[batch_id], fusion[batch_id]
        if h["events"] is not None and f["events"] is not None and h["events"] != f["events"]:
            issues.append(f"批次 {batch_id}: 事件数不同（HoSig={h['events']}，融合={f['events']}）")
        if h["truth"] and f["truth"] and h["truth"] != f["truth"]:
            issues.append(f"批次 {batch_id}: 标签分布不同")
    if issues:
        message = "输入报告的一致性检查发现问题：\n- " + "\n- ".join(issues[:20])
        if len(issues) > 20:
            message += f"\n……另有 {len(issues) - 20} 个问题"
        if mode == "strict":
            raise ValueError(message + "\n请确认两个报告来自同一份输入数据、相同批次划分。")
        elif mode == "warn":
            print("警告：" + message)
        else:
            print("已跳过批次一致性检查。")


def plot_mean_panels(output, metrics, names, mean_a, mean_b, pair_a, pair_b, pair_n, name_a, name_b, total_batches):
    # 每个指标有独立坐标轴；负数和不同量级都能正常显示。
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.0 * len(metrics), 8.0), squeeze=False, layout="constrained")
    for row, vals_a, vals_b, row_title in [
        (0, mean_a, mean_b, f"(a) 各方法各自有效批次均值（共同批次 {total_batches}）"),
        (1, pair_a, pair_b, "(b) 双方共同有效批次的配对均值"),
    ]:
        for j, ax in enumerate(axes[row]):
            a, b = vals_a[j], vals_b[j]
            vals = [v for v in (a, b) if np.isfinite(v)]
            bars = ax.bar([0, 1], [a if np.isfinite(a) else 0, b if np.isfinite(b) else 0],
                          color=COLORS, width=0.62)
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
                        ax.text(bar.get_x() + bar.get_width()/2, y, fmt(value), ha="center", va=va, fontsize=8)
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2, 0, "NA", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(0.5, 0.5, "没有有效值", transform=ax.transAxes, ha="center", va="center")
            if row == 1:
                ax.set_xlabel(f"配对批次数 n={pair_n[j]}")
        axes[row, 0].text(-0.20, 0.5, row_title, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle(f"图1  指标均值与配对均值对比（{total_batches} 个共同批次）", fontsize=16, fontweight="bold")
    fig.savefig(output / "01_metric_means.png", bbox_inches="tight")
    plt.close(fig)


def draw(hosig, fusion, output, metrics, name_a, name_b, validation="strict", paired_lines=True):
    font_setup()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [METRIC_NAMES.get(k, k) for k in metrics]

    ids = sorted(set(hosig) & set(fusion))
    if not ids:
        raise ValueError("两个报告没有共同 batch_id，无法进行逐批对比。")
    validate_batches(hosig, fusion, ids, validation)
    excluded_h = sorted(set(hosig) - set(fusion))
    excluded_f = sorted(set(fusion) - set(hosig))
    print(f"HoSig 批次={len(hosig)}；融合批次={len(fusion)}；共同批次={len(ids)}")
    if excluded_h or excluded_f:
        print(f"提示：未参与配对比较的 HoSig 批次={len(excluded_h)}，融合批次={len(excluded_f)}")

    A = np.array([[hosig[i].get(k, np.nan) for k in metrics] for i in ids], dtype=float)
    B = np.array([[fusion[i].get(k, np.nan) for k in metrics] for i in ids], dtype=float)
    paired = np.isfinite(A) & np.isfinite(B)

    mean_a = np.array([safe_mean(A[:, j]) for j in range(len(metrics))])
    mean_b = np.array([safe_mean(B[:, j]) for j in range(len(metrics))])
    pair_a = np.array([safe_mean(A[paired[:, j], j]) for j in range(len(metrics))])
    pair_b = np.array([safe_mean(B[paired[:, j], j]) for j in range(len(metrics))])
    valid_a = np.isfinite(A).sum(axis=0)
    valid_b = np.isfinite(B).sum(axis=0)
    pair_n = paired.sum(axis=0)
    wins_b = ((B > A + EPS) & paired).sum(axis=0)
    wins_a = ((A > B + EPS) & paired).sum(axis=0)
    ties = pair_n - wins_a - wins_b

    # 图1：每项指标单独坐标轴，避免指标量纲互相压缩。
    plot_mean_panels(output, metrics, names, mean_a, mean_b, pair_a, pair_b, pair_n,
                     name_a, name_b, len(ids))

    # 图2：有效批次数
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(metrics)), 5.6), layout="constrained")
    b1 = ax.bar(x - width/2, valid_a, width, color=COLORS[0], label=name_a)
    b2 = ax.bar(x + width/2, valid_b, width, color=COLORS[1], label=name_b)
    ax.bar_label(b1, padding=3, fontsize=8)
    ax.bar_label(b2, padding=3, fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(len(ids), int(max(np.max(valid_a), np.max(valid_b)))) * 1.12 + 1)
    ax.set_ylabel("有效批次数")
    ax.set_title(f"图2  各指标有效批次数（共同批次 {len(ids)}）", fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(output / "02_valid_batch_counts.png", bbox_inches="tight")
    plt.close(fig)

    # 图3：逐批折线；CH 与 Silhouette 默认只画双方共同有效的批次。
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, max(3.2 * len(metrics), 8)),
                             sharex=True, squeeze=False, layout="constrained")
    axes = axes[:, 0]
    for j, ax in enumerate(axes):
        ya, yb = A[:, j].copy(), B[:, j].copy()
        if paired_lines and metrics[j] in ("ch", "silhouette"):
            ya[~paired[:, j]] = np.nan
            yb[~paired[:, j]] = np.nan
        ax.plot(ids, ya, color=COLORS[0], label=name_a, marker="o", markersize=2.5, linewidth=1.2)
        ax.plot(ids, yb, color=COLORS[1], label=name_b, marker="s", markersize=2.5, linewidth=1.2)
        ax.set_ylabel(names[j])
        suffix = f"；配对有效 n={pair_n[j]}" if paired_lines and metrics[j] in ("ch", "silhouette") else ""
        ax.set_title(f"({chr(97+j)}) {names[j]}{suffix}", loc="left", fontweight="bold", fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        if metrics[j] == "ch":
            # 原始线性坐标保留真实量值；提示极端值可能压缩其他波动，不做隐式截断。
            ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4), useMathText=True)
    axes[-1].set_xlabel("批次编号")
    if len(ids) <= 30:
        axes[-1].set_xticks(ids)
    else:
        step = max(1, math.ceil(len(ids) / 20))
        axes[-1].set_xticks(ids[::step])
    fig.suptitle(f"图3  各指标逐批变化（{len(ids)} 个共同批次）", fontweight="bold", fontsize=16)
    fig.savefig(output / "03_batch_metric_lines.png", bbox_inches="tight")
    plt.close(fig)

    # 图4：每项指标高值胜出次数。此图只表示数值高低，不代表统计显著性。
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.9 * len(metrics) + 2)), layout="constrained")
    ax.barh(y, wins_b, color="#36936C", label=f"{name_b} 数值更高")
    ax.barh(y, wins_a, left=wins_b, color="#D37D6D", label=f"{name_a} 数值更高")
    ax.barh(y, ties, left=wins_b + wins_a, color="#A6AAB2", label="持平")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, int(np.max(pair_n))) * 1.16 + 1)
    ax.set_xlabel("配对批次数")
    ax.set_title("图4  有效配对批次中的数值高低统计", fontweight="bold", fontsize=14)
    for j in range(len(metrics)):
        segments = [(wins_b[j], wins_b[j]/2),
                    (wins_a[j], wins_b[j] + wins_a[j]/2),
                    (ties[j], wins_b[j] + wins_a[j] + ties[j]/2)]
        for count, xpos in segments:
            if count:
                ax.text(xpos, j, str(int(count)), ha="center", va="center",
                        color="white" if count > 2 else "#202020", fontsize=9, fontweight="bold")
        ax.text(pair_n[j] + 0.3, j, f"n={int(pair_n[j])}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(output / "04_paired_wins_ties.png", bbox_inches="tight")
    plt.close(fig)

    # 导出可审计的逐批数据。
    with (output / "batch_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["batch_id", "events_hosig", "events_fusion", "truth_counts_match"]
        for metric in metrics:
            header.extend([f"{metric}_hosig", f"{metric}_fusion", f"{metric}_paired_valid"])
        writer.writerow(header)
        for pos, batch_id in enumerate(ids):
            row = [batch_id, hosig[batch_id]["events"], fusion[batch_id]["events"],
                   hosig[batch_id]["truth"] == fusion[batch_id]["truth"]]
            for j, metric in enumerate(metrics):
                row.extend([A[pos, j], B[pos, j], bool(paired[pos, j])])
            writer.writerow(row)

    # 汇总表：分别报告各自有效均值、配对均值、有效数、配对胜负。
    summary_rows = []
    for j, metric in enumerate(metrics):
        summary_rows.append({
            "metric": metric,
            "metric_name": names[j],
            f"{name_a}_mean_all_valid": mean_a[j] if np.isfinite(mean_a[j]) else None,
            f"{name_b}_mean_all_valid": mean_b[j] if np.isfinite(mean_b[j]) else None,
            f"{name_a}_valid_batches": int(valid_a[j]),
            f"{name_b}_valid_batches": int(valid_b[j]),
            f"{name_a}_paired_mean": pair_a[j] if np.isfinite(pair_a[j]) else None,
            f"{name_b}_paired_mean": pair_b[j] if np.isfinite(pair_b[j]) else None,
            "paired_batches": int(pair_n[j]),
            f"{name_b}_higher_count": int(wins_b[j]),
            f"{name_a}_higher_count": int(wins_a[j]),
            "ties": int(ties[j]),
        })
    with (output / "metric_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = {
        "hosig_report_batches": len(hosig),
        "fusion_report_batches": len(fusion),
        "common_batches": len(ids),
        "common_batch_ids": ids,
        "validation_mode": validation,
        "metrics": summary_rows,
        "notes": [
            "各指标均值只使用该指标的有限数值。",
            "配对均值只使用同一批次上两种方法均有效的值。",
            "胜负统计只比较同批次双方均有效的值；数值更高不代表统计显著。",
            "图1每个指标使用独立坐标轴，避免不同量纲互相压缩。",
            "CH 和 Silhouette 的逐批折线默认仅绘制双方共同有效的批次。",
        ],
    }
    (output / "metric_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README_结果说明.txt").write_text(
        f"共同批次数：{len(ids)}\n"
        f"HoSig 报告批次数：{len(hosig)}\n"
        f"融合报告批次数：{len(fusion)}\n\n"
        "01_metric_means.png：各自有效均值与配对均值；每项指标单独坐标轴。\n"
        "02_valid_batch_counts.png：各指标有效批次数。\n"
        "03_batch_metric_lines.png：逐批指标变化；CH、Silhouette 默认只画共同有效值。\n"
        "04_paired_wins_ties.png：同批配对数值高低统计，不代表统计显著性。\n"
        "batch_metrics.csv：逐批原始指标及有效性标记。\n"
        "metric_summary.csv/json：均值、有效批次数、配对均值和胜负统计。\n\n"
        "注意：ARI/NMI 等类别指标不能单独证明发现了同一次攻击活动的跨阶段关联；需确认参考标签是否匹配研究目标。\n",
        encoding="utf-8",
    )
    print(f"已生成 4 张对比图和统计文件：{output.resolve()}")
    print(f"共同批次数：{len(ids)}")
    print("各指标配对有效批次数：", dict(zip(metrics, map(int, pair_n))))


def main():
    parser = argparse.ArgumentParser(
        description="灵活比较 HoSig 与融合方法的批次级评价结果并绘制四类图表。"
    )
    parser.add_argument("--hosig", default="data/hosig_report.json", help="HoSig 原方法报告 JSON")
    parser.add_argument("--fusion", default="data/fusion_report.json", help="融合方法报告 JSON")
    parser.add_argument("--output", default="figures", help="图表和统计文件输出目录")
    parser.add_argument("--hosig-name", default="原始 HoSig", help="图例中的 HoSig 方法名称")
    parser.add_argument("--fusion-name", default="HoSig + Louvain", help="图例中的融合方法名称")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="要比较的指标，逗号分隔，例如 modularity,ari,nmi,ch,silhouette")
    parser.add_argument("--validation", choices=["strict", "warn", "off"], default="strict",
                        help="批次一致性检查：strict=不一致时报错；warn=警告继续；off=跳过")
    parser.add_argument("--all-valid-lines", action="store_true",
                        help="图3的 CH/Silhouette 也绘制各自所有有效值；默认只画双方配对有效值")
    args = parser.parse_args()

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    supported = set(METRIC_NAMES)
    unknown = [m for m in metrics if m not in supported]
    if not metrics:
        parser.error("--metrics 至少需要一个指标")
    if unknown:
        parser.error(f"不支持的指标：{unknown}；支持：{sorted(supported)}")

    hosig = read_report(args.hosig, "hosig", metrics)
    fusion = read_report(args.fusion, "fusion", metrics)
    draw(hosig, fusion, args.output, metrics, args.hosig_name, args.fusion_name,
         validation=args.validation, paired_lines=not args.all_valid_lines)


if __name__ == "__main__":
    main()
