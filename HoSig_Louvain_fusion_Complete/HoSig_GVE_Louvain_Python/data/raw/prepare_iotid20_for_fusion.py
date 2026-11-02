
import argparse
import re
from pathlib import Path

import pandas as pd


def normalize(name):
    """统一列名格式，方便匹配不同版本的字段。"""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def find_column(columns, candidates, required=True):
    """查找数据集中的指定字段。"""
    mapping = {normalize(c): c for c in columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in mapping:
            return mapping[key]

    if required:
        raise ValueError(
            f"找不到必需字段：{candidates}\n"
            f"实际字段：{list(columns)}"
        )

    return None


def main():
    parser = argparse.ArgumentParser(
        description="将 IoTID20 转换为 HoSig + Louvain 输入格式"
    )

    parser.add_argument("--input", required=True, help="原始 CSV 路径")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{input_path}\n"
            "请检查文件名和路径。"
        )

    print("正在读取原始数据……")
    df = pd.read_csv(
        input_path,
        low_memory=False,
        encoding="utf-8-sig"
    )

    df.columns = [str(c).strip() for c in df.columns]

    print(f"原始记录数：{len(df):,}")
    print(f"原始字段数：{len(df.columns)}")

    # 1. 查找必需字段
    timestamp_col = find_column(
        df.columns, ["Timestamp", "timestamp", "ts"]
    )
    src_col = find_column(
        df.columns, ["Src_IP", "src_ip", "orig_h"]
    )
    dst_col = find_column(
        df.columns, ["Dst_IP", "dst_ip", "resp_h"]
    )
    label_col = find_column(
        df.columns, ["Sub_Cat", "sub_cat"]
    )

    # 2. 查找可选网络特征
    optional_fields = {
        "proto": ["Protocol", "proto"],
        "src_port": ["Src_Port", "src_port", "orig_p"],
        "dst_port": ["Dst_Port", "dst_port", "resp_p"],
        "duration": ["Flow_Duration", "duration"],
        "orig_bytes": ["TotLen_Fwd_Pkts", "orig_bytes"],
        "resp_bytes": ["TotLen_Bwd_Pkts", "resp_bytes"],
        "orig_pkts": ["Tot_Fwd_Pkts", "orig_pkts"],
        "resp_pkts": ["Tot_Bwd_Pkts", "resp_pkts"],
    }

    matched = {
        target: find_column(df.columns, candidates, required=False)
        for target, candidates in optional_fields.items()
    }

    # 3. 解析时间戳
    raw_ts = df[timestamp_col].astype("string").str.strip()

    timestamps = pd.to_datetime(
        raw_ts,
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # 如果部分记录格式不同，再尝试自动解析
    failed = timestamps.isna()

    if failed.any():
        timestamps.loc[failed] = pd.to_datetime(
            raw_ts.loc[failed],
            dayfirst=True,
            errors="coerce"
        )

    # 4. 清理缺失的时间、IP 和标签
    src_ip = df[src_col].astype("string").str.strip()
    dst_ip = df[dst_col].astype("string").str.strip()
    labels = df[label_col].astype("string").str.strip()

    valid = (
        timestamps.notna()
        & src_ip.notna()
        & dst_ip.notna()
        & src_ip.ne("")
        & dst_ip.ne("")
        & labels.notna()
        & labels.ne("")
    )

    dropped = int((~valid).sum())

    data = df.loc[valid].copy()
    timestamps = timestamps.loc[valid]
    src_ip = src_ip.loc[valid]
    dst_ip = dst_ip.loc[valid]
    labels = labels.loc[valid]

    # 5. 生成统一输入字段
    result = pd.DataFrame({
        "timestamp": timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "src_ip": src_ip.to_numpy(),
        "dst_ip": dst_ip.to_numpy(),
        "label": labels.to_numpy(),
    })

    # 保留可用的网络特征
    for target, source in matched.items():
        if source is not None:
            result[target] = data[source].to_numpy()

    # 其他数值型网络特征也可以保留。
    # 不把原始标签、IP、时间戳或流量 ID 作为数值特征。
    excluded = {
        normalize(timestamp_col),
        normalize(src_col),
        normalize(dst_col),
        normalize(label_col),
        normalize("Label"),
        normalize("Cat"),
        normalize("Sub_Cat"),
        normalize("Flow_ID"),
        normalize("uid"),
        normalize("event_id"),
    }

    excluded.update(
        normalize(source)
        for source in matched.values()
        if source is not None
    )

    existing_names = {
        normalize(column) for column in result.columns
    }

    for column in data.columns:
        if normalize(column) in excluded:
            continue

        numeric = pd.to_numeric(data[column], errors="coerce")

        if len(numeric) == 0:
            continue

        if numeric.notna().mean() < 0.5:
            continue

        name = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            str(column).strip()
        ).strip("_").lower()

        if name and normalize(name) not in existing_names:
            result[name] = numeric.to_numpy()
            existing_names.add(normalize(name))

    # 6. 按时间升序排列
    result["_sort_time"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce"
    )

    result = (
        result.sort_values("_sort_time", kind="mergesort")
        .drop(columns="_sort_time")
        .reset_index(drop=True)
    )

    # 7. 保存 CSV
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 8. 输出检查结果
    print("\n转换完成！")
    print(f"保留记录数：{len(result):,}")
    print(f"删除无效记录数：{dropped:,}")
    print(f"标签种类数：{result['label'].nunique()}")

    print("\nSub_Cat 标签分布（前 20 类）：")
    print(result["label"].value_counts().head(20).to_string())

    print("\n转换后字段：")
    print(result.columns.tolist())

    print("\n前 5 条记录：")
    print(result.head().to_string(index=False))

    print(f"\n输出文件：{output_path.resolve()}")


if __name__ == "__main__":
    main()
