import csv
from datetime import datetime
from pathlib import Path

csv.field_size_limit(100_000_000)

base = Path(__file__).resolve().parent
source = base / "raw" / "salad_test.csv"
output_dir = base / "ready"
output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / "salad_test_sorted_label.csv"

if not source.exists():
    raise FileNotFoundError(f"找不到原始数据：{source}")

print("正在读取原始数据，请稍等……")

rows = []

with open(source, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    original_fields = reader.fieldnames

    if not original_fields:
        raise ValueError("CSV 没有表头")

    required = {"timestamp", "attack_category", "src_ip", "dst_ip"}
    missing = required - set(original_fields)
    if missing:
        raise ValueError(f"原始文件缺少必要字段：{missing}")

    output_fields = [
        "label" if name == "attack_category" else name
        for name in original_fields
    ]

    for index, row in enumerate(reader):
        raw_time = (row.get("timestamp") or "").strip()
        try:
            parsed_time = datetime.fromisoformat(
                raw_time.replace("Z", "+00:00")
            )
        except ValueError:
            raise ValueError(
                f"第 {index + 2} 行的时间戳无法解析：{raw_time}"
            )

        rows.append((parsed_time, index, row))

        if (index + 1) % 100000 == 0:
            print(f"已读取 {index + 1:,} 条记录")

print(f"读取完成，共 {len(rows):,} 条记录")
print("正在按时间从早到晚排序……")

rows.sort(key=lambda item: (item[0], item[1]))

print("正在写入新 CSV……")

with open(output, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=output_fields,
        extrasaction="ignore"
    )
    writer.writeheader()

    for _, _, row in rows:
        row["label"] = row.pop("attack_category")
        writer.writerow(row)

print("\n转换完成！")
print(f"原始记录数：{len(rows):,}")
print(f"字段数量：{len(output_fields)}")
print(f"输出文件：{output}")
print("已按 timestamp 升序排列。")
print("attack_category 已改名为 label。")
print("其他字段和记录均予以保留。")
