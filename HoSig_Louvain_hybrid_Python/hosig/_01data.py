#数据输入：读取事件、定义事件结构、划分微批、生成合成数据
from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Event:
    # 时间戳
    timestamp: float
    src_ip: str
    dst_ip: str
    # 用于保存这条网络安全事件的内容信息。
    # 可以保存原始内容的文本形式，也可以保存从网络记录中提取、整理后的内容特征。
    # payload 可以由协议、服务、DNS、HTTP等字段拼接而成。,
    payload: str = ""
    # context 是用来补充描述网络安全事件的背景信息的。
    # **`{}` 只在类定义的时候创建一次，所有实例共用同一个字典**。修改一个实例的 data，别的实例也跟着被改。
    context: dict = field(default_factory=dict)
    label: str | None = None
    event_id: str = ""

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")



def parse_timestamp(value):
    try:
        # 1. 首先尝试按照 Unix 时间戳解析
        result = float(value)

    except (TypeError, ValueError):
        # 2. 如果不是数字，则按照 ISO 时间格式解析
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            # 3. 没有时区时，保留原始时间的相对时间关系
            # 使用无时区的参考起点，不声称它属于 UTC 或其他时区
            epoch = datetime(1970, 1, 1)
            result = (parsed - epoch).total_seconds()
        else:
            # 4. 如果原始时间带有时区，正常转换
            result = parsed.timestamp()

    # 5. 检查时间戳是否有效
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")

    return result


# read_events() 是一个数据读取与格式转换函数，
# 负责把 CSV 或 JSONL 文件中的原始网络安全数据，转换成 HoSig-Align I 后续算法能够统一处理的 Event 事件对象。
def read_events(path, data_format="auto", proxy_interval=None, limit=None):
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if data_format == "auto":
        data_format = "jsonl" if path.suffix.lower() == ".jsonl" else "csv"
    if data_format not in {"jsonl", "csv", "network-csv"}:
        raise ValueError(f"Unknown data format: {data_format}")
    if data_format == "network-csv" and (
        proxy_interval is None or not math.isfinite(proxy_interval) or proxy_interval <= 0
    ):
        raise ValueError("network-csv requires explicit --proxy-interval > 0; timestamps are synthetic")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if data_format == "jsonl":
            rows = (json.loads(line) for line in handle if line.strip())
        else:
            # 如果当前不是 jsonl 读取模式，就使用 CSV 解析器读取文件，并将每一行数据整理成一个字典。
            reader = csv.DictReader(handle)
            # 检查CSV文件是否包含·程序需要的字段。如果缺少必要字段，就报错；如果字段符合要求，就把 reader 交给后续代码使用。
            # 检查CSV文件是否有表头
            if not reader.fieldnames:
                raise ValueError("Input has no CSV header")
            required = {"src_ip", "dst_ip"}
            if data_format != "network-csv":
                required.add("timestamp")
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Missing CSV columns: {sorted(missing)}")
            rows = reader
        # 逐条读取；普通 CSV/JSONL 使用真实时间，network-csv 保留原来的模拟时间模式。
        for index, row in enumerate(itertools.islice(rows, limit)):
            try:
                # 不属于内容特征的字段：时间、源/目的 IP、标签、事件标识。
                # 注意：duration 等持续时间/统计字段按本次实验要求保留。
                excluded_fields = {
                    "timestamp", "ts", "src_ip", "dst_ip",
                    "id.orig_h", "id.resp_h", "source_ip", "destination_ip",
                    "event_id", "uid", "flow_id", "id",
                    "label", "detailed-label", "detailed_label", "det_label",
                    "attack_cat", "attack_category", "attack_type",
                    "type", "class", "target", "ground_truth", "is_attack",
                    "binary_label", "category_label",
                    "start_time", "end_time", "start_timestamp", "end_timestamp",
                    "time_delta", "time_interval",
                    "payload", "context","is_malicious"
                }

                def is_valid(value):
                    # 数字 0 是有效值；None、NaN、空字符串和常见缺失标记不是。
                    if value is None:
                        return False
                    if isinstance(value, float) and not math.isfinite(value):
                        return False
                    if isinstance(value, str):
                        return value.strip().lower() not in {
                            "", "-", "nan", "none", "null", "?", "n/a",
                        }
                    return True

                # 自动遍历当前文件的实际字段；每个有效特征同时进入两种表示。
                features = {}
                for name, value in row.items():
                    if not isinstance(name, str):
                        continue
                    if name.strip().lower() in excluded_fields or not is_valid(value):
                        continue
                    features[name] = value

                # payload 是文本：将每个特征转换为 name=value 形式。
                # 若原文件已有 payload，则保留它，并追加其他有效字段。
                existing_payload = row.get("payload")
                payload_parts = []
                if is_valid(existing_payload):
                    payload_parts.append(str(existing_payload))
                for name, value in features.items():
                    if isinstance(value, (dict, list, tuple)):
                        value_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
                    else:
                        value_text = str(value).strip()
                    payload_parts.append(f"{name}={value_text}")
                payload = " ".join(payload_parts)

                # context 是字典：保留原有 context，同时加入所有有效特征。
                context = row.get("context") or {}
                if isinstance(context, str):
                    context = json.loads(context)
                if not isinstance(context, dict):
                    raise ValueError("context must be a JSON object")
                context = {
                    name: value for name, value in context.items()
                    if isinstance(name, str)
                    and name.strip().lower() not in excluded_fields
                    and is_valid(value)
                }
                context.update(features)

                # 保留旧 network-csv 的模拟时间与标签规则；其他模式使用真实时间。
                if data_format == "network-csv":
                    timestamp = index * proxy_interval
                    label = row.get("type") or row.get("label")
                    event_id = str(index)
                else:
                    timestamp = parse_timestamp(row["timestamp"])
                    label = row.get("label")
                    event_id = str(row.get("event_id") or index)

                yield Event(
                    timestamp=timestamp,
                    src_ip=str(row["src_ip"]),
                    dst_ip=str(row["dst_ip"]),
                    payload=payload,
                    context=context,
                    label=None if label in (None, "") else str(label),
                    event_id=event_id,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid event at record {index + 1}: {error}") from error

# 连续输入的网络安全事件，按照指定的事件数量划分成一个个微批次（Micro-batch），然后逐批输出。
# 以列表的形式来输出的 batch = []
def micro_batches(events: Iterable[Event], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    for event in events:
        batch.append(event)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch



'''
这个函数模拟一个网络安全监测系统，不断产生安全事件
并且让这些事件具有不同的攻击战役标签、IP 地址、时间、通信内容和上下文信息。
'''
'''攻击战役2，利用攻击特征2，攻击家族2，请求变体1，随机噪声45和随机噪声12'''
def generate_synthetic(n_events=768, campaigns=3, seed=42):
    if n_events < 1 or not 1 <= campaigns <= 200:
        raise ValueError("n_events must be positive and campaigns must be in [1, 200]")
    # 表示创建一个随机数生成器，并使用指定的随机种子初始化它
    generator = np.random.default_rng(seed)
    # 给模拟事件设置一个统一的起始时间。
    timestamp = datetime(2025, 1, 6, tzinfo=timezone.utc).timestamp()
    # 提前为每一条模拟事件分配一个随机的攻击战役编号。
    shuffled_labels = generator.integers(0, campaigns, size=n_events)

    for index, campaign in enumerate(shuffled_labels):
        # 如果当前事件位于每 100 条事件的前 20 条，就使用 2；否则使用 72。
        # generator.exponential(2)**用这个随机数生成器，抽取若干个指数分布的非负随机数**
        # scale 控制随机数的平均大小。
        # rng.exponential(scale, size)
        timestamp += generator.exponential(2 if index % 100 < 20 else 72)
        # f"campaign{campaign} exploit{campaign} family{campaign}"
        # 得到字符串：`"campaign1 exploit1 family1"`
        # ["campaign1 exploit1 family1", "campaign1 exploit1 family1", "campaign1 exploit1 family1"]
        # 者通过重复三次，让每个战役对应的文本特征多次出现。
        # 这样做的效果是：强化同一个攻击战役的共同内容特征。
        words = " ".join([f"campaign{campaign} exploit{campaign} family{campaign}"] * 3)#特征
        nuisance = " ".join(f"noise{value}" for value in generator.integers(0, 100, size=2))#噪声
        yield Event(
            timestamp=timestamp,
            src_ip=f"10.{campaign + 1}.{index % 3}.{index % 250 + 1}",
            dst_ip=f"172.{campaign + 16}.0.{index % 5 + 1}",
            payload=f"{words} request variant{index % 4} {nuisance}",
            context={"service": f"service{campaign}", "sensor": f"sensor{index % 4}"},
            label=f"campaign{campaign}", 
            # 事件唯一编号
            event_id=str(index),
        )

def write_events(path, events):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
