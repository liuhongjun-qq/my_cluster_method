# 同源块签名：生成、保存和加载 HoSig 签名

from collections import Counter
from dataclasses import dataclass
import json

import numpy as np

from .config import Config
from ._02encoding import payload_tokens, stable_hash
from ._06sketches import CountMinSketch, HyperLogLog, context_tokens


@dataclass
class HoSig:
    block_id: str
    # 特征空间标识
    feature_space_id: str
    # 同源块包含多少条事件 
    count: int
    # 同源块的原型向量
    prototype: np.ndarray
    # 时间分布直方图
    time_histogram: np.ndarray
    # 上下文：用较少的存储空间，近似记录上下文信息的出现频率。
    cms: np.ndarray
    # 以较低的内存开销，估计同源块涉及多少种不同的上下文元素。
    hll_registers: np.ndarray
    # 保存的是计算出来的估计结果。
    distinct_context_estimate: float
    # 可能反映块与外部的连接情况
    boundary_ratio: float | None
    # 帮助解释块的主要特征
    top_features: list[dict]

    @property
    def numeric_bytes(self):
        return int(sum(array.nbytes for array in (
            self.prototype, self.time_histogram, self.cms, self.hll_registers,
        )))

    def to_dict(self):
        return {
            "block_id": self.block_id, "feature_space_id": self.feature_space_id,
            "count": self.count, "prototype": self.prototype.tolist(),
            "time_histogram": self.time_histogram.tolist(), "cms": self.cms.tolist(),
            "hll_registers": self.hll_registers.tolist(),
            "distinct_context_estimate": self.distinct_context_estimate,
            "boundary_ratio": self.boundary_ratio, "boundary_ratio_direction": "lower_is_clearer",
            "top_features": self.top_features, "numeric_bytes": self.numeric_bytes,
        }

    @classmethod
    def from_dict(cls, record):
        return cls(
            block_id=record["block_id"], feature_space_id=record["feature_space_id"],
            count=record["count"], prototype=np.asarray(record["prototype"], dtype=np.float32),
            time_histogram=np.asarray(record["time_histogram"], dtype=np.uint32),
            cms=np.asarray(record["cms"], dtype=np.uint32),
            hll_registers=np.asarray(record["hll_registers"], dtype=np.uint8),
            distinct_context_estimate=record["distinct_context_estimate"],
            boundary_ratio=record["boundary_ratio"], top_features=record["top_features"],
        )

# present整个批次有多少个事件包含该 Token，
# present_in_block当前同源块有多少个事件包含该 Token
def _information_gain(total, block_size, present, present_in_block):
    # 第一个位置：块内包含 Token 的事件数
    # 第二个位置：块内不包含 Token 的事件数
    # 第三个位置：块外包含 Token 的事件数
    # 第四个位置：块外不包含 Token 的事件数
    counts = np.array([
        [present_in_block, block_size - present_in_block],
        [present - present_in_block, total - block_size - present + present_in_block],
    ], dtype=float)
    # 第二行：计算 joint 联合概率
    joint = counts / total
    # 第三行：计算 independent 独立概率
    # joint.sum(axis=1)：判断属于同源块的概率：
    # joint.sum(axis=0)：判断时间中含有token的概率
    # np.outer() 表示计算两个向量的外积。
    independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))

    mask = joint > 0
    return max(0.0, float(np.sum(joint[mask] * np.log2(joint[mask] / independent[mask]))))


def build_signatures(events, graph, blocks, rff, config: Config, batch_id="0"):
    token_sets = []
    for event in events:
        # 1. 提取当前事件的上下文 Token：得到一个集合
        context_set = set(context_tokens(event.context))
        # 2. 提取当前事件的载荷 Token
        payload_set = set()
        for token in payload_tokens(event.payload):
            payload_set.add(f"payload:{token}")
        # 3. 合并两个集合
        combined_set = context_set | payload_set
        # 4. 保存当前事件的 Token 集合
        token_sets.append(combined_set)
    # Counter 负责统计这些 Token 的出现次数。
    global_counts = Counter(token for tokens in token_sets for token in tokens)
    signatures = []
    n_events = len(events)
    # 遍历每个同源块，为每个同源块生成一个 HoSig 签名。
    for block_number, indices in enumerate(blocks):
       
        # 时间统计：
        # 统计块内每条时间的原始时间戳
        times = np.array([events[index].timestamp for index in indices])
        #相对时间：把当前同源块中最早发生的事件作为时间起点，计算其他事件距离这个起点过去了多少时间 
        relative_times = times - times.min()
        # 时间跨度=块内的最晚时间-最早的时间
        span = float(relative_times.max())
        # 把当前同源块的时间跨度划分成若干个时间区间，然后统计每个区间内发生了多少条事件。
        # np.histogram 返回 (counts, bin_edges)，所以 [0] 取的是每个箱子里样本数量；
        histogram = np.histogram(relative_times, bins=config.time_bins, range=(0.0, span if span > 0 else 1.0))[0].astype(np.uint32)
        
    
        # 上下文压缩统计
        # 某个上下文 Token 出现了多少次？
        cms = CountMinSketch(config.cms_width, config.cms_depth, config.context_salt)
        # 一共有多少种不同的上下文 Token？
        hll = HyperLogLog(config.hll_precision, config.context_salt)

        for index in indices:
            for token in context_tokens(events[index].context):
                cms.add(token)
                hll.add(token)
        # 计算当前同源块内部所有事件之间的连接权重之和。
        within_weight = float(graph[indices][:, indices].sum())
        # 计算当前同源块与其他同源块中的事件之间的连接权重之和。
        crossing_weight = float(graph[indices].sum()) - within_weight
        # 当前同源块内部所有连接权重的总和
        # 除以块内可能的有序事件对数量 n*(n-1)
        # 得到同源块内部的平均连接权重
        within_mean = within_weight / (len(indices) * max(len(indices) - 1, 1))
        cross_mean = crossing_weight / max(len(indices) * (n_events - len(indices)), 1)
        # 比较当前同源块与外部事件的平均连接强度，以及同源块内部的平均连接强度，从而衡量同源块的边界是否清晰。
        ratio = cross_mean / within_mean if within_mean > 0 else None
        # 统计当前同源块中，每一种 Token 出现在多少个事件中。
        block_counts = Counter(token for index in indices for token in token_sets[index])
        features = []
        for token, present in block_counts.items():
            feature = {
                "token_hash": f"{stable_hash(token, config.context_salt):016x}",
                "information_gain": _information_gain(
                    n_events,
                    len(indices),
                    global_counts[token],#整个批次有多少个事件包含该 Token
                    present#当前同源块有多少个事件包含该 Token
                ),
                "presence_count": present
            }
            features.append(feature)
        # 按照每个 Token 的信息增益从大到小排序。如果两个 Token 的信息增益相同，就按照它们的哈希值从小到大排序。
        features.sort(key=lambda item: (-item["information_gain"], item["token_hash"]))
        signatures.append(HoSig(
            block_id=f"{batch_id}:{block_number}", 
            # 给当前生成的 HoSig 签名记录一个“特征空间标识”，用于说明这个签名是基于哪一套特征表示规则生成的。
            feature_space_id=config.feature_space_id,
            count=len(indices), 
            prototype=rff[indices].mean(axis=0).astype(np.float32),
            time_histogram=histogram, 
            cms=cms.table, 
            hll_registers=hll.registers,
            # 不同token的数目
            distinct_context_estimate=hll.estimate(),
            boundary_ratio=ratio,
            top_features=features[:config.top_features],
        ))
    return signatures


def save_signatures(path, signatures):
    with open(path, "w", encoding="utf-8") as handle:
        for signature in signatures:
            handle.write(json.dumps(signature.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def load_signatures(path):
    with open(path, encoding="utf-8") as handle:
        return [HoSig.from_dict(json.loads(line)) for line in handle if line.strip()]

