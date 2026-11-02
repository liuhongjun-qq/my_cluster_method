# 紧凑统计：Count-Min Sketch、HyperLogLog 等统计结构

import json

import numpy as np

from ._02encoding import stable_hash



# 提取事件的上下文特征 → 提取载荷特征 → 合并成事件的 Token 集合。
def context_tokens(context):
    excluded = {"label", "ground_truth", "target", "timestamp", "event_id"}

    result = []
    # 逐个取出字典中的字段名称和字段值。
    for name, value in sorted(context.items()):
        if name not in excluded:
            token = f"{name}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            result.append(token)

    return result


class CountMinSketch:
    def __init__(self, width=128, depth=4, salt=""):
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.table = np.zeros((depth, width), dtype=np.uint32)
        self.salt = salt

    # add(token)通过哈希定位，增加计数
    def add(self, token, count=1):
        if count < 0:
            raise ValueError("count must be nonnegative")
        # 遍历 CMS 的每一行:，需要在每一行分别更新一个计数位置。
        for row in range(self.table.shape[0]):
            column = stable_hash(f"{row}:{token}", self.salt) % self.table.shape[1]
            value = int(self.table[row, column]) + count
            if value > np.iinfo(np.uint32).max:
                raise OverflowError("Count-Min Sketch counter overflow")
            self.table[row, column] = value

# estimate(token)查询各行对应的计数，取最小值
    def estimate(self, token):
        return min(
            int(self.table[row, stable_hash(f"{row}:{token}", self.salt) % self.table.shape[1]])
            for row in range(self.table.shape[0])
        )


class HyperLogLog:
    def __init__(self, precision=8, salt=""):
        if not 4 <= precision <= 16:
            raise ValueError("precision must be in [4, 16]")
        self.precision = precision
        self.salt = salt
        self.registers = np.zeros(2**precision, dtype=np.uint8)

    def add(self, token):
        hashed = stable_hash(token, self.salt)
        # 前4位用来选择寄存器，剩余60位用于计算统计值。
        remaining_bits = 64 - self.precision
        index = hashed >> remaining_bits
        remaining = hashed & ((1 << remaining_bits) - 1)
        rank = remaining_bits - remaining.bit_length() + 1
        self.registers[index] = max(int(self.registers[index]), rank)

    def estimate(self):
        count = len(self.registers)
        alpha = {16: 0.673, 32: 0.697, 64: 0.709}.get(count, 0.7213 / (1 + 1.079 / count))
        estimate = alpha * count**2 / np.exp2(-self.registers.astype(float)).sum()
        empty = np.count_nonzero(self.registers == 0)
        if estimate <= 2.5 * count and empty:
            estimate = count * np.log(count / empty)
        return float(estimate)
