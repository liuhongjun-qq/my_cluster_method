# Homologous Blocks 发现：计算边界收益、检测分段边界

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .config import Config


@dataclass
class Segmentation:
    '''
    labels    每个事件被分配到哪个同源块
    blocks    每个同源块包含哪些事件
    boundaries    排序序列中的切分位置
    costs    各候选切分位置的对比代价值
    penalty    分割惩罚值，用于抑制过度切分
    '''
    labels: np.ndarray
    blocks: list[np.ndarray]
    boundaries: list[int]
    costs: np.ndarray
    penalty: float


def contrastive_costs(graph, order, rff, left_window=64, right_window=64, eta=1.0):
    n_events = len(order)
    costs = np.zeros(n_events + 1, dtype=float)
    features = rff[order].astype(float)
    # prefix 叫作前缀和，它提前记录从开头到当前位置之前的向量累加结果
    prefix = np.vstack((np.zeros(features.shape[1]), np.cumsum(features, axis=0)))
    ordered_graph = sparse.csr_matrix(graph)[order][:, order]
    for boundary in range(1, n_events):
        start = max(0, boundary - left_window)
        end = min(n_events, boundary + right_window)
        left_size, right_size = boundary - start, end - boundary
        difference = (prefix[boundary] - prefix[start]) / left_size - (prefix[end] - prefix[boundary]) / right_size
        left = ordered_graph[start:boundary, start:boundary].sum() / left_size**2
        right = ordered_graph[boundary:end, boundary:end].sum() / right_size**2
        cross = ordered_graph[start:boundary, boundary:end].sum() / (left_size * right_size)
        costs[boundary] = difference @ difference + eta * (left + right - 2 * cross)
    return costs


def _optimal_boundaries(start, end, gains, minimum):
    length = end - start
    if length < 2 * minimum:
        return []
    best = np.zeros(length + 1)
    previous = np.full(length + 1, -1, dtype=int)
    boundary = np.full(length + 1, -1, dtype=int)
    for offset in range(1, length + 1):
        best[offset] = best[offset - 1]
        previous[offset] = offset - 1
        absolute = start + offset
        if offset >= minimum and end - absolute >= minimum:
            candidate = best[offset - minimum] + gains[absolute]
            if candidate > best[offset] + 1e-12:
                best[offset] = candidate
                previous[offset] = offset - minimum
                boundary[offset] = absolute
    selected = []
    cursor = length
    while cursor > 0:
        if boundary[cursor] >= 0:
            selected.append(int(boundary[cursor]))
        cursor = previous[cursor]
    return sorted(selected)


# Fiedler 排序 → 计算 costs → 扣除 penalty 得到 gains → 根据连通分量强制切分 → 动态规划选择内部切分点 → 生成同源块。
def contrastive_segment(graph, order, components, rff, config: Config):
    n_events = len(order)
    costs = contrastive_costs(graph, order, rff, config.left_window, config.right_window, config.eta)
    # 计算分割惩罚值（penalty），防止算法把事件切分成太多零碎的同源块。
    penalty = config.segmentation_beta * np.log(max(n_events, 2))
    gains = costs - penalty
    #按照 Fiedler 排序后的事件顺序，重新排列每条事件所属的连通分量编号。
    ordered_components = components[order]
    # forced：先把图中互不连通的不同事件群强制分开。
    forced = (np.flatnonzero(np.diff(ordered_components) != 0) + 1).tolist()
    # 把前面找到的强制切分位置，加上整个事件序列的起点和终点，从而确定每个连通分量对应的范围。
    component_ends = [0] + forced + [n_events]
    boundaries = list(forced)
    for start, end in zip(component_ends[:-1], component_ends[1:]):
        boundaries.extend(_optimal_boundaries(start, end, gains, config.min_block_size))
    boundaries = sorted(set(boundaries))
    ends = [0] + boundaries + [n_events]
    blocks = [order[start:end] for start, end in zip(ends[:-1], ends[1:]) if end > start]
    labels = np.empty(n_events, dtype=int)
    for label, indices in enumerate(blocks):
        labels[indices] = label
    return Segmentation(labels, blocks, boundaries, costs, float(penalty))
