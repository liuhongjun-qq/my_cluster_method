#图构建与排序：时间衰减、稀疏图构建、Fiedler 谱排序

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from .config import Config


# 这个函数会观察一批事件发生的时间间隔，根据这些间隔的大小，确定短半衰期 short 和长半衰期 long。
def calibrate_half_lives(timestamps, config: Config):
    times = np.sort(np.asarray(timestamps, dtype=float))
    # 计算相邻事件的时间间隔
    gaps = np.diff(times)
    # 去掉等于0的时间间隔
    gaps = gaps[gaps > 0]
    # 代表当前这批事件正时间间隔的中位数，用来估计一个典型的事件时间间隔。
    median = float(np.median(gaps)) if len(gaps) else 1.0
    # 中位数绝对偏差。
    mad = float(np.median(np.abs(gaps - median))) if len(gaps) else 0.0
    short = config.short_half_life or median
    long = config.long_half_life or max(short, median + 3 * mad)
    return float(short), float(long)


def temporal_decay(delta, floor, alpha, short_half_life, long_half_life):
    if not 0 <= floor <= 1 or not 0 <= alpha <= 1:
        raise ValueError("floor and alpha must be in [0, 1]")
    if short_half_life <= 0 or long_half_life <= 0:
        raise ValueError("half lives must be positive")
    delta = np.abs(np.asarray(delta, dtype=float))
    return floor + (1 - floor) * (
        alpha * np.exp2(-delta / short_half_life)
        + (1 - alpha) * np.exp2(-delta / long_half_life)
    )


# 给每条网络安全事件寻找与它相似的其他事件，同时考虑它们发生的时间是否接近，
# 最后把符合条件的事件连接起来，形成一张稀疏关系图。
def mutual_topk_graph(forest, timestamps, config: Config):
    timestamps = np.asarray(timestamps, dtype=float)
    if not np.isfinite(timestamps).all() or len(timestamps) != len(forest.leaf_ids_):
        raise ValueError("timestamps must be finite and match the forest")
    short, long = calibrate_half_lives(timestamps, config)
    rows, columns, weights = [], [], []
    for index in range(len(timestamps)):
        candidates, similarities = forest.neighbors(index)
        # 把随机划分森林计算出来的内容相似度，与时间衰减权重相乘，得到两个事件之间的综合相似度 scores。
        scores = similarities * temporal_decay(
            timestamps[index] - timestamps[candidates], config.decay_floor, config.alpha, short, long,
        )
        positive = scores > 0
        candidates, scores = candidates[positive], scores[positive]
        # selected 保存的是下标，而 candidates[selected] 才是真正选中的事件编号。
        # 排序
        selected = np.lexsort((candidates, -scores))[:config.top_k]
        rows.extend([index] * len(selected))
        columns.extend(candidates[selected].tolist())
        weights.extend(scores[selected].tolist())
    # 创建一张 n×n 的事件关系表，把每个 weights[i] 放在第 rows[i] 行、第 columns[i] 列。”
    directed = sparse.csr_matrix((weights, (rows, columns)), shape=(len(timestamps), len(timestamps)))
    # 把两个矩阵相同位置的数值进行比较，保留较小的那个数。
    # 把计算结果转换成 CSR（Compressed Sparse Row，压缩稀疏行）格式
    graph = directed.minimum(directed.T).tocsr()
    # 清理内部显式存储的零值
    # graph.eliminate_zeros() 不会改变图的实际连接关系，也不会改变事件数量，只是优化稀疏矩阵的存储。
    graph.eliminate_zeros()
    return graph, {"short_half_life": short, "long_half_life": long}


'''
order    谱排序后的事件编号顺序
components    每条事件属于哪个连通分量
warnings    计算过程中出现的警告信息
'''
def fiedler_order(graph, seed=42):
    graph = sparse.csr_matrix(graph)
    n_events = graph.shape[0]
    if graph.shape[1] != n_events:
        raise ValueError("graph must be square")
    if not n_events:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), []
    # 求连通分量：返回两个值：第一个结果：n_components（连通分量的数目），
    # components 数组的位置表示事件编号，数组中的值表示该事件所属的连通分量编号。
    _, components = connected_components(graph, directed=False)
    # 找出数组中非零元素的位置。对于布尔数组，就是找出所有 True 所在的位置。
    groups = [np.flatnonzero(components == label) for label in np.unique(components)]
    # 按照每个连通分量中第一个事件的编号，从小到大排列 groups 中的各个连通分量。
    groups.sort(key=lambda indices: int(indices[0]))
    # 保存最终经过 Fiedler 谱排序后的事件编号。
    order = []
    # 保存谱排序过程中出现的警告信息
    warnings = []

    for indices in groups:
        if len(indices) <= 2:
            order.extend(indices.tolist())
            continue
        operator = laplacian(graph[indices][:, indices], normed=True)
        if len(indices) <= 64:
            values, vectors = np.linalg.eigh(operator.toarray())
        else:
            try:
                values, vectors = eigsh(
                    operator, k=2, which="SM", tol=1e-6,
                    v0=np.random.default_rng(seed).normal(size=len(indices)),
                    maxiter=max(1000, len(indices) * 20),
                )
            except ArpackNoConvergence:
                warnings.append(f"Lanczos did not converge for component size {len(indices)}; index order used")
                order.extend(indices.tolist())
                continue
        vector = vectors[:, np.argsort(values)[1]]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector = -vector
        order.extend(indices[np.lexsort((indices, vector))].tolist())
    return np.asarray(order, dtype=int), components, warnings
