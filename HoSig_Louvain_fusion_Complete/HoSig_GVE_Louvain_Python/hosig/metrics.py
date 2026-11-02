# 计算聚类和块质量指标

import math
from collections import Counter

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# 计算各个同源块的原型向量之间的距离，并根据距离生成 IIRk 和 IIRnorm 两组评价数据。
def boundary_clarity(entities, proportion=0.05):
    if not 0 < proportion <= 1:
        raise ValueError("proportion must be in (0, 1]")
    entities = np.asarray(entities, dtype=float)
    count = len(entities)
    if count < 2:
        return {"entity_count": count, "neighbor_rank": None, "iir_k": [], "iir_norm": []}
    #StandardScaler().fit_transform()：
    # fit它会计算每一列特征的平均值和标准差。
    # transform()：根据统计信息转换数据,就是先减去这一列的平均值，再除以这一列的标准差。
    # standardized.shape=({实体数量},{特征数量})
    standardized = StandardScaler().fit_transform(entities)
    # 计算每个实体与其他所有实体之间的距离
    # 形状： (实体数量, 实体数量)
    # 含义： distances[i, j] 表示第 i 个实体与第 j 个实体之间的距离。
    distances = cdist(standardized, standardized)
    #把距离矩阵中，对角线上的所有数值都改成正无穷大 np.inf
    # 排除实体自身，防止后续寻找近邻时，把自身错误地当成最近邻
    np.fill_diagonal(distances, np.inf)
    # 计算一个合理的近邻排名，并保证排名至少为1、最多不超过其他实体的数量。
    rank = min(count - 1, max(1, math.ceil(proportion * count)))
    # 找出每个实体的第 rank 近邻距离，并保存起来。
    # np.partition() 不保证把整行完全排序，只保证指定位置上的数值正确，并满足分区条件。
    # **不完全排序，只做分区**：保证下标 `kth` 位置上的元素，就是数组完全排好序后该位置的值；
    # **左边全部 ≤ 它，右边全部 ≥ 它**，但是左边内部、右边内部是乱序、不保证有序NumPy
    # np.partition(a, kth, axis=-1)
    # nearest.shape=(N,)
    nearest = np.partition(distances, rank - 1, axis=1)[:, rank - 1]
    median = float(np.median(nearest))
    # 把 nearest 中每个近邻距离取倒数，并将计算结果保存到 reciprocal 列表中。
    # reciprocal 可以让我们通过一个数值，观察各个原型与指定近邻之间的接近程度。
    reciprocal = [float(1 / value) if value > 0 else None for value in nearest]
    # normalized 大于1，表示该同源块的近邻距离小于中位数；
    # 小于1，则表示近邻距离大于中位数。它只是距离的相对比较值，不能单独用来判断整个同源块划分得好不好。
    normalized = [float(median / value) if value > 0 and median > 0 else None for value in nearest]
    return {
        # 一共有多少个同源块原型参与计算
        "entity_count": count,
        # 选择考察第几个近邻 
        "neighbor_rank": rank, 
        # 每个原型对应的近邻距离的倒数
        "iir_k": reciprocal,
        # 每个原型的近邻距离相对于整体中位数的比较值
        "iir_norm": normalized, 
        # 与其他原型距离恰好为零的数量
        "duplicate_entity_count": int(np.count_nonzero(nearest == 0)),
    }


'''
输入：N 是当前微批次的事件数量，B 是同源块数量。
embeddings    result.embeddings    np.ndarray (N,D)    每条事件的内容编码向量
labels    result.labels    np.ndarray (N,)    每条事件被分配的同源块编号
graph    result.graph    稀疏矩阵 (N,N)    事件之间的图连接关系及权重
entities    entities    np.ndarray (B,D_RFF)    每个同源块的原型向量
truth    [event.label for event in batch]    list，长度N    数据集中已有的真实标签
seed    config.seed    int    随机种子


'''
def cluster_metrics(embeddings, labels, graph=None, entities=None, truth=None, seed=42):
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    valid = labels >= 0
    # np.unique() 去除重复的标签。
    # 看有哪些块
    '''
    np.unique([0, 0, 1, 1, 2])
    # array([0, 1, 2])
    '''
    cluster_ids = np.unique(labels[valid])

    scores = {
        # 总事件
        "n_events": len(labels),
        # 总的分块数
        "n_clusters": len(cluster_ids),
        # 噪声事件是当前微批次中不属于任何有效同源块的事件，
        # 计算噪声事件的比例-每条事件都会被分配到一个同源块。（没有噪声）
        "noise_rate": float(np.mean(~valid)) if len(labels) else 0.0,
        # 当前微批次划分出来的同源块，是否满足“块内事件相似、块间事件差异明显”。
        # 越大越好。
        "ch": None, 
        # 轮廓系数：评价每条事件是否被划分到了合适的同源块。
        "silhouette": None, 
        # modularity：模块度
        # 构建的稀疏图是否被合理地划分成不同同源块。
        # ：模块度越高，通常表示图上的同源块划分越合理。
        "modularity": None,
        # 比较算法发现的同源块与数据集真实标签的划分是否一致。
        # ARI 越接近1，表示算法分块与真实标签越一致。
        "ari": None, 
        # 预测分块与真实标签的信息关联程度
        "nmi": None,
        # 各真实类别的事件数量
        # 这个字段用于统计每种真实标签分别出现了多少次。
        "truth_class_count": 0, 
        # 每种真实类别有多少条事件
        "truth_label_counts": {},
    }

    if 1 < len(cluster_ids) < np.count_nonzero(valid):
        # ch指数
        scores["ch"] = float(calinski_harabasz_score(embeddings[valid], labels[valid]))
        try:
            scores["silhouette"] = float(silhouette_score(
                embeddings[valid], labels[valid], sample_size=min(2000, int(valid.sum())), random_state=seed,
            ))

        except ValueError:
            scores["silhouette"] = None
    if graph is not None and np.count_nonzero(valid):
        # 把稀疏矩阵转换成图
        network = nx.from_scipy_sparse_array(graph[valid][:, valid])
        valid_labels = labels[valid]
        # 把每个同源块包含的事件编号找出来，整理成一个列表，交给后面的模块度计算函数使用。
        '''
        communities = [
            {0, 1},
            {2, 3},
            {4}
        ]
        '''
        communities = [set(np.flatnonzero(valid_labels == label).tolist()) for label in cluster_ids]
        if network.size(weight="weight") > 0:
            scores["modularity"] = float(nx.community.modularity(network, communities, weight="weight"))
    # 算法划分出来的组，与真实标签所反映的分组情况有多一致？
    if truth is not None:
        known = np.array([value is not None for value in truth])
        # 
        scores["truth_label_counts"] = dict(sorted(Counter(str(value) for value in truth if value is not None).items()))
        scores["truth_class_count"] = len(scores["truth_label_counts"])
        if np.count_nonzero(known) >= 2:
            known_truth = np.array([str(value) if value is not None else "" for value in truth])
            # 参数的下表不一定是事件的编号
            scores["ari"] = float(adjusted_rand_score(known_truth[known], labels[known]))
            scores["nmi"] = float(normalized_mutual_info_score(known_truth[known], labels[known]))
    if entities is None:
        # 计算每个分组的中心向量
        # entities.shape=(K,D)
        entities = np.array([embeddings[labels == label].mean(axis=0) for label in cluster_ids])
    scores["boundary_clarity"] = boundary_clarity(entities)
    return scores
