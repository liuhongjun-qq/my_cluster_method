# 计算签名距离、进行块对齐、构建或更新超图

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import jensenshannon, squareform


def dtw_distance(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not len(first) or not len(second):
        raise ValueError("DTW sequences must be nonempty")
    # 保存上一轮计算结果的一行数组。
    previous = np.full(len(second) + 1, np.inf)
    previous[0] = 0
    for first_value in first:
        current = np.full(len(second) + 1, np.inf)
        for column, second_value in enumerate(second, 1):
            current[column] = (first_value - second_value)**2 + min(
                previous[column], current[column - 1], previous[column - 1],
            )
        previous = current
    return float(np.sqrt(previous[-1]))


def block_distance(first, second, weights=(1 / 3, 1 / 3, 1 / 3)):
    # 确保相关特征表示配置兼容
    if first.feature_space_id != second.feature_space_id:
        raise ValueError("Incompatible feature spaces; all sites must share encoder, RFF and sketch configuration")
    if len(weights) != 3 or any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("weights must contain three finite nonnegative values with positive sum")
    # 计算内容距离： prototype的欧式距离
    prototype = float(np.linalg.norm(first.prototype - second.prototype))
    # 计算时间距离
    temporal = dtw_distance(first.time_histogram, second.time_histogram)
    # 分别取出两个签名中的 CMS（计数最小草图），把数据转换为浮点数，再整理成一维数组，方便后续比较。
    first_counts, second_counts = first.cms.astype(float).ravel(), second.cms.astype(float).ravel()
    # 计算上下文距离js
    if first_counts.sum() == 0 and second_counts.sum() == 0:
        contextual = 0.0
    elif first_counts.sum() == 0 or second_counts.sum() == 0:
        contextual = float(np.sqrt(np.log(2)))
    else:
        contextual = float(jensenshannon(first_counts, second_counts))
    return float(np.dot(weights, (prototype, temporal, contextual)))


@dataclass
class Hyperedge:
    # 这条超边包含哪些签名？
    members: list = field(default_factory=list)
    # 哪一个签名被选作代表签名？
    representative: int = 0

# 利用签名之间的距离，按照设定的规则将签名组织成若干个超边（Hyperedge）。
class PrototypeHypergraph:
    def __init__(self, cluster_threshold=1.0, merge_threshold=1.0, split_threshold=0.5, weights=(1 / 3, 1 / 3, 1 / 3)):
        if not all(math.isfinite(value) for value in (cluster_threshold, merge_threshold, split_threshold)) or cluster_threshold < 0 or merge_threshold < 0 or not 0 <= split_threshold <= 1:
            raise ValueError("Invalid hypergraph thresholds")
        if len(weights) != 3 or any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("weights must contain three finite nonnegative values with positive sum")
        # 控制初次聚类时的距离阈值。
        self.cluster_threshold = cluster_threshold
        # 合并阈值：控制新签名能否加入已有超边
        self.merge_threshold = merge_threshold
        # 控制增量更新时，何时触发超边拆分判断。
        self.split_threshold = split_threshold
        # 控制三个视图的距离对综合距离的贡献
        self.weights = weights
        self.edges = []

    # 计算所有成员签名之间的距离，形成一个距离矩阵
    def _matrix(self, members):
        distances = np.zeros((len(members), len(members)))
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                distances[first, second] = distances[second, first] = block_distance(
                    members[first], members[second], self.weights,
                )
        return distances

    #找到新的超边中的代表签名：计算这些签名两两之间的距离。
    # 找出与其他签名距离总和最小的成员，把它作为代表签名，创建一个 Hyperedge 对象。 
    def _edge(self, members):
        distances = self._matrix(members)
        return Hyperedge(members, int(np.argmin(distances.sum(axis=1))))

    # 根据指定的距离阈值，把候选超边中的成员重新分组。
    def _cluster(self, members, threshold):
        if len(members) <= 1:
            return [self._edge(members)] if members else []
        
        distances = self._matrix(members)
        # squareform(distances)：把矩阵转换成一维距离数组。
        # squareform 提取 / 构造的是 **k=1 的上三角**，不包含对角线，和 `np.triu_indices(n, k=1)` 对应。
        labels = fcluster(linkage(squareform(distances), method="complete"), threshold, criterion="distance")
        # 根据前面得到的聚类标签 labels，把属于同一簇的 HoSig 签名找出来，为每个簇创建一条超边，最后返回所有新超边。
        # np.flatnonzero() 的作用是：找到数组中非零元素的位置，并返回这些位置的索引。
        return [self._edge([members[index] for index in np.flatnonzero(labels == label)]) for label in np.unique(labels)]

    def fit(self, signatures):
        self.edges = self._cluster(list(signatures), self.cluster_threshold)
        return self

    # 这条候选超边内部的所有签名，整体上到底有多相似
    def cohesion(self, edge):
        # 代码约定将这种情况视为凝聚度最高，避免单成员超边因为无法计算两两距离而出现错误。
        if len(edge.members) < 2:
            return 1.0
        # 计算所有成员的距离矩阵
        distances = self._matrix(edge.members)
        return float(np.exp(-distances).mean())

    def add(self, signature):
        # 如果当前超图中还没有任何超边，就把新来的签名单独放进一条新超边，然后结束这次处理。
        if not self.edges:
            self.edges.append(Hyperedge([signature]))
            return
        # 计算新签名与每一条已有超边的代表签名之间的距离。
        distances = []
        for edge in self.edges:
            representative_signature = edge.members[edge.representative]
            distance = block_distance(
                signature,
                representative_signature,
                self.weights
            )
            distances.append(distance)
        # np.argmin(distances)，找出最小距离所在在索引
        closest = int(np.argmin(distances))
        # 超过合并阈值：新建一条超边：
        if distances[closest] > self.merge_threshold:
            self.edges.append(Hyperedge([signature]))
            return
        # 把新签名和最近超边中的原有签名放在一起，构造一个候选超边，并重新选出这个超边的代表签名。
        edge = self._edge(self.edges[closest].members + [signature])
        # 检查候选超边内部的成员是否足够相似。如果相似程度不够，就重新聚类，把它拆成几条更合理的超边。
        if self.cohesion(edge) < self.split_threshold:
            # -np.log(max(self.split_threshold, 1e-12))把相似度转换成距离
            # max(self.split_threshold, 1e-12))保证传入 np.log() 的数至少是 0.000000000001。
            threshold = min(self.cluster_threshold, -np.log(max(self.split_threshold, 1e-12)))
            # Python 的列表切片赋值允许用多个元素替换原来选中的一个元素。
            self.edges[closest:closest + 1] = self._cluster(edge.members, threshold)
        else:
            self.edges[closest] = edge

    def to_dict(self):
        return {
            "cluster_threshold": self.cluster_threshold, "merge_threshold": self.merge_threshold,
            "split_threshold": self.split_threshold, "weights": list(self.weights),
            "hyperedges": [
                {
                    "members": [member.block_id for member in edge.members],
                    "representative": edge.members[edge.representative].block_id,
                    "cohesion": self.cohesion(edge),
                } for edge in self.edges
            ],
        }

# 根据指定的距离阈值，把候选超边中的成员重新分组。
def _cluster(self, members, threshold):
    members = list(members)

    # 如果没有成员，就返回空列表
    if not members:
        return []

    # 开始时，每个签名单独作为一个簇
    clusters = [[member] for member in members]

'''    # 不断尝试合并簇，直到没有满足条件的合并方案
    while len(clusters) > 1:
        # 用来记录当前这一轮中，最值得合并的两个簇。
        best_pair = None
        best_score = float("inf")
        # 枚举所有可能合并的簇对
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # 假设把第 i 个簇和第 j 个簇合并
                candidate_members = clusters[i] + clusters[j]
                # 计算候选簇中所有成员之间的距离
                distances = self._matrix(candidate_members)
                # 选择与其他成员距离总和最小的签名作为代表
                representative = int(
                    np.argmin(distances.sum(axis=1))
                )
                # 计算所有成员到代表签名的距离
                representative_distances = distances[:, representative]
                # 取出成员到代表签名的最大距离
                max_distance = float(
                    np.max(representative_distances)
                )
                # 必须满足论文公式（11）
                if max_distance > threshold:
                    continue
                # 在符合约束的候选合并中，
                # 优先选择代表距离最大值最小的方案
                if max_distance < best_score:
                    best_score = max_distance
                    best_pair = (i, j)
        # 如果不存在满足约束的合并，就停止
        if best_pair is None:
            break
        # 将选中的两个簇合并
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        # 删除已经合并的第 j 个簇
        del clusters[j]
    # 将每个最终簇转换成一条超边
    return [
        self._edge(group)
        for group in clusters
    ]
'''