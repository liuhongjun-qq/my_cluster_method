#事件相似度：随机划分森林、计算共同叶相似度
import numpy as np

# RandomPartitionForest 就是通过多棵随机划分树，
# 将内容相似的事件尽可能划分到相同叶节点，再利用共同叶节点的比例估计事件之间的相似度。
class RandomPartitionForest:
    def __init__(self, trees=64, leaf_size=16, seed=42):
        if trees < 1 or leaf_size < 1:
            raise ValueError("trees and leaf_size must be positive")
        self.trees = trees
        self.leaf_size = leaf_size
        self.seed = seed

    def fit(self, embeddings):
        embeddings = np.asarray(embeddings)
        
        if embeddings.ndim != 2 or embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be a finite matrix with at least one feature")
        generator = np.random.default_rng(self.seed)
        n_events = len(embeddings)
        # 提前创建一个二维 NumPy 数组，用来记录每条网络安全事件在每棵随机划分树中属于哪个叶节点。
        self.leaf_ids_ = np.empty((n_events, self.trees), dtype=np.int32)
        # 保存每棵树的每个叶子节点有哪些事件
        self.members_ = []

        for tree_index in range(self.trees):
            # 创建一个空列表，用来保存当前这棵随机划分树最终形成的所有叶节点。
            # leaves 记录了当前这棵树的每个叶节点里面有哪些事件。
            leaves = []
            # 存储未划分的事件
            pending = [np.arange(n_events)] if n_events else []
            # 把一批网络安全事件不断分成更小的组，直到每组的事件数量不超过 leaf_size，或者已经无法继续划分。
            while pending:
                indices = pending.pop()
                if len(indices) > self.leaf_size:
                    # 根据事件编号，从 embeddings 中取出当前需要处理的那些事件的特征向量。
                    values = embeddings[indices]
                    minima = values.min(axis=0)
                    maxima = values.max(axis=0)
                    # 找出数组中值不为0（对于布尔数组就是 True）的位置编号。
                    # 保存的是当前这批事件中所有可以尝试用于随机划分的特征编号。
                    dimensions = np.flatnonzero(maxima > minima)
                    '''
                    随机选择一个能够区分事件的特征。
                    在这个特征的取值范围内随机选择一个划分阈值。
                    根据阈值把事件分成左边和右边两组。
                    将划分后的两组事件放回 pending，等待后续检查。
                    '''
                    if len(dimensions):
                        dimension = int(generator.choice(dimensions))
                        #随机选择一个划分阈值
                        threshold = generator.uniform(minima[dimension], maxima[dimension])
                        # 防止随机生成的划分阈值等于特征的最大值，从而导致右边没有事件。
                        threshold = min(threshold, np.nextafter(float(maxima[dimension]), -np.inf))
                        coordinate = values[:, dimension].astype(float)

                        left = indices[coordinate <= threshold]
                        right = indices[coordinate > threshold]
                        if len(left) and len(right):
                            pending.extend((right, left))
                            continue
                self.leaf_ids_[indices, tree_index] = len(leaves)
                leaves.append(indices)
            self.members_.append(leaves)
        return self

    # 查看它在每棵随机树中和哪些事件落在同一个叶子里，然后统计这些事件共同出现了多少次，最后计算内容相似度。
    def neighbors(self, index):
        members = []
        for tree in range(self.trees):
            # 找到事件在这棵树的哪些节点
            leaf_id = self.leaf_ids_[index, tree]
            # 找到这个节点有哪些事件编号
            leaf_members = self.members_[tree][leaf_id]
            members.append(leaf_members)
        # candidates = [0, 1, 2]
        #  counts = [3, 2, 2]
        candidates, counts = np.unique(np.concatenate(members), return_counts=True)
        keep = candidates != index
        return candidates[keep], counts[keep] / self.trees

    def pair_similarity(self, first, second):
        return float(np.mean(self.leaf_ids_[first] == self.leaf_ids_[second]))
