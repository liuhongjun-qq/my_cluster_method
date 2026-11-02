from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse

from .config import Config
from ._01data import micro_batches
from ._02encoding import ContentEncoder, RandomFourierFeatures
from ._03forest import RandomPartitionForest
from ._04graph import fiedler_order, mutual_topk_graph
from ._05segmentation import Segmentation, contrastive_segment
from ._07signatures import HoSig, build_signatures


@dataclass
class BatchResult:
    # 内容特征向量
    embeddings: np.ndarray
    # 稀疏关系图
    graph: sparse.csr_matrix
    # 排序结果
    order: np.ndarray
    # 分割结果：同源块划分、块边界等信息
    segmentation: Segmentation
    # HoSig签名：为各个同源块生成的签名列表
    signatures: list[HoSig]
    # 诊断信息：事件数、块数、边数、运行时间、内存占用等统计
    diagnostics: dict

    @property
    def labels(self):
        return self.segmentation.labels


# 程序专门创建了一个负责执行 HoSig-Align I 算法的工具。
class HoSigAlign:
    # 初始化模型:准备处理工具
    def __init__(self, config=None):
        self.config = config or Config()
        self.encoder = ContentEncoder(self.config)
        self.rff = RandomFourierFeatures(self.config)

    #处理一批网络安全事件	正式分析一批数据 
    # 接收一个微批次的网络安全事件，
    # 执行 HoSig-Align I 的核心算法，发现其中的同源块，并为每个同源块生成 HoSig 签名。
    def process_batch(self, events, batch_id="0"):
        if not events:
            raise ValueError("Cannot process an empty batch")
        if len(events) > self.config.batch_size:
            raise ValueError("Batch exceeds batch_size; use process_stream or increase the configured budget")
        start = perf_counter()
        # 创建一个空字典，用来记录当前微批次中各个算法阶段的运行耗时。
        timings = {}

        # 把当前微批次中的网络安全事件转换成数值特征向量，并将转换结果保存在 embeddings 中。
        embeddings = self.encoder.transform(events)
        timings["encoding"] = perf_counter() - start
        # 返回一个二维数组:

        # 随机森林
        stage_start = perf_counter()
        forest = RandomPartitionForest(
            self.config.trees, self.config.leaf_size, self.config.seed,
        ).fit(embeddings)
        timings["forest"] = perf_counter() - stage_start

        # 构图（双向top_k）
        stage_start = perf_counter()
        timestamps = np.array([event.timestamp for event in events])
        graph, decay = mutual_topk_graph(forest, timestamps, self.config)
        '''第一个结果 graph： 构建完成的双向 Top-k 稀疏图。
        第二个结果 decay： 一个字典，记录构图使用的短半衰期和长半衰期，'''
        timings["graph"] = perf_counter() - stage_start

        # 谱排序
        stage_start = perf_counter()
        order, components, warnings = fiedler_order(graph, self.config.seed)
        timings["spectral"] = perf_counter() - stage_start

        # rff向量的转化
        # 变化点检测与同源块分割阶段
        stage_start = perf_counter()
        rff = self.rff.transform(embeddings)
        '''
        graph    双向 Top-k 构图    提供事件之间的连接关系
        order    Fiedler 谱排序    提供事件重新排列后的顺序
        components    连通分量检测    提供每个事件所属的连通分量
        rff    RFF 转换    提供转换后的事件特征
        self.config    配置文件    提供分割算法需要的参数
        
        '''
        segmentation = contrastive_segment(graph, order, components, rff, self.config)
        timings["segmentation"] = perf_counter() - stage_start

        # 它属于 HoSig-Align I 流程中的 Step 9：HoSig 签名生成
        stage_start = perf_counter()
        signatures = build_signatures(events, graph, segmentation.blocks, rff, self.config, batch_id)
        timings["signatures"] = perf_counter() - stage_start

        timings["total"] = perf_counter() - start

        # 计算稀疏图占用的内存
        graph_bytes = graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes
        diagnostics = {
            "n_events": len(events), "n_blocks": len(signatures), 
            # 图的边数
            "n_edges": graph.nnz // 2,
            # 每个事件平均与多少个其他事件相连
            "mean_degree": float(graph.nnz / len(events)),
            # 连通分量数量
            "n_components": int(len(np.unique(components))),
            # 分割惩罚值
            "segmentation_penalty": segmentation.penalty,
            # 统计 segmentation.costs 中有多少个内部位置的值超过惩罚值
            "positive_gain_candidates": int(np.count_nonzero(segmentation.costs[1:-1] > segmentation.penalty)),
            "warnings": warnings, 
            # 记录构图时使用的时间衰减尺度
            "half_lives": decay,
            # 记录各个算法阶段花费的时间，单位为秒
            "timings_seconds": timings,
            # 事件编码矩阵占用的字节数
            "representation_bytes": embeddings.nbytes,
            # 稀疏图主要存储数组占用的字节数
            "sparse_graph_bytes": int(graph_bytes),
            # 签名数值数据内存
            "signature_numeric_bytes": sum(signature.numeric_bytes for signature in signatures),
        }
        #  Segmentation(labels, blocks, boundaries, costs, float(penalty))
        return BatchResult(embeddings, graph, order, segmentation, signatures, diagnostics)
    # 将连续事件分成多个批次处理:负责安排一批接一批地处理
    def process_stream(self, events):
        for batch_id, batch in enumerate(micro_batches(events, self.config.batch_size)):
            yield self.process_batch(batch, str(batch_id))



