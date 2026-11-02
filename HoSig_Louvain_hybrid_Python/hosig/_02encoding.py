#内容编码：把事件内容转换为数值表示
import hashlib
import ipaddress
import re

import numpy as np
from scipy.special import ndtr

from .config import Config


def stable_hash(text, salt="", digest_size=8):
    digest = hashlib.blake2b(
        str(text).encode("utf-8"), key=salt.encode("utf-8")[:64], digest_size=digest_size,
    ).digest()
    return int.from_bytes(digest, "big")


def payload_tokens(payload):
    return re.findall(r"[\w./:@=+-]+", payload.lower())


class ContentEncoder:
    def __init__(self, config: Config):
        self.config = config
        # 创建一个随机数生成器
        generator = np.random.default_rng(config.seed)
        # 编码一条网络事件的维度
        dimension = config.vocabulary_size + 4
        # 生成一个由随机数字组成的矩阵，并保存起来，后面用它把网络安全事件的原始特征转换成统一长度的数字向量
        # 就是让这个生成器按照正态分布生成随机数字。
        # 在 ContentEncoder 中，使用 1 / np.sqrt(dimension)，
        # 是为了让随机投影矩阵中的权重随着输入维度增大而适当减小，控制多个特征相加后的数值规模。
        self.projection = generator.normal(
            0, 1 / np.sqrt(dimension), (dimension, config.embedding_dim),
        ).astype(np.float32)

    def _ip_features(self, value):
        # 将一个IP地址转换成两个数值特征，用来表示该IP所属的不同大小的网络范围。
        address = ipaddress.ip_address(value)
        if address.version == 4:
            prefixes = (24, 16)
        else:
             prefixes = (64, 48)
        result = []

        for prefix in prefixes:
            network = ipaddress.ip_network(f"{address}/{prefix}", strict=False)
            network_str = str(network)
             # 输入一个网络地址字符串，输出一个整数。使用长度为4字节的哈希摘要。
            hash_value = stable_hash(network_str, digest_size=4)
            # 第二行：将哈希整数缩放到0～1
            feature_value = hash_value / (2**32 - 1)

            result.append(feature_value)

        return result

    def transform(self, events):
        if not events:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)
        # 创建一个全部初始化为0的二维特征矩阵，用来存放当前微批次中每条网络安全事件的IP特征和Payload文本特征。
        features = np.zeros((len(events), self.config.vocabulary_size + 4), dtype=np.float32)
        for index, event in enumerate(events):
            # ip特征
            features[index, :4] = self._ip_features(event.src_ip) + self._ip_features(event.dst_ip)
            # 把当前网络安全事件的 Payload（载荷）拆分成多个 Token（词元），
            # 再将每个 Token 映射到特征矩阵中的某一列，并统计它出现的次数。
            for token in payload_tokens(event.payload):
                slot = 4 + stable_hash(token) % self.config.vocabulary_size
                features[index, slot] += 1
        projected = features @ self.projection
        # **对 projected 里面每一个数，查标准正态分布表，得到 0~1 之间的累积概率值，用来做特征归一化**。
        activated = projected * ndtr(projected)
        # 每条事件单独计算自己的平均值。去中心化处理：让每条事件的特征均值变成0
        centered = activated - activated.mean(axis=1, keepdims=True)
        # 计算每条事件的特征方差。
        variance = activated.var(axis=1, keepdims=True)
        # 对方差加一个很小的数，再开平方。
        denominator = np.sqrt(variance + 1e-5)
        # 用中心化后的特征除以计算结果
        normalized = centered / denominator
        # 转换为 float32，返回最终特征矩阵
        # 它让每条事件的特征均值接近0、方差接近1，从而减少不同事件特征数值尺度的差异。
        result = normalized.astype(np.float32)
        return result

class RandomFourierFeatures:
    def __init__(self, config: Config):
        self.config = config
        gamma = config.rff_gamma if config.rff_gamma is not None else 1 / config.embedding_dim
        generator = np.random.default_rng(config.seed + 1)
        self.weights = generator.normal(
            # 照正态分布生成一个随机权重矩阵，并保存到 self.weights 中。
            0, np.sqrt(2 * gamma), (config.embedding_dim, config.rff_dim),
        ).astype(np.float32)
        # 随机生成一组相位参数，并将它们保存到 self.phase 中，供后续RFF特征转换使用。
        # RFF后面会使用余弦函数，而余弦函数每经过 \(2\pi\) 就会重复一次。
        self.phase = generator.uniform(0, 2 * np.pi, config.rff_dim).astype(np.float32)

    def transform(self, embeddings):
        return (np.sqrt(2 / self.config.rff_dim) * np.cos(embeddings @ self.weights + self.phase)).astype(np.float32)
