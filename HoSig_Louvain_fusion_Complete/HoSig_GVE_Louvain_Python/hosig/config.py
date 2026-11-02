from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Config:
    seed: int = 42
    #编码维度（论文元数据）
    embedding_dim: int = 256
    top_k: int = 20
    # 控制短时间尺度衰减和长时间尺度衰减各占多大权重。
    alpha: float = 0.5
    # RFF（Random Fourier Features，随机傅里叶特征）的输出维度默认为 512。
    rff_dim: int = 512
    # 不是用来决定哪些事件应该分到一组的，而是用来记录已经分好的一组事件，在时间上有什么特点。
    time_bins: int = 64
    left_window: int = 64
    right_window: int = 64
    # 分割参数
    segmentation_beta: float = 0.8
    # 表示算法为事件中的文本内容准备了 2048 个位置，用来把原本计算机不容易直接处理的文本转换成数字。
    vocabulary_size: int = 2048
    # trees=64 表示建立 64 棵随机划分树，通过统计两条安全事件在多少棵树中落入同一个叶子节点，估计它们的内容相似度。
    trees: int = 64
    # 在建立随机划分树时，不断把安全事件分成更小的组，直到每个最终小组最多只有16条事件。（最后的叶子节点最多容纳16个事件）
    # 随机划分树中，一个叶子节点最多容纳多少条事件
    leaf_size: int = 16
    '''论文中这三个参数都没有给出默认值'''
    #时间衰减系数的最低保留值 
    decay_floor: float = 0.05
    # 短时间尺度的半衰期
    short_half_life: float | None = None
    # 长时间尺度的半衰期
    long_half_life: float | None = None
    # 两条安全事件有多大的差别时，算法才会觉得它们“不太相似”。
    # 控制这个转换过程对原始特征差异有多敏感的参数。
    rff_gamma: float | None = None
    # eta 决定算法在寻找分界点时，要多重视事件之间的“图连接关系”。
    eta: float = 1.0
    # 算法在正常选择分界点时，要求切分出来的每个事件块至少包含8条安全事件。
    min_block_size: int = 8
    # 程序默认每次取512条安全事件作为一个微批次（micro-batch），交给 HoSig-Align I 算法处理。
    # 最后不足的时候也可以当作一个微批次
    batch_size: int = 512
    # 每一行代表一种不同的哈希映射方式，每一列代表一个可以存储计数的格子。每个特征在每一行都会通过哈希计算找到对应的格子，并将该格子的计数加1。
    # 把同源块中大量的上下文特征统计信息，压缩成大小固定、可以保存和传递的轻量化摘要。
    cms_width: int = 128
    # CMS的行数（cms_depth）：表示程序用多少种不同的哈希方式，对同一个块的特征进行重复统计。
    cms_depth: int = 4
    # 所有特征中，一共有多少种不同的特征？
    hll_precision: int = 8
    # 就是给每个同源块挑选最多10个最有区分能力的代表性特征，而不是随机挑10个，也不是简单挑出现次数最多的10个。
    top_features: int = 10
    # 上下文特征进行哈希计算时，统一加入 "hosig-reproduction-public-demo-salt" 这段固定文字。
    # 相同特征 + 相同盐值 + 相同行号 + 相同哈希规则，得到的位置一定相同。
    context_salt: str = "hosig-reproduction-public-demo-salt"

    #  __post_init__() 函数，负责检查这些参数的取值是否合法。
    def __post_init__(self):
        # 检查这些值是不是整数
        for name in (
            "embedding_dim", "vocabulary_size", "trees", "leaf_size", "top_k",
            "rff_dim", "time_bins", "left_window", "right_window", "min_block_size",
            "batch_size", "cms_width", "cms_depth", "top_features",
        ):
            # getattr() 是 Python 的内置函数，作用是：
            # 根据属性名称，获取对象中对应的属性值。
            value = getattr(self, name)
            # isinstance() 用来判断一个值是否属于某种类型。
            # 因为在 Python 中，bool 是 int 的子类。
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("alpha", "decay_floor"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("eta", "segmentation_beta"):
            # math.isfinite() 是Python的 math 模块提供的函数。
            # math.isfinite()：检查是不是有限数
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("short_half_life", "long_half_life", "rff_gamma"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.hll_precision, int) or not 4 <= self.hll_precision <= 16:
            raise ValueError("hll_precision must be an integer in [4, 16]")
        if not self.context_salt:
            raise ValueError("context_salt cannot be empty")

    # 对象中的所有配置参数转换成一个 Python 字典（dict），然后返回这个字典。
    def to_dict(self):
        return asdict(self)

    @classmethod
    # 从一个 JSON 配置文件中读取参数，并利用这些参数创建一个 Config 对象。
    # cls：代表这个类本身
    # 因为 load（） 的任务本来就是读取参数并创建对象。
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            # 这里得到的是一个字典
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        # set() 可以把数据转换成集合。如果被转换的对象是字典，那么转换后的结果就是字典中的键
        unknown = set(values) - set(cls.__dataclass_fields__)
        # 检查有没有写错参数名称，__dataclass_fields__保存了 @dataclass 定义的字段信息。
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)
    @property
    # 根据一组与特征表示有关的配置参数，生成一个固定长度的标识符（ID），用来标记当前使用的特征空间配置。
    def feature_space_id(self):
        names = (
            "seed", "embedding_dim", "vocabulary_size", "rff_dim", "rff_gamma",
            "time_bins", "cms_width", "cms_depth", "hll_precision", "context_salt",
        )
        values = {name: getattr(self, name) for name in names}
        values["encoding_version"] = "hashed-bow-v1"
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        '''
        # 第一步：把参数字典转换成JSON字符串
                json_text = json.dumps(values, sort_keys=True)

        # 第二步：把字符串转换成字节数据
                byte_data = json_text.encode()

        # 第三步：计算SHA-256哈希
                hash_result = hashlib.sha256(byte_data)

        # 第四步：转换成十六进制字符串
                feature_id = hash_result.hexdigest()

        # 第五步：返回生成的ID
                return feature_ids
        
        '''