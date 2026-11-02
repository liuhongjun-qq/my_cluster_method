# HoSig 内容编码改进版：BOW + 数值特征 + 语义嵌入

本版本在原项目中新增实验性编码器，不删除原始 HoSig 编码器，也不修改后续随机分区森林、时间衰减建图、互为 Top-k 建图和 Louvain 社区发现模块。

## 1. 新方法做什么

新编码器位于：`hosig/_02encoding_hybrid.py`

一条事件仍然来自 `_01data.py` 中的 `Event` 对象。CSV 中有效字段会放入 `Event.context`，同时以 `字段名=字段值` 的形式拼接到 `Event.payload`。

新编码器分别处理三个分支：

1. **原始 BOW/IP 分支**：保留原来的 IP 前缀哈希、Payload Token 哈希和词频统计。
2. **数值分支**：从 `Event.context` 读取数值字段，例如 `duration`、`orig_bytes`、`resp_bytes`；先做 signed-log1p，再用中位数和 IQR 做 Robust Scaling。这样 `duration=1.25` 和 `duration=1.30` 会被表示为同一个数值字段上的两个接近数值，而不是两个互不相关的 Token。
3. **语义分支**：对类别字段构造带字段上下文的文本，例如 `network security field service value http`，再使用预训练 SentenceTransformer 得到语义向量。不同值是否语义接近由预训练模型决定；可选的 JSON 别名文件可将经过数据领域核验的同义值映射到同一个规范描述。

三个分支按权重拼接，再用固定随机矩阵投影到 256 维，最后使用 GELU 和逐事件归一化。输出仍然是 `N × 256` 的事件表示，后续代码接口不变。

连接状态代码（如 `SF`）、Zeek history、端口号等机器代码/标识不交给通用自然语言模型猜测语义；它们仍由原始 BOW 分支保留。像 `proto=TCP/UDP`、`service=http/dns` 这类文本类别值才进入语义分支。若 `Protocol` 在某个数据集中被编码为纯数字（例如 6、17），程序会把它作为精确类别保留，而不会把数字交给语言模型解释。

## 2. 安装语义模型依赖

在 Windows PowerShell 中进入项目根目录：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hybrid.txt
```

默认语义模型为 `sentence-transformers/all-MiniLM-L6-v2`。第一次运行需要下载模型文件，因此第一次运行需要能够访问模型托管站点。若已经下载到本地，可用 `--semantic-model` 指定本地模型目录。

## 3. 运行新的融合编码方法

按照此前 IoTID20 预处理输出目录的示例：

```powershell
.\.venv\Scripts\python.exe run_fusion.py --input ".\data\ready\iotid20_ready.csv" --batch-size 256 --encoder hybrid --output ".\outputs\iotid20_hybrid_256"
```

如果你的 CSV 实际路径不同，只需要修改 `--input`。

建议第一次先做小规模检查：

```powershell
.\.venv\Scripts\python.exe run_fusion.py --input ".\data\ready\iotid20_ready.csv" --limit 512 --batch-size 256 --encoder hybrid --output ".\outputs\iotid20_hybrid_smoke"
```

## 4. 运行原始方法作为对照

```powershell
.\.venv\Scripts\python.exe run_fusion.py --input ".\data\ready\iotid20_ready.csv" --limit 512 --batch-size 256 --encoder original --output ".\outputs\iotid20_original_smoke"
```

原始方法仍然可用。CLI 默认值也保持为 `original`，避免旧命令无意中切换编码器。要运行新方法必须明确写 `--encoder hybrid`。

## 5. 分支权重

默认三个分支权重相同：BOW、数值、语义各为 1/3。程序会将输入权重归一化，使其总和为 1。

例如：

```powershell
.\.venv\Scripts\python.exe run_fusion.py --input ".\data\ready\iotid20_ready.csv" --batch-size 256 --encoder hybrid --bow-weight 0.5 --numeric-weight 0.25 --semantic-weight 0.25 --output ".\outputs\iotid20_hybrid_weighted"
```

初次实验建议先使用默认权重，不要同时修改多个参数。

## 6. 可选语义别名

`semantic_aliases.example.json` 是可选示例。它演示如何把某个字段下、经过领域核验的多个值映射到同一个规范描述。示例中把 `service` 字段的 `http` 和 `www` 映射到 `web service`：

```json
{
  "service": {
    "http": "web service",
    "www": "web service"
  }
}
```

只有当它们在你的数据定义中确实表达相同含义时才应使用该映射。不要把仅仅相关、但含义不同的类别强行合并。运行时通过 `--semantic-aliases semantic_aliases.example.json` 启用。

## 7. 输出与检查

输出目录仍然包含：

- `report.json`
- `evaluation_batches.csv`
- `event_communities.csv`
- `local/batch_XXXX/graph.npz`
- `local/batch_XXXX/representations.npz`
- `local/batch_XXXX/assignments.json`

`report.json` 新增：

- `encoder_type`
- `encoder`：编码器名称、语义模型、识别出的数值字段、语义字段、分支权重和输入/输出维度
- `encoder_fit_seconds`：编码器拟合耗时（单独记录）

## 8. 重要实验说明

- 新方法是对原始 HoSig 内容编码模块的实验性改进，不是 HoSig 原论文原有算法。
- 数值字段统计量在整个输入 CSV 的第一遍扫描中估计，第二遍才按微批次处理；第一遍不读取标签和时间戳。这是当前离线数据集实验的拟合方式。若要做严格在线流式实验，需要进一步改为仅使用历史训练窗口或预先固定的数值统计量。
- 通用预训练语言模型不保证能理解所有网络协议缩写或服务名称。特别是 `http`、`www` 等值，应该先检查语义相似度，必要时使用经过核验的领域别名映射。
- 高基数标识字段不会进入语义分支，避免将大量 ID 当作语义类别；原始 BOW 分支仍然保留原来的 Token 信息。
- 比较原始和新方法时，必须使用相同数据、批次大小、随机种子、森林参数、建图参数和评价设置。新方法只替换内容编码器，但额外的模型拟合耗时应单独报告。
