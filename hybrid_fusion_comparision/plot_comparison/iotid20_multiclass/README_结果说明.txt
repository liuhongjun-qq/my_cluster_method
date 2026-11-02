分析范围：多类别批次
A 方法：原始HoSig编码
B 方法：Hybrid均衡编码(1/3)
本次分析批次数：590
共同批次数（筛选前）：1223
A 报告批次数：1223
B 报告批次数：1223

01_metric_means.png：各自有效均值与配对均值；每项指标单独坐标轴。
02_valid_batch_counts.png：各指标有效批次数。
03_batch_metric_lines.png：逐批指标变化；CH、Silhouette 默认只画共同有效值。
04_paired_wins_ties.png：同批配对数值高低统计，不代表统计显著性。
batch_metrics.csv：逐批原始指标及有效性标记。
metric_summary.csv/json：均值、有效批次数、配对均值和胜负统计。

multiclass：只分析真实类别数至少为2的共同批次。
all：分析全部共同批次，包括单一真实类别批次。
五项指标的胜负图按数值越高越好统计；胜负次数不代表统计显著性。
