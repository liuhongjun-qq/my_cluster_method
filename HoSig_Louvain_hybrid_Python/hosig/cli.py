
import argparse 
from dataclasses import replace
import hashlib
import json

from pathlib import Path

import sys
import numpy as np
from scipy import sparse

from .alignment import PrototypeHypergraph
from .config import Config
from ._01data import generate_synthetic, micro_batches, read_events, write_events
from .experiments import benchmark
from .iot23 import convert_iot23
from .metrics import cluster_metrics
from .pipeline import HoSigAlign
from .reporting import environment_info, plot_benchmark, plot_boundaries, write_benchmark_csv, write_json
from ._07signatures import load_signatures

# 生成参数的配置文件
def _config(args):
    # 判断是否指定配置文件
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    overrides = {
        name: getattr(args, name)
        for name in ("batch_size", "top_k", "seed")
        if getattr(args, name, None) is not None
    }
    # replace() 并不是直接修改原来的对象，而是创建一个新对象，并在新对象中替换指定参数。
    # replace是replace 不是普通的Python内置函数，它来自 dataclasses 模块。
    return replace(config, **overrides)


def _input_metadata(args):
    with open(args.input, "rb") as handle:
        # 计算文件的哈希值
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    #创建一个字典，把这次实验使用的数据文件的相关信息记录下来。
    metadata = {
        # path.resolve表示把路径转换成完整路径。
        "path": str(Path(args.input).resolve()),
        "format": args.format,
        #文件的识别字符串
        "sha256": checksum,
        # 是否指定模拟时间间隔：当数据没有真实时间时，用来设置相邻记录之间模拟相隔多少秒的参数。
        "proxy_interval": args.proxy_interval,
        # 是否限制读取数量
        "limit": args.limit,
        # “如果文件明确是 network-csv，程序先把时间相关结论标记为不可直接采用；
        # 如果不是，就暂时标记为 True，但这个 True 不代表已经验证时间真实。”
        "temporal_claims_valid": args.format != "network-csv",
        "warning": "Synthetic row-order timestamps and structured-token payload proxy; not paper data" if args.format == "network-csv" else None,
    }
    # 找到说明文件
    provenance_path = Path(args.input).with_suffix(".metadata.json")
    if provenance_path.exists():
        with provenance_path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        if provenance.get("output_sha256") != checksum:
            raise ValueError("Input does not match its provenance metadata SHA256")
        metadata["provenance"] = provenance
        metadata["warning"] = provenance.get("warning")
        metadata["timestamps_real"] = provenance.get("timestamps_real")
        metadata["payload_real"] = provenance.get("payload_real")
        metadata["temporal_claims_valid"] = False if provenance.get("alert_arrival_times_available") is False else metadata["temporal_claims_valid"]
    return metadata

# 是整个HoSig-Align I复现程序的数据处理入口函数。
# 它本身并没有把内容编码、随机分区森林、Fiedler向量、变化点检测等算法全部写在里面，而是负责组织这些步骤，调用其他函数完成实际工作。
# 函数只负责本地批次处理和签名导出
def run_stream(args, config):
    # 调用 _input_metadata() 函数，收集并检查输入数据文件的相关信息，然后把这些信息保存到 input_metadata 变量中。
    # 输入元数据
    input_metadata = _input_metadata(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # 创建一个 HoSig-Align I 处理工具，把它保存到 model 中，等后面网络安全事件到来时，再让它开始处理。”
    model = HoSigAlign(config)
    # 报告集合:创建一个空列表，准备存放后续每个微批次的处理报告。
    reports = []
    # read_events() 是一个函数，负责根据指定的输入路径、数据格式等参数读取事件,返回一个可处理数据对象的迭代器
    events = read_events(args.input, args.format, args.proxy_interval, args.limit)
    # (output / "signatures.jsonl")用于保存生成的 HoSig 签名。
    with (output / "signatures.jsonl").open("w", encoding="utf-8") as exported:
        # 把读取到的网络安全事件按照指定大小划分成一个个微批次（Micro-batch），
        # 然后逐批处理，并为每个批次生成编号。
        for batch_id, batch in enumerate(micro_batches(events, config.batch_size)):
            # 把当前微批次中的网络安全事件交给 HoSig-Align I 模型处理，并将处理结果保存到 result 变量中。
            result = model.process_batch(batch, f"{args.site_id}:{batch_id}")
            # Segmentation(labels, blocks, boundaries, costs, float(penalty))
            # return BatchResult(embeddings, graph, order, segmentation, signatures, diagnostics)

            # 遍历当前批次生成的所有 HoSig 签名，每次取出一个签名进行处理。
            for signature in result.signatures:
                exported.write(json.dumps(signature.to_dict(), allow_nan=False) + "\n")
            local = output / "local" / f"batch_{batch_id:04d}"
            # 创建当前批次的本地结果文件夹
            local.mkdir(parents=True, exist_ok=True)
            write_json(local / "assignments.json", {
                # 它会遍历当前批次的每个事件，提取事件编号。
                "event_ids": [event.event_id for event in batch], 
                # result.labels 保存算法给事件分配的标签
                "labels": result.labels.tolist(),
                # evaluation_truth：真实标签
                "evaluation_truth": [event.label for event in batch], 
                # order：事件排序结果
                "order": result.order.tolist(),
                # boundaries：同源块分割边界
                "boundaries": result.segmentation.boundaries, 
                # costs：分割代价或候选得分
                "costs": result.segmentation.costs.tolist(),
            })

            
            # 如果用户在运行 run 命令时开启了 --save-arrays，就执行下面保存文件的代码。
            if args.save_arrays:
                # graph.npz 保存的是 result.graph，也就是算法处理当前批次后得到的稀疏图。
                sparse.save_npz(local / "graph.npz", result.graph)
                # 这个文件保存的是当前微批次中每条网络事件经过内容编码器计算后得到的数值向量，
                # 而不是原始网络日志、数据包内容或者已经划分好的同源块。
                np.savez_compressed(local / "representations.npz", embeddings=result.embeddings)
            entities = np.array([signature.prototype for signature in result.signatures])

            # 评价划分质量
            scores = cluster_metrics(result.embeddings, result.labels, result.graph, entities, [event.label for event in batch], config.seed)
            # 写入报告
            reports.append({"batch_id": batch_id, "diagnostics": result.diagnostics, "metrics": scores})
            if batch_id == 0:
                # 
                plot_boundaries(result, output / "first_batch_boundaries.png")
            print(f"batch={batch_id} events={len(batch)} blocks={len(result.signatures)} degree={result.diagnostics['mean_degree']:.2f}")
    if not reports:
        raise ValueError("Input contains no events")
    write_json(output / "report.json", {
        "reproduction_status": "independent_method_reproduction_not_paper_results",
        "config": config.to_dict(), "environment": environment_info(), "input": input_metadata,
        "total_events": sum(report["diagnostics"]["n_events"] for report in reports),
        "total_blocks": sum(report["diagnostics"]["n_blocks"] for report in reports),
        # 把 reports 列表中保存的所有微批次报告，放到最终的 report.json 文件的 "batches" 字段中。
        "batches": reports, 
        "signature_json_bytes": (output / "signatures.jsonl").stat().st_size,
    })
    print(f"Report: {output / 'report.json'}")


def _common_input(parser):
    # 它的作用是告诉 run：要处理的网络安全事件保存在哪个文件中。
    parser.add_argument("--input", required=True)
    # 应该按照什么格式读取输入文件。而是根据文件扩展名选择格式。
    parser.add_argument("--format", choices=("auto", "csv", "jsonl", "network-csv"), default="auto")
    # 它是你复现代码中用于处理某些缺少真实时间戳的网络日志的辅助参数。
    # 当某些输入数据没有真实时间戳时，按照人为设定的间隔生成代理时间戳。
    parser.add_argument("--proxy-interval", type=float)
    # 控制本次实验最多读取多少条网络安全事件。
    # 它控制读取事件的总数量，不是微批次大小，也不是同源块数量。
    # 不填写时没有额外的读取数量上限，程序会继续按读取规则处理输入数据。
    parser.add_argument("--limit", type=int)
    # 给 run 命令添加一个可选参数，让用户指定本次算法运行所使用的配置文件路径。
    # 不填写 --config：使用 Config() 中定义的默认配置。
    # 填写 --config：从指定文件加载配置。
    # 如果命令行另外指定了 --batch-size、--top-k 或 --seed：这些明确指定的值会覆盖配置文件中的对应设置。
    parser.add_argument("--config")
    # 微批次的大小
    parser.add_argument("--batch-size", type=int)
    # run --seed：控制算法怎样进行随机计算。（比如速记划分森林时候）
    parser.add_argument("--seed", type=int)
    # 它指定的是结果保存目录，程序会在这个目录下生成签名、报告、分块结果等文件。
    parser.add_argument("--output", required=True)
    # site-id 表示数据处理站点的标识，而不是 IP 地址，也不是事件数量。
    # 默认值是 "local"，表示本地站点。
    parser.add_argument("--site-id", default="local")


# 建立解析器：负责建立命令规则
def build_parser():
    parser = argparse.ArgumentParser(description="Independent HoSig-Align I reproduction")
    commands = parser.add_subparsers(dest="command", required=True)

    # 登记了各个命令和参数，是 generate、run、demo 等命令的具体定义。
    
    '''登记generate的功能：生成一份供测试使用的合成网络安全事件数据，并将它保存到指定文件中。'''
    generate = commands.add_parser("generate", help="Generate labeled synthetic telemetry for smoke testing")
    # 结果存在哪里
    generate.add_argument("--output", default="data/synthetic.jsonl")
    #生成多少条事件
    generate.add_argument("--events", type=int, default=768)
    # campaigns：先设置合成数据中有多少个模拟活动类别，再让算法自己进行分块，最后利用预先设置的标签评价算法结果。
    generate.add_argument("--campaigns", type=int, default=3)
    #随机种子，用来控制每一个产生的随机数据基本相同
    generate.add_argument("--seed", type=int, default=42)


    ''''
    这个子命令不是 HoSig-Align I 论文提出的算法步骤，而是复现项目为了方便管理配置而设计的功能。
    write-config 子命令：负责将代码中定义的默认配置保存成一个文件，方便查看、管理和修改这些参数。
    本身不是论文中的算法，但它生成的配置文件可能包含论文算法所使用的参数。
    '''
    '''generate 是生成测试数据，而 write-config 是生成参数配置文件。'''
    config = commands.add_parser("write-config", help="Write all explicit defaults")
    config.add_argument("--output", default="configs/paper.json")

    '''
    convert-iot23 不是 HoSig-Align I 论文提出的核心算法步骤，
    而是复现项目为了处理 IoT-23 数据集而设计的数据预处理命令。
    它的作用是：把 IoT-23 数据集中 Zeek 生成的网络连接日志，
    转换成 HoSig-Align I 程序能够读取和处理的事件数据格式。
    '''
    conversion = commands.add_parser("convert-iot23", help="Convert labeled Zeek logs to explicit metadata-proxy events")
    # 必须指定输入文件的位置。
    conversion.add_argument("--input", required=True)
    # 必须指定输入文件的位置。
    conversion.add_argument("--output", required=True)
    #指明数据的采集场景
    conversion.add_argument("--scenario")
    # 用来决定：转换 IoT-23 数据时，事件的标签要保留详细类别，还是简化成正常和恶意两大类。
    conversion.add_argument("--label-mode", choices=("detailed", "binary"), default="detailed")
    #timestamp：按照事件发生的时间先后排列。
    # source：按照事件在原始文件中出现的先后排列。    
    conversion.add_argument("--order", choices=("timestamp", "source"), default="timestamp")
    # --limit 控制的是数据转换阶段的记录数量。
    conversion.add_argument("--limit", type=int)



    '''
    run 是用来执行论文核心算法流程的命令入口。
    run 的作用就是读取已经准备好的网络安全事件数据，
    把数据分成一小批、一小批，然后执行 HoSig-Align I 的处理流程，发现同源事件块并生成相应的签名。
    '''
    # help:以大小受限的微批次方式处理网络安全遥测数据。
    run = commands.add_parser("run", help="Process telemetry in bounded micro-batches")
    # 请给 run 命令添加那一套已经定义好的通用参数。
    _common_input(run)
    # 对应论文明确提出的 Top-k 建图参数
    run.add_argument("--top-k", type=int)
    # 不是论文算法参数，是代码中保存中间结果的开关实验，默认设置为 Top-k = 20。
    run.add_argument("--save-arrays", action="store_true")


    '''
    就是创建一个叫 demo 的子命令，用于提供生成模拟数据并运行完整本地算法流程的演示入口。
    demo → 生成模拟数据 → 运行算法 → 保存结果
    没有直接调用 generate 命令，但是调用了与 generate 命令相同的模拟数据生成函数。
    '''
    demo = commands.add_parser("demo", help="Generate synthetic data and run the complete local pipeline")
    # --output 的参数，用来指定演示结果保存的位置。
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--events", type=int, default=768)
    demo.add_argument("--config")
    demo.add_argument("--batch-size", type=int)
    demo.add_argument("--top-k", type=int)
    demo.add_argument("--seed", type=int)
    demo.add_argument("--site-id", default="local")
    demo.add_argument("--save-arrays", action="store_true")

    # benchmark（基准测试）：让 HoSig-Align I 和其他算法在相同的数据上接受测试，看看它们各自的表现怎么样。
    evaluation = commands.add_parser("benchmark", help="Tune baselines on held-out prefixes and evaluate grid")
    _common_input(evaluation)
    # --top-k-grid 是复现代码为了方便进行参数实验而设计的。
    '''
    因为不同数据集的事件分布可能不一样：
    - k 太小：可能遗漏本来相关的事件。
    - k 太大：可能把不相关的事件也连接起来。
    - 合适的 k：有助于构建更合理的事件关系图。
    所以我们可以通过实验比较不同 k 的效果。
    '''
    evaluation.add_argument("--top-k-grid", type=int, nargs="+", default=[5, 10, 20, 40])
    # -clients 是为了测试不同数量的站点参与时的情况；
    evaluation.add_argument("--clients", type=int, nargs="+", default=[2, 4, 8])
    # 验证集占比：先拿一部分数据试一试，帮助选择合适的参数，然后再用没有参与调参的数据进行评估。
    evaluation.add_argument("--validation-fraction", type=float, default=0.2)
    # 它主要控制基线算法分组之后，采用哪种特征表示来计算每个聚类的代表向量，用于后续评估。
    evaluation.add_argument("--entity-protocol", choices=("paper", "common-rff"), default="paper")
    # 拟合（Fit）：给算法一批数据，让算法根据数据计算出模型或聚类结果。
    # 调优（Tune）：尝试不同参数，多次运行算法，比较效果并选择参数。
    # 时间预算（Fit Budget）：限制拟合或调优阶段可以花费的计算时间。
    evaluation.add_argument("--fit-budget-seconds", type=float)

    '''构建或增量维护一个原型超图。'''
    alignment = commands.add_parser("align", help="Build or incrementally maintain a prototype hypergraph")
    # 读取前面已经生成的签名，供 align 进行后续关联分析,必须输入一个或者多个
    alignment.add_argument("--signatures", nargs="+", required=True)
    # 这个参数用于指定 align 生成的超图结果保存在哪里。
    alignment.add_argument("--output", default="outputs/hypergraph.json")
    # 这个参数是用来设置HoSig 签名三个部分的权重，决定在比较不同 HoSig 签名时，更重视哪一方面的信息。
    alignment.add_argument("--weights", nargs=3, type=float, default=[1 / 3, 1 / 3, 1 / 3])
    # 就是一条判断标准，用来决定哪些 HoSig 签名足够相似，可以考虑分到同一组。差异分数不超过 1.0，
    # 就认为它们足够接近。（用的是权重那里计算出来的结果）
    # 阈值设置得比较小，条件比较严格。
    # 阈值设置得比较大，条件比较宽松。
    '''程序先结合 HoSig 的三个特征，计算签名之间的差异，
    再根据 cluster-threshold 设定的标准，判断这些签名是否满足聚类条件。'''
    alignment.add_argument("--cluster-threshold", type=float, default=1.0)
    # 已经形成的两个签名簇，是否满足合并的条件。
    '''
    之前分开的两个签名簇，随着新信息加入，
    可能被发现属于同一类相关攻击活动，因此需要有条件地合并，而不是永远保持分开。
    '''
    alignment.add_argument("--merge-threshold", type=float, default=1.0)

    # 拆分阈值：防止一个签名簇内部混入差异较大的 HoSig 签名，让最终形成的关联簇尽可能保持内部特征一致。
    alignment.add_argument("--split-threshold", type=float, default=0.5)
    '''align 中这三个参数理解为：
    先聚类，再根据需要合并或拆分，让签名之间的关联结构更加合理。'''

    # --incremental 的意思是“是否启用增量更新模式”
    '''已经有一个签名关联超图了，现在来了新的 HoSig 签名，我们希望在已有结果的基础上更新，而不是每次都从零开始。'''
    alignment.add_argument("--incremental", action="store_true")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "write-config":
            # 将配置写到文件
            write_json(args.output, Config().to_dict())
        elif args.command == "generate":
            write_events(args.output, generate_synthetic(args.events, args.campaigns, args.seed))
            print(f"Synthetic telemetry: {args.output}")
        elif args.command == "convert-iot23":
            metadata = convert_iot23(args.input, args.output, args.label_mode, args.scenario, args.order, args.limit)
            print(f"IoT-23: {metadata['converted_records']} events -> {args.output}")
            print(f"Label counts: {metadata['converted_label_counts']}")
            print(metadata["warning"])
        elif args.command == "demo":
            config = _config(args)
            path = Path(args.output) / "synthetic.jsonl"
            write_events(path, generate_synthetic(args.events, seed=config.seed))
            args.input, args.format, args.proxy_interval, args.limit = str(path), "jsonl", None, None
            run_stream(args, config)
        elif args.command == "run":
            run_stream(args, _config(args))
        elif args.command == "benchmark":
            config = _config(args)
            input_metadata = _input_metadata(args)
            events = list(read_events(args.input, args.format, args.proxy_interval, args.limit))
            if not events:
                raise ValueError("Input contains no events")
            if args.fit_budget_seconds is not None and args.fit_budget_seconds <= 0:
                raise ValueError("fit-budget-seconds must be positive")
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            report = benchmark(
                events, config, args.top_k_grid, args.clients, args.validation_fraction,
                args.entity_protocol, args.fit_budget_seconds, progress=print,
            )
            report.update({
                "config": config.to_dict(), "environment": environment_info(), "input": input_metadata,
                "site_protocol": "single_capture_site" if args.clients == [1] else "synthetic_round_robin_sites_not_original_organizations",
                "validation_fraction": args.validation_fraction, "fit_budget_seconds": args.fit_budget_seconds,
            })
            write_json(output / "benchmark.json", report)
            write_benchmark_csv(output / "metrics.csv", report["records"])
            plot_benchmark(report["records"], output)
            print(f"Benchmark: {output / 'metrics.csv'}")
        elif args.command == "align":
            signatures = [signature for path in args.signatures for signature in load_signatures(path)]
            if not signatures:
                raise ValueError("No signatures supplied")
            if len({signature.block_id for signature in signatures}) != len(signatures):
                raise ValueError("Duplicate block IDs; namespace each site's block_id before cross-site alignment")
            graph = PrototypeHypergraph(args.cluster_threshold, args.merge_threshold, args.split_threshold, tuple(args.weights))
            # 增量处理：每次拿一个签名出来，加入当前超图，再处理下一个签名。
            if args.incremental:
                import time

                total = len(signatures)
                start = time.perf_counter()

                print(f"开始增量构建超图，共 {total} 条签名", flush=True)

                for i, signature in enumerate(signatures, start=1):
                    t0 = time.perf_counter()

                    print(f"正在处理第 {i}/{total} 条签名...", flush=True)

                    graph.add(signature)

                    cost = time.perf_counter() - t0

                    if i % 10 == 0 or cost > 5:
                        print(
                        f"已完成 {i}/{total} 条，"
                        f"当前签名耗时 {cost:.2f} 秒，"
                         f"累计耗时 {time.perf_counter() - start:.2f} 秒",
                        flush=True
                        )
                print("全部签名处理完成！", flush=True)

            else:
                graph.fit(signatures)
            write_json(args.output, graph.to_dict())
            print(f"Hypergraph: {args.output}")
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
