"""训练共享 TCN，保存可重载的模型，并报告验证集 Rank IC。

从策略目录试跑 100 个训练批次，随后验证两个交易日的小规模推理与评价。
    trial_dir=$(mktemp -d)
    python scripts/train_tcn.py --smoke --output "$trial_dir/run"
默认读取共享 Qlib 行情、训练缓存和 data/training_metadata/metadata.json。
成功时输出目录包含 model.pt 和 report.json，报告含分段训练与固定样本验证损失。
--pilot 训练 5,000 批，并为固定的 6 个验证交易日导出逐股逐分钟预测。
全量训练每 1,000 批记录训练损失，每 10,000 批验证固定交易日，轮末再作完整验证。
省略 --smoke 和 --pilot 时训练全量一轮。
默认在窗内归一化后按预测分钟及窗口位置去均值，再用重估的分时倍率缩放四列价格。
--no-center-positions 恢复未经去均值的训练口径。--price-scale 单独控制去均值之后的价格乘数。
默认在首个残差块后用每根 bar 的时钟对通道做乘性加权。--hidden-channels 配置卷积宽度，预测头的隐藏层固定 16 维。
--clock-mode head 运行原时钟后融合结构，--channels 6/8 分别运行纯量价及原八通道结构。
"""

import argparse  # 解析训练数据、设备和输出路径。
import json  # 读取数据来源并保存实验报告。
import sys  # 支持直接执行脚本时定位源码包。
from hashlib import file_digest  # 对元数据与缓存索引做流式 SHA-256 校验。
from math import isfinite  # 检查验证损失可用于比较。
from pathlib import Path  # 定位 Qlib、缓存和运行产物。
from time import perf_counter  # 分别记录训练和验证耗时。

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 定位 TCN 策略目录。
SEED = 20261126  # 固定模型初始化和训练索引打乱种子。
SCALE_SEED = 20261127  # 独立固定训练标签尺度的抽样种子。
SCALE_STOCK_DAYS = 12_288  # 从训练缓存均匀抽出这些股票日估计分钟标准差。
RETURN_SCALE = 0.0030267951862079545  # 保存基于训练标签的全局标准差作为验证单位及备用尺度。
LEARNING_RATE = 0.001  # 使用确认的固定学习率。
WEIGHT_DECAY = 0.0001  # 使用确认的 AdamW 权重衰减。
SMOKE_BATCHES = 100  # 短试跑更新的训练批次数。
PILOT_BATCHES = 5_000  # 固定验证日预测试验更新的训练批次数。


def _sha256(path):  # 计算已存在文件的 SHA-256。
    """path 为文件路径，返回十六进制校验值，读取失败时传递 OSError。"""
    with path.open("rb") as stream:  # 流式读取文件，避免将整份 SQLite 数据库放进内存。
        return file_digest(stream, "sha256").hexdigest()  # 记录内容的校验值。


def _data_sources(args, metadata_spec):  # 记录模型数据的可复现来源。
    """args 为命令行路径，metadata_spec 为元数据清单字典，返回可序列化的来源信息。"""
    cache_manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))  # 读出已校验的 Qlib 来源签名。
    auxiliary_files = {  # 解析元数据清单引用的文件路径并去重。
        (args.metadata.parent / declaration["path"]).resolve()  # 将相对路径解释为清单所在目录。
        for kind in ("daily", "ipo", "delist", "full_halt", "intraday_halt")  # 与正式元数据需要的来源类别一致。
        for declaration in metadata_spec[kind]  # 处理同一类别的各段覆盖范围。
    }  # 得到本次实际关联的辅助文件。
    return {  # 将所有输入的版本信息写入检查点及报告。
        "qlib": str(args.qlib.resolve()),  # 保存 Qlib 行情库路径。
        "qlib_source_signature": cache_manifest["source_signature"],  # 保存训练缓存已核验的行情来源签名。
        "metadata": str(args.metadata),  # 保存元数据清单的固定路径。
        "metadata_sha256": _sha256(args.metadata),  # 保存清单内容校验值。
        "metadata_files_sha256": {str(path): _sha256(path) for path in sorted(auxiliary_files)},  # 保存每份辅助数据的版本。
        "stock_day_cache": str(args.cache.resolve()),  # 保存原始分钟缓存路径。
        "stock_day_index_sha256": _sha256(args.cache / "keys.npy"),  # 保存实际训练股票日目录的校验值。
        "cached_stock_days": cache_manifest["stock_days"],  # 保存训练范围中的股票日数。
    }  # 返回数据来源记录。


def _parse_args():  # 提取运行范围与数据位置。
    """从命令行解析路径和试跑开关，参数无效时由 argparse 退出。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # 在帮助中保留试跑命令。
    parser.add_argument("--qlib", type=Path, default=PROJECT_ROOT.parent / "data/qlib_cn_1min")  # 默认使用共享分钟库。
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data/stock_day_cache/train")  # 读取已建的全量训练缓存。
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/training_metadata/metadata.json")  # 指向固定的核验清单。
    parser.add_argument("--output", type=Path, required=True)  # 指定一个新的模型与报告目录。
    modes = parser.add_mutually_exclusive_group()  # 防止一个运行同时选择两种短程范围。
    modes.add_argument("--smoke", action="store_true")  # 运行 100 批训练及较小的验证范围。
    modes.add_argument("--pilot", action="store_true")  # 运行 5,000 批训练并导出六个固定验证日的预测。
    clock = parser.add_mutually_exclusive_group()  # 新默认和历史通道对照分别选择一处显式时钟。
    clock.add_argument("--clock-mode", choices=("gate", "head"), default="gate")  # 默认首块后加权，head 恢复原时钟预测头。
    clock.add_argument("--channels", type=int, choices=(6, 8))  # 保留原六通道纯量价和原八通道线性头入口。
    parser.add_argument("--hidden-channels", type=int, default=16, help="通道加权模型的卷积宽度，默认 16。")  # 预测头隐藏层宽度固定。
    parser.add_argument("--no-center-positions", dest="center_positions", action="store_false", help="沿用未逐位置去均值的基线输入。")  # 默认启用训练集逐位置去均值。
    parser.add_argument("--price-scale", choices=("minute", "global", "none"), default="minute", help="价格输入倍率的估计口径，默认按预测分钟。")  # 提供分时、全天共用和单位倍率三种设置。
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")  # 自动优先使用可用的 GPU。
    parser.add_argument("--threads", type=int)  # 可在 CPU 试跑后指定 PyTorch 计算线程数。
    args = parser.parse_args()  # 读取传入的运行参数。
    if args.threads is not None and args.threads <= 0:  # 检查用户指定的 CPU 线程数。
        parser.error("--threads 必须为正整数")  # 提前报告无效参数。
    args.time_fusion = {6: "none", 8: "early"}.get(args.channels, {"gate": "gate_only", "head": "late"}[args.clock_mode])  # 统一解析旧通道参数与当前时钟模式。
    if args.hidden_channels < 1 or (args.time_fusion != "gate_only" and args.hidden_channels != 16):  # 在读取数据前核对请求的架构。
        parser.error("卷积宽度须为正整数，历史结构使用 16 通道。")  # 提示调用方保留旧模型的宽度。
    return args  # 将解析后的参数交给主程序。


def _input_tables(estimate, center_positions, price_mode):  # 将训练拟合结果转换为模型实际使用的输入变换。
    """返回可选位置均值表、可选价格倍率表以及实际使用的全天备用倍率。"""
    import torch  # 让缩放与检查点恢复使用相同的单精度缓冲区。

    means = torch.from_numpy(estimate.position_means) if center_positions else None  # 去均值独立于倍率开关。
    if price_mode == "none":  # 单位倍率仍可使用训练位置均值表。
        return means, None, 1.0  # 不创建单位倍率缓冲区，沿用共同前向路径。
    multipliers = torch.from_numpy(estimate.multipliers)  # 默认按预测分钟使用此次拟合的倍率。
    if price_mode == "global":  # 全天统一倍率来自同一批、同一预处理口径的训练窗口。
        multipliers = torch.full_like(multipliers, estimate.global_multiplier)  # 将全天值广播成同索引格式。
    return means, multipliers, estimate.global_multiplier  # 保存所选倍率和对应的全天回退信息。


def _checked_huber(metrics):  # 核对验证损失的有效样本数量。
    """metrics 为验证指标字典。返回可比较的平均 Huber，损失缺失或少算样本时抛出 FloatingPointError。"""
    mean_huber = metrics["mean_huber"]  # 读取以训练标签尺度计算的平均验证损失。
    if mean_huber is None or not isfinite(mean_huber) or metrics["huber_samples"] != metrics["labels_available"]:  # 检查损失与有效标签覆盖。
        raise FloatingPointError("验证 Huber 无法覆盖全部有效标签")  # 防止不同检查点在不同样本上比较损失。
    return mean_huber  # 交付可报告的固定样本损失。


def _export_predictions(path, daily_forecasts, *, scale, return_label):  # 逐日保存实际单位的样本外预测。
    """daily_forecasts 交付逐日预测数组。写出 Parquet，返回覆盖率对应的行数和分时段误差。"""
    import numpy as np  # 计算模型和零收益基线的原始单位误差。
    import pyarrow as pa  # 显式声明时间戳、可空标签和收益率类型。
    import pyarrow.compute as pc  # 将北京时间墙上时钟标注为有时区的时间戳。
    import pyarrow.parquet as pq  # 按交易日追加行组，限制内存使用量。

    schema = pa.schema([  # 固定预测文件的字段名、类型和单位。
        ("code", pa.string()),  # 当日按项目口径统一后的股票代码。
        ("t", pa.timestamp("ms", tz="Asia/Shanghai")),  # 出分使用的北京时间分钟。
        ("prediction", pa.float64()),  # 还原到原始单位的预测对数收益。
        ("label", pa.float64()),  # 可用时写实际对数收益，否则写空值。
        ("label_valid", pa.bool_()),  # 标明该候选能否参与事后绩效评估。
    ], metadata={"return_label": return_label, "return_unit": "log_return"})  # 随文件保存标签与单位口径。
    pending = path.with_suffix(".parquet.tmp")  # 预测完全写出前使用临时文件。
    periods = ("morning", "afternoon")  # 按两个竞价时段统计误差和分母。
    counts = dict.fromkeys(periods, 0)  # 累计每个时段的有效标签条数。
    errors = dict.fromkeys(periods, 0.0)  # 累计模型的绝对收益率误差。
    zero_errors = dict.fromkeys(periods, 0.0)  # 累计预测恒为零时的绝对误差。
    rows = 0  # 记录所有能输出预测的候选数量。
    zero_huber_total = 0.0  # 累计零收益基线的验证 Huber。

    with pq.ParquetWriter(pending, schema) as writer:  # 当天处理结束后释放当日的预测数组。
        for day in daily_forecasts:  # 每次接收一个交易日的预测和回看标签。
            valid = day["label_valid"]  # 标签掩码独立于是否能生成预测。
            labels = day["label"][valid]  # 误差评价只使用有标签的候选。
            prediction = day["prediction"]  # 每个候选的物理收益率预测。
            clock = day["t"][valid].astype(np.int64) % (24 * 60)  # 取有效标签的北京时间分钟索引。
            morning = clock < 13 * 60  # 用下午竞价开始前的分钟标记上午时段。
            error = np.abs(prediction[valid] - labels)  # 绝对误差仍为对数收益率。
            zero_error = np.abs(labels)  # 同一批候选的零预测绝对误差。
            standardized = zero_error / scale  # 使用和模型验证损失相同的固定单位。
            zero_huber_total += float(np.sum(np.where(standardized <= 1, 0.5 * standardized**2, standardized - 0.5)))  # 按同一个 Huber 阈值评价零预测。
            for period, mask in zip(periods, (morning, ~morning), strict=True):  # 分别汇总上午和下午的误差。
                counts[period] += int(np.count_nonzero(mask))  # 记录有效对数以跨日按样本汇总。
                errors[period] += float(error[mask].sum())  # 累计模型预测误差。
                zero_errors[period] += float(zero_error[mask].sum())  # 累计零预测误差。
            times = pc.assume_timezone(pa.array(day["t"].astype("datetime64[ms]")), "Asia/Shanghai")  # 保存实际北京时间。
            table = pa.Table.from_arrays([  # 将股票、出分分钟、预测和标签按行对齐。
                pa.array(day["code"].tolist()), times, pa.array(prediction),  # 写入每个候选的代码、时间和预测。
                pa.array(day["label"], mask=~valid), pa.array(valid),  # 让无标签记录仍然可以保留预测。
            ], schema=schema)  # 为每个交易日沿用同一文件结构。
            writer.write_table(table)  # 完整追加当天结果。
            rows += len(table)  # 记录当天已写入的预测条数。
    pending.replace(path)  # Parquet 关闭并完成文件末尾后再发布。
    valid_rows = sum(counts.values())  # 各时段有效标签计数之和即全部可评价预测。
    return {  # 将导出结果及分时段误差交给训练报告。
        "file": path.name,  # 保存相对报告所在目录的文件名。
        "rows": rows,  # 记录带预测的候选总数。
        "labels_available": valid_rows,  # 记录实际收益可用的行数。
        "zero_prediction_huber": zero_huber_total / valid_rows if valid_rows else None,  # 与同样本的验证损失比较。
        "by_session": {  # 使用基点记录上午及下午的误差。
            period: {  # 每个时段独立报告样本数和两种预测的平均误差。
                "labels_available": count,  # 保存该时段的有效样本数。
                "mae_bp": 10_000 * errors[period] / count if count else None,  # 模型预测的平均绝对误差。
                "zero_prediction_mae_bp": 10_000 * zero_errors[period] / count if count else None,  # 零预测的平均绝对误差。
            }  # 完成时段结果。
            for period, count in counts.items()  # 按各时段样本数计算平均误差。
        },  # 完成时段统计。
    }  # 返回可序列化的预测文件信息。


def main():  # 配置并执行一次完整的训练、保存和验证流程。
    """使用已确认的训练设置写出 model.pt 与 report.json。错误时保留具体异常位置。"""
    if str(PROJECT_ROOT) not in sys.path:  # 直接运行脚本时，确保能导入 src 包。
        sys.path.insert(0, str(PROJECT_ROOT))  # 将策略目录加入模块查找路径。

    import numpy as np  # 创建训练迭代器使用的随机数生成器。
    import torch  # 用 PyTorch 训练、保存和重载模型。

    from src.data.metadata import Metadata  # 打开已核验的日线与交易事件。
    from src.data.minutes import MinuteLibrary  # 复用统一的 Qlib 和训练缓存读取器。
    from src.features.batches import (  # 读取已有的组批设置和标签计数。
        DEFAULT_STOCK_DAYS_PER_GROUP,  # 使用每组 128 个股票日。
        DEFAULT_TRAIN_BATCH_SIZE,  # 使用每批最多 2,048 条样本。
        TrainingSampleCounts,  # 统计候选数及其中可用于训练的标签。
    )  # 完成训练批次配置导入。
    from src.features.price_scale import (
        PriceScaleAccumulator,  # 使用训练窗口估计价格输入尺度。
    )
    from src.features.samples import (  # 取得正式特征顺序及日期切分。
        ENTRY_OFFSET,  # 出分后的入场分钟偏移。
        FEATURES,  # 六列基础量价特征。
        HORIZON,  # 出场相对预测时点的分钟偏移。
        RETURN_LABEL,  # 记录本次训练的收益率标签定义。
        TIME_FEATURES,  # 两列日内时段特征。
        split_for,  # 判断给定交易日所属的集合。
    )  # 完成特征定义导入。
    from src.models.evaluation import (  # 共用验证及逐日预测的推理入口。
        evaluate,  # 计算成组的 Rank IC 和验证 Huber。
        predict_day,  # 为指定日期生成全部合格候选的预测。
    )  # 完成验证与导出接口导入。
    from src.models.fit import fit_epoch  # 使用 PyTorch Huber 和优化器执行训练循环。
    from src.models.minute_scale import (  # 使用训练集分分钟尺度。
        estimate_minute_scales,  # 仅用正式训练标签拟合各预测分钟的尺度。
    )  # 完成尺度估计接口导入。
    from src.models.tcn import ReturnTCN  # 实例化与本次特征通道数匹配的共享 TCN。
    from src.models.training_data import (  # 复用与 TCN 输入布局匹配的加载器。
        make_training_loader,  # 通过正式 DataLoader 读取批次。
    )  # 完成训练加载器导入。

    args = _parse_args()  # 确定本次运行路径和范围。
    if args.output.exists() or args.output.is_symlink():  # 在训练前防止覆盖已有结果。
        raise FileExistsError(f"输出目录已存在：{args.output}")  # 提示选择一个新目录。
    if args.threads is not None:  # 使用显式 CPU 线程设置时须在首次计算前应用。
        torch.set_num_threads(args.threads)  # 设置本次进程的计算线程数。
    device_name = args.device  # 读取请求的设备类型。
    if device_name == "auto":  # 根据当前进程可访问的硬件确定默认设备。
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # 优先选可用的 GPU。
    device = torch.device(device_name)  # 构造 PyTorch 设备对象。
    torch.manual_seed(SEED)  # 固定模型初始化的随机状态。
    rng = np.random.default_rng(SEED)  # 固定股票日和组内样本的初始打乱顺序。
    args.metadata = args.metadata.resolve()  # 保证来源引用和报告都使用绝对路径。
    spec = json.loads(args.metadata.read_text(encoding="utf-8"))  # 读取正式元数据清单。
    library = MinuteLibrary(args.qlib, stock_day_cache=args.cache)  # 校验并加载训练股票日缓存。
    dates = sorted(day for day in library.dates if day.startswith("2025") and split_for(day) == "validation")  # 按正式切分选取验证日。
    if not dates:  # 在拟合尺度或模型之前确认能做样本外验证。
        raise ValueError("分钟库没有 2025 年上半年的验证交易日")  # 明确给出缺少的日期范围。
    monitor_indices = np.linspace(0, len(dates) - 1, min(6, len(dates)), dtype=int)  # 按日历等距选取固定验证日。
    monitor_dates = dates[:2] if args.smoke else [dates[index] for index in monitor_indices]  # 短试跑单独使用前两个验证日。
    validation_dates = monitor_dates if args.smoke or args.pilot else dates  # 对整轮训练另作完整验证。
    validation_limit = 32 if args.smoke else None  # 短试跑每个验证日最多读取三十二只合格股票。
    loss_interval = 50 if args.smoke else 500 if args.pilot else 1_000  # 按固定批次跨度记录训练损失。
    monitor_interval = 50 if args.smoke else 2_500 if args.pilot else 10_000  # 到该间隔在固定验证日检查损失。
    batch_limit = SMOKE_BATCHES if args.smoke else PILOT_BATCHES if args.pilot else None  # 正式训练消费一整轮。
    sources = _data_sources(args, spec)  # 在训练启动前记录并核验输入文件。
    time_at_head = args.time_fusion == "late"  # 只有原后融合结构在预测头显式接收两列时钟。
    window_clock_mode = "gate" if args.time_fusion == "gate_only" else "none"  # 仅在选择通道加权时启用首块后的乘性门。
    input_channels = args.channels or len(FEATURES)  # 通道加权与旧后融合模型都只接收六列原始量价。
    features = FEATURES + TIME_FEATURES if input_channels == 8 else FEATURES  # 记录卷积实际收到的列。
    fit_inputs = args.center_positions or args.price_scale != "none"  # 两种输入变换共用已抽取的训练窗口。
    price_accumulator = PriceScaleAccumulator(center_positions=args.center_positions) if fit_inputs else None  # 默认先按位置去均值再重估倍率。
    print("开始估计训练标签与输入尺度", flush=True)  # 区分前置尺度估计与模型训练耗时。
    with Metadata(spec, args.metadata.parent) as scale_metadata:  # 尺度估计沿用训练标签所需的交易元数据。
        minute_scales, scale_stats = estimate_minute_scales(  # 仅从训练缓存均匀抽取股票日。
            library, scale_metadata, sources["cached_stock_days"],  # 使用和正式训练一致的数据与候选规则。
            seed=SCALE_SEED, sample_size=SCALE_STOCK_DAYS, fallback_scale=RETURN_SCALE,  # 固定抽样并配置未观测分钟的备用值。
            price_accumulator=price_accumulator,  # 与标签尺度共用同一遍训练样本读取。
        )  # 返回尺度表及样本数、均值和标准差。
    print(f"已估计 {len(scale_stats)} 个分钟的标签尺度 · {sum(row['samples'] for row in scale_stats):,} 条训练标签", flush=True)  # 确认尺度估计的实际覆盖。
    price_estimate = None if price_accumulator is None else price_accumulator.finish()  # 把训练窗口的矩汇总为固定倍率。
    position_means, minute_price_multipliers, global_price_multiplier = _input_tables(price_estimate, args.center_positions, args.price_scale)  # 冻结本次训练的输入统计量。
    if minute_price_multipliers is not None:  # 记录训练真正采用的分时或全天倍率范围。
        print(f"价格输入倍率 {args.price_scale} · {minute_price_multipliers.min():.4f}–{minute_price_multipliers.max():.4f}", flush=True)  # 此时缓冲区仍在 CPU。
    model = ReturnTCN(  # 保存输入倍率及按分钟还原收益率的尺度。
        return_scale=RETURN_SCALE, input_channels=input_channels, minute_return_scales=torch.from_numpy(minute_scales),  # 配置卷积通道及收益率尺度。
        time_at_head=time_at_head, window_clock_mode=window_clock_mode,  # 采用已选的单一时钟接入位置。
        hidden_channels=args.hidden_channels,  # 所有卷积阶段采用相同宽度。
        minute_price_multipliers=minute_price_multipliers,  # 训练与推理使用固定输入倍率。
        position_means=position_means,  # 在卷积前先减去对应预测分钟的三列位置均值。
    ).to(device)  # 将参数与尺度缓冲区都移到训练设备。
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # 对实际模型参数配置 AdamW。
    counts = TrainingSampleCounts()  # 在训练迭代过程中累计候选及有效标签数。
    config = {  # 记录影响训练结果的设置。
        "mode": "smoke" if args.smoke else "pilot" if args.pilot else "full_epoch",  # 区分短试跑、固定验证日预测和全量一轮。
        "seed": SEED,  # 记录模型初始化和训练打乱的随机种子。
        "return_label": RETURN_LABEL,  # 区分下一分钟开盘入场标签与历史报告中的收盘起算标签。
        "entry_offset_minutes": ENTRY_OFFSET,  # 记录入场分钟相对预测分钟的偏移。
        "exit_offset_minutes": HORIZON,  # 记录退出分钟相对预测分钟的偏移。
        "return_scale": RETURN_SCALE,  # 记录用于验证 Huber 和备用尺度的训练集总体标准差。
        "label_scale_mode": "prediction_minute",  # 按北京时间的出分分钟除以标准差，均值不扣除。
        "label_scale_seed": SCALE_SEED,  # 记录与训练打乱独立的股票日抽样种子。
        "label_scale_stock_days": min(SCALE_STOCK_DAYS, sources["cached_stock_days"]),  # 记录实际用于估计标准差的训练股票日数。
        "optimizer": "AdamW",  # 记录优化器名称。
        "learning_rate": LEARNING_RATE,  # 记录固定学习率。
        "weight_decay": WEIGHT_DECAY,  # 记录优化器权重衰减。
        "huber_delta": 1.0,  # 记录缩放后的 Huber 损失阈值。
        "training_loss_interval": loss_interval,  # 保存每段训练损失的批次跨度。
        "validation_loss_interval": monitor_interval,  # 保存训练中重复验证的间隔。
        "batch_size": DEFAULT_TRAIN_BATCH_SIZE,  # 记录每批样本数上限。
        "stock_days_per_group": DEFAULT_STOCK_DAYS_PER_GROUP,  # 记录每组读入的股票日数。
        "features": list(features),  # 记录模型的实际输入通道及排列。
        "input_channels": input_channels,  # 为检查点重载保存第一层所需通道数。
        "time_fusion": args.time_fusion,  # 记录时钟实际接入的位置。
        "window_clock_mode": window_clock_mode,  # 让重载按训练时的配置恢复乘性门。
        "first_conv_input_channels": input_channels,  # 交付进入初始卷积的实际列数。
        "hidden_channels": args.hidden_channels,  # 检查点按此宽度恢复卷积网络。
        "trainable_parameters": sum(p.numel() for p in model.parameters()),  # 模型中的所有网络参数均参与训练。
        "center_positions": args.center_positions,  # 标明检查点是否必须带训练集拟合的位置均值表。
        "price_input_scale_mode": args.price_scale,  # 区分分时、全天共用与单位倍率。
        "price_input_scale": None if args.price_scale == "minute" else global_price_multiplier,  # 单一倍率在报告中直接记录。
        "price_input_scale_fallback": global_price_multiplier,  # 保存未观察分钟的备用倍率。
        "head_features": list(TIME_FEATURES) if time_at_head else [],  # 记录预测头收到的时钟特征顺序。
        "device": str(device),  # 记录本次实际运行的设备。
        "torch_version": str(torch.__version__),  # 记录运行时 PyTorch 版本。
        "torch_threads": torch.get_num_threads(),  # 记录 CPU 并行设置，便于复验速度。
    }  # 完成训练配置记录。
    args.output.mkdir(parents=True)  # 为本次运行创建独立的结果目录。
    print(f"开始训练 · {config['mode']} · {device} · PyTorch 线程 {torch.get_num_threads()}", flush=True)  # 报告实际运行配置。

    monitoring_checks = []  # 保存初始模型和随后各检查点在同一验证子集上的指标。
    prediction_report = None  # 只有固定验证日预测模式才写出预测文件。
    with Metadata(spec, args.metadata.parent) as metadata:  # 保持训练与验证使用的只读元数据连接打开。
        def monitor(batch_number: int, *, force: bool = False) -> None:  # 在指定批次对固定验证子集计算指标。
            """batch_number 为已经更新的批次数，force 为 True 时在非整间隔位置也运行检查。"""
            if batch_number % monitor_interval and not force:  # 训练分段日志频率可以高于完整检查频率。
                return  # 继续训练，待下一次监测间隔再做推理。
            check_start = perf_counter()  # 单独记录这次监测用时。
            metrics = evaluate(  # 重复使用固定的验证日期和股票范围。
                model, library, metadata, monitor_dates,  # 通过和轮末相同的采样与推理流程。
                batch_size=DEFAULT_TRAIN_BATCH_SIZE, device=device,  # 沿用同样的推理批次和计算设备。
                max_stock_days_per_date=validation_limit,  # 试跑限定股票数，全量监测日覆盖所有合格股票。
            )  # 取得固定验证集损失和排序指标。
            mean_huber = _checked_huber(metrics)  # 核对每条有效标签都参与了验证损失。
            monitoring_checks.append({  # 记录检查点，供训练结束时保存完整曲线。
                "batch": batch_number,  # 对齐训练进度，零表示开始更新之前。
                **metrics,  # 记录平均损失、覆盖率和两个 Rank IC。
                "elapsed_seconds": perf_counter() - check_start,  # 保存这次验证用时。
            })  # 完成监测记录。
            print(f"批次 {batch_number:,} · 固定验证 Huber {mean_huber:.6f}", flush=True)  # 实时报告固定验证损失。

        monitor(0)  # 先测量随机初始化模型的验证损失作为参考。
        loader = make_training_loader(library, metadata, library.cached_stock_days, rng=rng, counts=counts)  # 使用确认的分组与两层打乱。
        training = fit_epoch(  # 运行试跑或全量一轮，同时记录分段曲线。
            model, loader, optimizer, device=device,  # 使用正式训练数据和已确认的优化器。
            max_batches=batch_limit,  # 只在短程运行时截断一轮。
            report_every=loss_interval, on_report=monitor,  # 记录训练分段损失，并按检查间隔验证。
        )  # 返回训练总量与分段记录。
        training["candidates_prepared"] = counts.candidates  # 预先组装的候选数可能包含试跑末组尚未消费的样本。
        training["labels_available_prepared"] = counts.labels_available  # 单列已生成的有效标签数，消费数由 fit_epoch 记录。
        print(json.dumps(training, ensure_ascii=False), flush=True)  # 在验证开始前报告训练进度与结果。

        checkpoint_path = args.output / "model.pt"  # 定位本次训练的模型文件。
        pending = checkpoint_path.with_suffix(".pt.tmp")  # 先写同目录临时文件，避免交付部分权重。
        torch.save({  # 保存模型和继续训练需要的状态。
            "model": model.state_dict(),  # 包含模型权重和把预测还原到收益率单位的尺度。
            "optimizer": optimizer.state_dict(),  # 包含 AdamW 的动量及更新步数。
            "config": config,  # 记录使用的训练超参数和设备。
            "sources": sources,  # 保存 Qlib、元数据和股票日清单的来源签名。
            "training": training,  # 保存训练轮次的候选数、损失与耗时。
        }, pending)  # 将完整检查点写入临时文件。
        pending.replace(checkpoint_path)  # 一次性发布检查点。
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # 立即检查保存的状态可以用限制模式读取。
        restored = ReturnTCN.from_checkpoint(saved)  # 和独立预测共享严格的模型及输入表重载逻辑。
        for name, value in restored.state_dict().items():  # 验证每个重载的张量都与实际训练状态一致。
            torch.testing.assert_close(value, model.state_dict()[name].detach().cpu(), rtol=0, atol=0)  # 对比保存前后的张量值。
        del saved, restored  # 验证完成后释放检查点副本。

        if monitoring_checks[-1]["batch"] != training["batches"]:  # 如果轮末不在整数检查间隔上，补记最终模型的固定样本损失。
            monitor(training["batches"], force=True)  # 让最终模型损失也进入同一条曲线。
        print("模型重载通过", flush=True)  # 确认已保存和验证磁盘上的模型。
        if args.smoke or args.pilot:  # 短程运行的最终验证范围与监测范围相同。
            validation = monitoring_checks[-1].copy()  # 复用最终检查结果，避免重复推理。
        else:  # 正式训练结束后使用完整验证期评价最终模型。
            print("开始完整验证", flush=True)  # 报告从固定子集监测切换到完整验证阶段。
            started = perf_counter()  # 开始记录完整验证耗时。
            validation = evaluate(  # 在全部验证交易日计算损失、排序指标和标签覆盖率。
                model, library, metadata, validation_dates,  # 传入完整验证日期。
                batch_size=DEFAULT_TRAIN_BATCH_SIZE, device=device,  # 沿用推理批次大小及设备。
            )  # 取得完整的轮末验证指标。
            _checked_huber(validation)  # 确认全部有效标签均有可用预测。
            validation["elapsed_seconds"] = perf_counter() - started  # 记录完整验证耗时。
        if args.pilot:  # 在固定的样本外日期保存逐股逐分钟的预测。
            print("开始导出固定验证日预测", flush=True)  # 说明最终检查后仍在输出可读结果。
            daily_forecasts = (  # 每次按一个交易日生成预测及独立的事后标签。
                predict_day(model, library, metadata, day, batch_size=DEFAULT_TRAIN_BATCH_SIZE, device=device)  # 复用正式验证的采样与模型推理。
                for day in validation_dates  # 使用固定日期的全部合格候选。
            )  # 延迟读取后续交易日，控制内存用量。
            prediction_report = _export_predictions(  # 按日流式输出原始对数收益率。
                args.output / "predictions.parquet", daily_forecasts, scale=RETURN_SCALE, return_label=RETURN_LABEL,  # 保存并对照同量纲的零收益基线。
            )  # 生成覆盖率及分时段误差报告。
            if (prediction_report["rows"], prediction_report["labels_available"]) != (validation["candidates"], validation["labels_available"]):  # 核对导出行与验证分母完全对齐。
                raise ValueError("导出预测的候选或标签数量与验证不一致")  # 拒绝报告口径不一致的结果。
    validation["dates"] = validation_dates  # 保存轮末验证的日期范围。
    validation["max_stock_days_per_date"] = validation_limit  # 记录试跑股票数量限制。

    report = {  # 保存训练、固定样本监测和轮末验证结果。
        "config": config,  # 记录所有训练与监测间隔。
        "sources": sources,  # 保留数据及来源签名。
        "training": training,  # 包含样本加权的分段和全程 Huber。
        "label_scale_statistics": scale_stats,  # 记录各预测分钟的训练标签数、均值及原始收益率标准差。
        "price_input_scale_statistics": [] if price_estimate is None else price_estimate.statistics,  # 留下估计分时输入倍率所用的通道尺度。
        "loss_monitoring": {  # 记录用来判断损失走势的固定样本。
            "dates": monitor_dates,  # 固定监测日期，逐次一致。
            "max_stock_days_per_date": validation_limit,  # 固定监测股票范围。
            "checks": monitoring_checks,  # 保留初始、训练中和轮末的检查点结果。
        },  # 完成固定样本监测报告。
        "validation": validation,  # 保留轮末的完整或试跑验证结果。
    }  # 完成实验报告。
    if prediction_report is not None:  # 有预测文件时同时保存覆盖与误差。
        report["predictions"] = prediction_report  # 将预测文件和统计结果与模型检查点关联。
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可独立阅读的实验结果。
    print(json.dumps(validation, ensure_ascii=False), flush=True)  # 报告验证指标。
    print(f"模型与报告已写入 {args.output}", flush=True)  # 告知模型和报告的位置。


if __name__ == "__main__":  # 直接执行脚本时启动训练。
    main()  # 依次完成读批、参数更新、保存重载和验证。
