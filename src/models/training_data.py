"""把现有 Qlib 训练批次接入 PyTorch。

调用方提供读取器、可重新遍历的股票日来源和随机状态。每轮重新调用现有
训练取批器，由 DataLoader 交付形状适合一维卷积的数值张量和原始样本键。
"""

from collections.abc import (  # 声明股票日来源与批次的迭代接口。
    Callable,  # 表示可以重新获取请求或批次的函数。
    Iterable,  # 表示可逐个读取的股票日请求。
    Iterator,  # 表示逐批交付的迭代器。
)  # 完成标准库的迭代类型导入。
from dataclasses import dataclass  # 将用于训练的张量与样本定位字段保存在一起。

import numpy as np  # 声明样本键数组和股票日打乱所用的随机数生成器类型。
import torch  # 用 PyTorch 转换数值批次并维护加载器随机状态。
from torch.utils.data import (  # 复用 PyTorch 的流式数据集和加载器。
    DataLoader,  # 负责向训练循环交付数据。
    IterableDataset,  # 让训练数据按顺序生成。
)  # 完成 PyTorch 数据接口导入。

from src.data.metadata import Metadata  # 声明调用方传入的行情元数据接口。
from src.data.minutes import MinuteLibrary  # 声明调用方传入的 Qlib 读取器。
from src.features.batches import (  # 复用现有的训练生成器与默认参数。
    DEFAULT_STOCK_DAYS_PER_GROUP,  # 沿用第一版每组股票日数。
    DEFAULT_TRAIN_BATCH_SIZE,  # 沿用第一版每批样本数。
    TrainingBatch,  # 表示已有生成器组好的 NumPy 批次。
    TrainingSampleCounts,  # 接受已有的候选数量和标签覆盖统计。
    iter_training_batches,  # 继续按项目口径生成有效标签样本。
)  # 完成现有批次接口的导入。
from src.features.samples import (  # 复用采样侧的输入特征名和时钟特征构造。
    FEATURES,  # 取得已有量价通道数。
    TIME_FEATURES,  # 取得可追加的时间通道数。
    WINDOW,  # 核对输入时间轴长度。
    time_features,  # 按预测分钟生成时间通道。
)  # 完成特征接口导入。


def prepare_model_inputs(  # 在训练和验证的模型边界统一组装输入。
    x: torch.Tensor,  # 现有 float32 量价张量，形状 [批大小, 6, 30]。
    prediction_minutes: np.ndarray,  # 与特征行对齐的北京时间预测分钟整数，从午夜起算。
    input_channels: int,  # 六通道沿用量价，八通道追加已确认的两个时间特征。
) -> torch.Tensor:  # 返回位于原设备、沿通道轴排列的输入张量。
    """六通道直接返回已有张量。八通道返回追加时钟特征后的新张量。

    八通道请求的形状、类型和分钟无效，或请求的模型通道数无效时抛出 ValueError。
    """
    market_channels = len(FEATURES)  # 取得已有 OHLCVA 通道数。
    if input_channels == market_channels:  # 六通道模型继续使用原始输入。
        return x  # 避免额外转换、拼接或拷贝。
    if input_channels != market_channels + len(TIME_FEATURES):  # 只有已确认的两个时间通道可追加。
        raise ValueError("Model input channels must be 6 or 8")  # 报告不受支持的模型规模。
    if x.ndim != 3 or x.shape[1:] != (market_channels, WINDOW) or x.dtype != torch.float32:  # 核对量价布局及数值类型。
        raise ValueError("Market inputs must be float32 with shape [batch, 6, 30]")  # 防止追加到错误的轴上。
    clock = time_features(prediction_minutes)  # 使用与训练阶段无关的已知时钟生成每个窗口的特征。
    if len(clock) != len(x):  # 确认特征张量和预测分钟逐行对齐。
        raise ValueError("Market inputs and prediction minutes must have the same batch size")  # 报告输入行数不匹配。
    clock_tensor = torch.from_numpy(clock).to(device=x.device)  # 把时间特征放在量价张量所在的设备。
    return torch.cat((x, clock_tensor), dim=1)  # 在原有六个通道后按规定顺序追加时间通道。


@dataclass(frozen=True)  # 将同一批的数值张量和样本键组合成一个对象。
class TorchTrainingBatch:  # 定义 DataLoader 向模型训练循环交付的数据。
    """记录两个 CPU 单精度张量、股票代码和北京时间的预测时刻。"""

    x: torch.Tensor  # 六列基础量价的形状为 [批大小, 6, 30]，通道在分钟轴之前。
    y: torch.Tensor  # 收益率标签的形状为 [批大小]，数值均可用于训练。
    code: np.ndarray  # 每个样本对应的 Qlib 股票代码。
    t: np.ndarray  # 每个样本的预测时刻，以北京时间记到分钟。


class _TrainingDataset(IterableDataset[TorchTrainingBatch]):  # 为现有的分组批次提供 PyTorch 流式接口。
    """用每轮生成器创建能重复遍历的 PyTorch 数据集。"""

    def __init__(self, batches_for_epoch: Callable[[], Iterator[TrainingBatch]]):  # 保存每轮的取批函数。
        """batches_for_epoch 无参数，每次调用返回当前轮的新批次迭代器。"""
        super().__init__()  # 初始化 PyTorch 流式数据集。
        self._batches_for_epoch = batches_for_epoch  # 将取批延迟到训练循环开始迭代。

    def __iter__(self) -> Iterator[TorchTrainingBatch]:  # 将原有 NumPy 批次逐批转成模型输入。
        """一轮开始时创建原始迭代器，再按原顺序返回模型批次。"""
        for batch in self._batches_for_epoch():  # 按原分组、大小和组尾边界交付批次。
            yield TorchTrainingBatch(  # 把张量与该批样本定位信息一起返回。
                x=torch.from_numpy(batch.x).transpose(1, 2),  # 共享原特征数组的内存，转为一维卷积布局。
                y=torch.from_numpy(batch.y).to(dtype=torch.float32),  # 在模型边界将标签转换成单精度。
                code=batch.code,  # 按原顺序保留股票代码。
                t=batch.t,  # 保留 NumPy 的分钟时间戳及其精度。
            )  # 完成对原批次的 PyTorch 转换。


def make_training_loader(  # 为正式训练循环构造 PyTorch 数据加载器。
    library: MinuteLibrary,  # 调用方创建并维护的 Qlib 分钟行情读取器。
    metadata: Metadata,  # 调用方在加载器遍历期间保持打开的元数据读取器。
    requests_for_epoch: Callable[[], Iterable[tuple[str, str]]],  # 每轮返回股票代码和北京时间交易日的可迭代对象。
    *,  # 其余参数通过名称传入。
    rng: np.random.Generator,  # 在各轮之间持续推进的 NumPy 数据随机状态。
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,  # 每批样本数上限，默认 2,048。
    stock_days_per_group: int = DEFAULT_STOCK_DAYS_PER_GROUP,  # 每组读入的股票日数，默认 128。
    counts: TrainingSampleCounts | None = None,  # 指定时将候选数量和有效标签数量累计到传入对象。
) -> DataLoader[TorchTrainingBatch]:  # 返回可在多个轮次中重复遍历的单进程加载器。
    """每次遍历获取一轮新股票日请求，按原顺序交付训练张量与样本键。

    requests_for_epoch 可返回重复使用的股票日列表，也可生成新的迭代器。
    数据或参数错误由原训练取批器在遍历时报告。调用方负责将数值张量移到模型设备。
    """

    def batches_for_epoch() -> Iterator[TrainingBatch]:  # 为当前轮次创建原始训练批次迭代器。
        """从本轮股票日请求获取原有 NumPy 批次。"""
        return iter_training_batches(  # 调用正式样本筛选、两层打乱和组内分批流程。
            library, metadata, requests_for_epoch(),  # 每次迭代重新取得股票日来源。
            rng=rng, batch_size=batch_size,  # 应用持续推进的随机状态和批大小设置。
            stock_days_per_group=stock_days_per_group, counts=counts,  # 使用项目的分组规则和可选统计。
        )  # 保持此前关于缺失标签与组尾批次的处理。

    return DataLoader(  # 交给 PyTorch 管理训练数据迭代。
        _TrainingDataset(batches_for_epoch),  # 一条数据已经是一个完整的训练批次。
        batch_size=None,  # 直接交付预先组好的批次。
        num_workers=0,  # 第一版由主进程独享当前行情与元数据读取器。
        generator=torch.Generator(),  # 将加载器的随机状态与模型的全局随机状态隔离。
    )  # 返回训练循环可以直接迭代的加载器。
