"""用合成的连续竞价分钟行情验收训练批次。

测试输入由两个训练股票日及验证、测试各一个股票日组成。
依次检查标签覆盖、两层打乱与分组内存释放，测试结果由 pytest 报告。
"""

import weakref  # 导入弱引用工具，观察原始窗口的生存期。
from decimal import Decimal  # 导入十进制数类型，精确表示前收盘价。
from types import SimpleNamespace  # 导入轻量对象类型，用于提供行情和状态接口。

import numpy as np  # 导入数组工具，用于构造分钟行情和验证批次。
import pytest  # 导入 pytest，提供测试夹具及比较工具。

from src.features import batches  # 导入训练批次模块，以便在测试中观察其调用。
from src.features.samples import (  # 导入量价列索引及连续竞价分钟网格。
    AMOUNT,  # 成交额列的索引。
    TRADING_MINUTES,  # 正式样本使用的全部连续竞价分钟。
    VOLUME,  # 成交量列的索引。
)  # 完成样本字段与时间网格导入。

TRAIN_DAYS = [("SH600001", "2024-01-02"), ("SZ000001", "2024-01-03")]  # 定义两个彼此独立的训练股票日。
HELD_OUT_DAYS = [("SH600001", "2025-01-02"), ("SH600001", "2025-07-01")]  # 定义用于确认不会混入训练的验证和测试股票日。


@pytest.fixture  # 为每项测试提供新建的行情接口、状态接口和读取记录。
def training_fixture():  # 创建零成交分钟影响两个事后标签的合成股票日数据。
    """无需外部输入，返回行情接口、状态接口与已读取股票日的列表。

    行情接口收到未配置的股票日时抛出断言异常，便于发现越界读取。
    """
    stock_days = set(TRAIN_DAYS + HELD_OUT_DAYS)  # 汇总本夹具允许读取的股票日。
    reads = []  # 记录实际读取的股票日，从而检测读入顺序及数据泄漏。

    def session_bars(code, day, minutes):  # 按股票代码、交易日和所需分钟读取合成行情。
        """输入证券代码、交易日和分钟序列，返回相应的七列行情数组。

        股票日超出夹具范围时抛出断言异常。
        """
        assert (code, day) in stock_days  # 遇到未定义的股票日立即报错。
        reads.append((code, day))  # 记录股票日，供训练集隔离的测试比对。
        rows = np.tile([10.0, 10.0, 10.0, 10.0, 100.0, 1000.0, 1.0], (len(minutes), 1))  # 生成价格稳定且有成交的分钟线。
        if (code, day) == TRAIN_DAYS[0]:  # 在第一个训练股票日的 10:05 制造一个零成交缺口。
            rows[34, [VOLUME, AMOUNT]] = 0.0  # 该分钟不能作预测端，也使 10:00 的退出和 10:04 的买入标签无效。
        return rows  # 返回所需分钟的行情数组。

    library = SimpleNamespace(  # 构造读取器访问的最小行情库接口。
        dates={day for _, day in stock_days},  # 暴露已定义的 Qlib 交易日。
        has_coverage=lambda code, day: (code, day) in stock_days,  # 提供股票日在行情库中的覆盖状态。
        session_bars=session_bars,  # 用夹具的行情函数完成实际读取。
    )  # 完成合成行情库接口。
    daily = SimpleNamespace(  # 构造一个普通主板股票日的交易状态。
        preclose=Decimal("10.00"),  # 提供以元计的前收盘参考价。
        st=False,  # 设置为正常状态，避免风险警示筛选。
        trading=True,  # 标记当天正常交易。
        ipo=False,  # 标记当天不在上市初期排除期。
        delist=False,  # 标记当天不在退市整理期。
        full_halt=False,  # 标记当天没有全天停牌。
        intraday_halts=(),  # 设置为空，便于单独验证目标端零成交。
    )  # 完成合成的股票日状态。
    metadata = SimpleNamespace(  # 提供训练批次构造所需的元数据接口。
        canonical_code=lambda code, day: code,  # 保留输入代码，夹具中没有代码变更。
        historical_code=lambda code, day: code,  # 令历史代码与规范代码一致。
        is_mainboard=lambda code, day: True,  # 将两个股票都视为普通主板股票。
        day=lambda code, day: daily,  # 返回合成的日线交易状态。
    )  # 完成合成元数据接口。
    return library, metadata, reads  # 向测试返回接口与读取记录。


def test_training_counts_keep_missing_labels_out_of_batches(training_fixture):  # 验证候选覆盖计数以及训练和留出集的边界。
    """输入合成行情夹具，验证标签覆盖及训练批次内容；无返回值。

    统计口径、窗口尺寸或有效样本内容不符时抛出断言异常。
    """
    library, metadata, reads = training_fixture  # 取出行情、元数据接口和读取记录。
    counts = batches.TrainingSampleCounts()  # 创建独立的覆盖率计数器。
    result = list(batches.iter_training_batches(  # 消费一个完整训练轮次，计入全部训练组。
        library, metadata, TRAIN_DAYS + HELD_OUT_DAYS,  # 同时传入训练、验证和测试请求以检查隔离。
        batch_size=64,  # 使用不能整除总样本数的大小，检查最后一批也被保留。
        rng=np.random.default_rng(7),  # 固定随机数种子，方便复现。
        stock_days_per_group=2,  # 将两个训练股票日合在同一组内。
        counts=counts,  # 接收有效与无效标签的计数。
    ))  # 完整读取迭代器后检查计数。
    assert sorted(reads) == sorted(TRAIN_DAYS)  # 确认验证和测试股票日没有被读取。
    assert counts.candidates == 337  # 两天共 338 个时间候选，零成交预测时点排除一个。
    assert counts.labels_available == 335  # 两个事后端点各有一个候选遇到零成交，分母仍包括这两条候选。
    assert [len(batch.y) for batch in result] == [64, 64, 64, 64, 64, 15]  # 验证最后一批的十五条训练样本未遗漏。
    assert all(batch.x.shape == (len(batch.y), 30, 6) for batch in result)  # 确认 30 根分钟线和六个特征完整入批。
    assert all(np.all(batch.y == 0.0) for batch in result)  # 有效标签在价格不变时应为零，不得包含无效值。
    keys = {(code, str(timestamp)) for batch in result for code, timestamp in zip(batch.code, batch.t)}  # 收集所有已送入训练的样本主键。
    assert len(keys) == counts.labels_available  # 同一股票同一时点最多训练一次。
    assert ("SH600001", "2024-01-02T10:00") not in keys  # 退出分钟无成交，候选不参与训练。
    assert ("SH600001", "2024-01-02T10:04") not in keys  # 买入分钟无成交，候选不参与训练。
    assert ("SH600001", "2024-01-02T10:05") not in keys  # 预测时点无成交，因此不形成候选。
    assert ("SH600001", "2024-01-02T10:01") in keys  # 相邻时间的有效候选仍可进入训练。


def test_training_defaults_to_2048_samples_and_keeps_tail(training_fixture, monkeypatch):  # 通过正式取批接口核对大批次默认值。
    """接收合成行情夹具和临时替换工具，使用正常行情组装 13 个训练股票日。

    默认批大小错误或尾批样本遗漏时触发断言异常。
    """
    library, metadata, _ = training_fixture  # 使用已有的行情和元数据夹具。
    _, day = TRAIN_DAYS[1]  # 选取行情完整且无停牌的训练交易日。
    requests = [(f"SH{number:06d}", day) for number in range(600010, 600023)]  # 让同一组有 13 个不同股票日。
    bars = library.session_bars(*TRAIN_DAYS[1], TRADING_MINUTES)  # 复用夹具中正常交易日的连续行情。
    monkeypatch.setattr(library, "has_coverage", lambda code, date: (code, date) in requests)  # 仅允许本次 13 个股票日。
    monkeypatch.setattr(library, "session_bars", lambda code, date, minutes: bars)  # 为每只股票提供相同的健康行情。
    iterator = batches.iter_training_batches(  # 沿用默认批大小和默认股票日组大小。
        library, metadata, requests, rng=np.random.default_rng(7),  # 固定打乱用的随机数种子。
    )  # 由正式取批入口生产首批与尾批。
    assert [len(batch.y) for batch in iterator] == [2_048, 149]  # 13 × 169 条样本分成 2,048 条和 149 条。


def test_training_shuffle_is_reproducible_by_group(training_fixture):  # 验证股票日和组内样本分别打乱且可以复现。
    """输入合成行情夹具，验证两层随机顺序及股票日分组；无返回值。

    同种子无法复现、分组越界或不同种子无法改变顺序时抛出断言异常。
    """
    library, metadata, reads = training_fixture  # 获取合成行情、元数据与读入顺序记录。

    def run_epoch(seed):  # 按给定种子产生一轮批次以及相应的真实读取顺序。
        """输入整数种子，返回股票日读取顺序和样本主键顺序。

        合成行情收到非训练请求时在后续断言中报错。
        """
        reads.clear()  # 清除上一轮读取记录，单独观察本轮顺序。
        result = list(batches.iter_training_batches(  # 消费本轮全部训练批次。
            library, metadata, TRAIN_DAYS + HELD_OUT_DAYS,  # 用相同的训练、验证和测试请求测试种子差异。
            batch_size=500,  # 让每个股票日的样本汇成一批，便于验证组边界。
            rng=np.random.default_rng(seed),  # 用当前种子创建该轮独立的随机数发生器。
            stock_days_per_group=1,  # 每组只允许包含一个股票日。
        ))  # 完成本轮读取。
        assert len(result) == 2  # 一个股票日发出一个批次，验证与测试不产生批次。
        sample_order = []  # 逐组记录供比对的样本主键顺序。
        for batch, (code, day) in zip(result, reads, strict=True):  # 将各批次与该组唯一的股票日对应起来。
            assert np.all(batch.code == code)  # 确认批次没有跨股票分组。
            day_in_batch = batch.t.astype("datetime64[D]")  # 去掉分钟部分，只检查交易日。
            assert np.all(day_in_batch == np.datetime64(day))  # 确认批次没有跨交易日分组。
            assert np.any(batch.t[1:] < batch.t[:-1])  # 组内样本已打乱，时间顺序至少出现一处倒序。
            sample_order.extend(zip(batch.code.tolist(), batch.t.tolist()))  # 保留真实批次内的样本顺序。
        return reads.copy(), sample_order  # 返回本轮读取顺序和样本顺序的快照。

    first_reads, first_samples = run_epoch(9)  # 以固定种子生成第一轮基准。
    assert run_epoch(9) == (first_reads, first_samples)  # 相同种子必须复现两层顺序。
    other_results = [run_epoch(seed) for seed in range(10, 17)]  # 试用多组种子，避免两元素随机排列偶然重合。
    assert any(order != first_reads for order, _ in other_results)  # 至少一个种子应改变股票日读取顺序。
    assert any(order == first_reads and samples != first_samples for order, samples in other_results)  # 即使股票日顺序相同，组内样本仍能独立打乱。


def test_training_releases_original_windows_before_next_group(training_fixture, monkeypatch):  # 验证开始读取下一组前，上一组的原始窗口已释放。
    """输入合成行情夹具和临时替换工具，观察批次迭代期间的窗口引用；无返回值。

    若当前组读取时仍有上一组窗口，或提前读入下一组，则抛出断言异常。
    """
    library, metadata, reads = training_fixture  # 取出用于训练批次的模拟数据对象。
    sessions = batches.stock_day_sessions  # 保存真实的股票日读取函数，在观测期间继续走生产路径。
    sample_session = batches.sample_session  # 保存真实的窗口构造函数以生成正常训练样本。
    group = 0  # 标记正在读取的组号，组大小固定为一个股票日。
    first_arrays = []  # 弱引用第一组各时段的特征数组，不延长这些数组的生存期。
    live_at_boundary = []  # 记录开始读取第二组时仍存活的第一组窗口数。

    def observe_group(*args, **kwargs):  # 在读取下一股票日之前记录上一组窗口状态。
        """输入股票日读取函数的参数，返回原函数的时段迭代器。

        原函数的读取异常会原样传出。
        """
        nonlocal group  # 更新外围测试所观察的组号。
        if group == 1:  # 只在第一个组切换到第二个组的边界检查。
            live_at_boundary.append(sum(ref() is not None for ref in first_arrays))  # 计算仍有强引用的旧特征数组数。
        group += 1  # 标记即将读取的组。
        return sessions(*args, **kwargs)  # 将实际行情读取继续委托给生产函数。

    def observe_samples(*args, **kwargs):  # 记录时段样本构造函数交付的特征数组。
        """输入样本构造函数的参数，返回原函数的时段样本数组。

        原函数的样本构造异常会原样传出。
        """
        session = sample_session(*args, **kwargs)  # 通过真实采样函数构造整个时段的样本。
        if group == 1 and session.label_valid.any():  # 追踪第一组含有有效训练样本的时段。
            first_arrays.append(weakref.ref(session.x))  # 建立不会阻止垃圾回收的弱引用。
        return session  # 将原始时段结果继续交给训练批次迭代器。

    monkeypatch.setattr(batches, "stock_day_sessions", observe_group)  # 临时添加股票日边界观测。
    monkeypatch.setattr(batches, "sample_session", observe_samples)  # 临时添加窗口弱引用观测。
    iterator = batches.iter_training_batches(  # 构造会产生两个训练组的迭代器。
        library, metadata, TRAIN_DAYS,  # 每组恰好读取一个训练股票日。
        batch_size=500,  # 让组尾批次覆盖全部原始窗口，便于复现之前的内存滞留。
        rng=np.random.default_rng(9),  # 固定组的读取顺序。
        stock_days_per_group=1,  # 保证两次股票日读取正好跨越分组边界。
    )  # 仅构造迭代器，此时不应读取行情。
    assert reads == []  # 延迟到第一次索取批次时才读入组内行情。
    first_batch = next(iterator)  # 读取第一组训练样本。
    assert len(reads) == 1 and len(first_arrays) == 2  # 只读第一个股票日并记录其上午和下午的特征数组。
    next(iterator)  # 调用者继续持有第一批，同时让迭代器开始读取下一组。
    iterator.close()  # 结束本次局部轮次，释放迭代器资源。
    assert len(first_batch.y) > 0  # 第一批有可用标签，且调用者仍持有它。
    assert len(reads) == 2 and live_at_boundary == [0]  # 下一组开始时第一组的原时段特征数组均已释放。


def test_pytorch_loader_matches_two_training_epochs(training_fixture):  # 核对接入 PyTorch 后两轮的样本和组尾。
    """使用现有行情夹具比对原迭代器和 DataLoader 的特征、标签、键与统计。

    两轮的样本内容、批次边界、类型或随机状态异常时触发断言异常。
    """
    import torch  # 比较输出张量并读取 PyTorch 随机状态。

    from src.models.training_data import (
        make_training_loader,  # 使用正式的 PyTorch 取批入口。
    )  # 完成训练接口导入。

    library, metadata, _ = training_fixture  # 取出测试用行情与元数据读取器。
    requests = TRAIN_DAYS + HELD_OUT_DAYS  # 测试传入留出期时仍只取训练期。
    seed = 7  # 两个入口使用同一个起始种子。
    source_rng = np.random.default_rng(seed)  # 保存原迭代器跨轮推进的随机状态。
    source_counts = batches.TrainingSampleCounts()  # 累计原迭代器的候选统计。
    loader_counts = batches.TrainingSampleCounts()  # 累计 PyTorch 入口的候选统计。
    loader = make_training_loader(  # 构造重复迭代的正式 DataLoader。
        library, metadata, lambda: iter(requests),  # 模拟每轮重新打开的流式股票日清单。
        rng=np.random.default_rng(seed), counts=loader_counts,  # 注入独立但可复现的数据随机状态与统计。
        batch_size=128, stock_days_per_group=1,  # 让两个股票日分别产生尾批。
    )  # 完成 PyTorch 数据入口构造。
    torch_rng = torch.random.get_rng_state()  # 在迭代前保存模型侧的全局随机状态。
    orders = []  # 检查每轮产生的完整样本键顺序。
    for _ in range(2):  # 同一个 DataLoader 必须能连续跑两轮。
        expected = batches.iter_training_batches(  # 用原 NumPy 接口提供独立参考。
            library, metadata, requests, rng=source_rng, counts=source_counts,  # 原接口连续推进随机状态。
            batch_size=128, stock_days_per_group=1,  # 使用与 DataLoader 相同的组和批大小。
        )  # 构造当前轮的对照批次。
        seen = []  # 汇总当前轮出现过的批次长度和样本键。
        for actual, raw in zip(loader, expected, strict=True):  # 按相同顺序比对全部批次及组尾。
            assert actual.x.shape == (len(raw.y), 6, 30)  # PyTorch 一维卷积使用通道在前的布局。
            assert actual.x.dtype == actual.y.dtype == torch.float32  # 输入和标签可直接用于单精度模型。
            torch.testing.assert_close(actual.x, torch.from_numpy(raw.x).transpose(1, 2), rtol=0, atol=0)  # 比对全部特征值。
            torch.testing.assert_close(actual.y, torch.from_numpy(raw.y).float(), rtol=0, atol=0)  # 比对转换成单精度后的标签。
            np.testing.assert_array_equal(actual.code, raw.code)  # 股票代码保持原顺序。
            np.testing.assert_array_equal(actual.t, raw.t)  # 预测分钟保持原精度与顺序。
            seen.append((len(actual.y), tuple(zip(actual.code, actual.t))))  # 记录批次和键，供跨轮比较。
        assert sorted(size for size, _ in seen) == [38, 41, 128, 128]  # 两个股票日的尾批分别为三十八条和四十一条。
        assert loader_counts == source_counts  # 标签覆盖分母和分子与原接口一致。
        orders.append(seen)  # 保存完整的一轮供第二轮对照。
    assert orders[0] != orders[1]  # 同一数据加载器的下一轮继续打乱。
    assert loader_counts.labels_available == 670  # 两轮都交付了各自全部的三百三十五条有效样本。
    assert torch.equal(torch_rng, torch.random.get_rng_state())  # DataLoader 的迭代不消耗模型全局随机状态。


def test_return_tcn_accepts_training_loader_batch(training_fixture):  # 验证正式训练批次能直接送入 TCN。
    """输入合成分钟行情夹具，检查表示、收益率输出和反向传播；无返回值。

    形状不符合模型接口，或梯度为空或非有限值时触发断言异常。
    """
    import torch  # 使用 PyTorch 检查张量并计算用于试验的回归损失。

    from src.models.tcn import ReturnTCN  # 使用正式 TCN 模型。
    from src.models.training_data import (  # 从正式训练数据模块导入加载器。
        make_training_loader,  # 复用训练数据的正式加载入口。
    )  # 完成训练加载器导入。

    library, metadata, _ = training_fixture  # 取得构造训练批次所需的合成行情和股票日状态。
    loader = make_training_loader(  # 为一个股票日构造小批量数据加载器。
        library,  # 由夹具提供合成分钟行情。
        metadata,  # 由夹具提供股票日状态。
        lambda: iter(TRAIN_DAYS[:1]),  # 只读取首个训练股票日，以缩小测试范围。
        rng=np.random.default_rng(7),  # 固定数据打乱的种子。
        batch_size=2,  # 取两条样本，便于同时验证批维和单样本维度。
    )  # 完成正式训练数据加载器构造。
    batch = next(iter(loader))  # 取得一批通道在前的特征及对应收益率标签。
    model = ReturnTCN()  # 按默认的六通道输入构建 TCN。
    assert model.encode(batch.x).shape == (2, 16)  # 核对每条样本得到十六维时序表示。
    prediction = model(batch.x)  # 将加载器提供的输入直接送入回归模型。
    assert prediction.shape == batch.y.shape == (2,)  # 每条样本只需一个标量预测，与标签形状一致。
    assert prediction.dtype == batch.y.dtype == torch.float32  # 预测与标签均使用单精度。
    assert model(batch.x[:1]).shape == (1,)  # 批次只有一条样本时仍保留批维。
    loss = torch.nn.functional.mse_loss(prediction, batch.y)  # 用临时均方误差检查计算图连通。
    assert torch.isfinite(loss)  # 检查预测和标签计算出的损失可用于反向传播。
    loss.backward()  # 验证误差可以沿模型计算图回传。
    assert all(  # 验证模型的可训练参数均收到有限梯度。
        parameter.grad is not None and torch.isfinite(parameter.grad).all()  # 检查当前参数的梯度。
        for parameter in model.parameters()  # 遍历模型保存的参数。
        if parameter.requires_grad  # 只检查参与反向传播的参数。
    )  # 完成梯度检查。


@pytest.mark.parametrize(  # 用缺批维、通道轴错位和回看分钟不足的输入验证模型边界。
    "shape",  # 传给测试函数的输入形状。
    [(6, 30), (2, 30, 6), (2, 6, 29)],  # 三种不符合 [批大小, 6, 30] 的排列。
)  # 完成输入形状参数化。
def test_return_tcn_rejects_invalid_input_shape(shape):  # 检查无效形状得到明确的异常。
    """输入无效张量形状，要求模型抛出 ValueError；无返回值。"""
    import torch  # 使用 PyTorch 创建指定形状的单精度张量。

    from src.models.tcn import ReturnTCN  # 使用正式 TCN 模型。

    model = ReturnTCN()  # 创建使用默认输入规格的模型。
    with pytest.raises(ValueError):  # 检查模型提前拒绝不符合规格的张量。
        model(torch.zeros(shape, dtype=torch.float32))  # 将无效形状的张量交给预测接口。
