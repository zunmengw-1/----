"""验收股票日文件缓存与原 Qlib 读取的兼容性。

在临时目录写入最小 Qlib 行情库，比较命中、回退及源行情变化时的行为。
缓存的构建和读取都通过项目实际入口执行。
"""

import os  # 设置源行情文件的修改时间。
from pathlib import Path  # 声明合成行情库和输出目录使用的路径。

import numpy as np  # 写入 Qlib 二进制字段并核对浮点精度。
import pytest  # 生成临时行情库并核对失败行为。

from src.data import minutes as minute_reader  # 调用正式读取器并观测 Qlib 文件读取。
from src.data.stock_day_cache import build_stock_day_cache  # 构建缓存。
from src.features.samples import TRADING_MINUTES  # 使用训练阶段的固定分钟网格。

STOCK = "SH600001"  # 两个日期都使用同一个 Qlib 证券代码。
FIRST_DAY = "2024-06-03"  # 只为这个日期写入缓存。
SECOND_DAY = "2024-06-04"  # 保留另一个日期用于检验回退。
FIELD = Path("features/sh600001/open.1min.bin")  # 定位实际存在的 Qlib 开盘价字段。


@pytest.fixture  # 为每项测试提供可以独立修改的行情目录。
def qlib_provider(tmp_path):  # 建立含两个日期的最小 Qlib 分钟行情库。
    """输入 pytest 临时目录，返回包含零值、缺失分钟和缺失字段的行情库。"""
    provider = tmp_path / "qlib"  # 将源行情与同级缓存输出分开存放。
    for directory in ("calendars", "instruments", FIELD.parent):  # 按 Qlib 布局创建所需目录。
        (provider / directory).mkdir(parents=True)  # 创建本场景独立的目录树。
    timestamps = [  # 写入少量连续竞价分钟，并省略两天的 09:33。
        f"{day} 09:{minute}:00"  # 按时间顺序生成全库日历的位置。
        for day in (FIRST_DAY, SECOND_DAY)  # 分两天生成日历。
        for minute in (31, 32, 34)  # 保留三个不同的分钟位置。
    ]  # 结束小型分钟日历的构造。
    (provider / "calendars/1min.txt").write_text("\n".join(timestamps), encoding="utf-8")  # 写入分钟日历。
    (provider / "instruments/all.txt").write_text(  # 声明证券的全部分钟覆盖期。
        f"{STOCK}\t{timestamps[0]}\t{timestamps[-1]}\n",  # 证券代码及首末分钟位置。
        encoding="utf-8",  # 使用读取器兼容的文本编码。
    )  # 完成证券覆盖清单。
    np.array([0, -0.0, 0.1, np.nan, 4, 5, 6], dtype="<f4").tofile(provider / FIELD)  # 首元素为日历起点，随后是开盘价。
    return provider  # 将库路径交给测试用例。


def test_stock_day_cache_preserves_exact_bars_and_falls_back(qlib_provider, tmp_path, monkeypatch):  # 核对命中和回退都可直接供训练使用。
    """输入合成行情库、临时路径和替换工具，比较读取结果及底层文件访问次数。"""
    raw_library = minute_reader.MinuteLibrary(qlib_provider)  # 构造直接访问字段文件的对照读取器。
    output = tmp_path / "cache"  # 指定尚未创建的缓存输出目录。
    first_day = raw_library.session_bars(STOCK, FIRST_DAY, TRADING_MINUTES)  # 读取包含原始缺口的完整分钟矩阵。
    second_day = raw_library.session_bars(STOCK, SECOND_DAY, TRADING_MINUTES)  # 提前取得未缓存日期的对照值。
    reordered_minutes = (574, 571, 573, 571)  # 用乱序、重复和日历中缺失的位置构造另一种网格。
    reordered = raw_library.session_bars(STOCK, FIRST_DAY, reordered_minutes)  # 保留另一种请求顺序的预期结果。
    assert build_stock_day_cache(raw_library, [(STOCK, FIRST_DAY)], TRADING_MINUTES, minute_reader.FIELDS, output) == 1  # 只缓存首个交易日。
    cached_library = minute_reader.MinuteLibrary(qlib_provider, stock_day_cache=output)  # 打开正式读取器的缓存模式。
    for _ in range(2):  # 同一缓存索引应能供训练连续读取两轮。
        assert list(cached_library.cached_stock_days()) == [(STOCK, FIRST_DAY)]  # 每轮交付缓存覆盖的规范股票日。
    original = minute_reader._read_field_slice  # 保留底层字段读取，便于统计实际访问。

    reads = []  # 保存之后每次 Qlib 字段读取的路径。

    def record_read(path, *args, **kwargs):  # 记录实际字段读取的路径并继续原操作。
        """输入字段路径与原函数参数，记录后返回原函数的结果。"""
        reads.append(path)  # 记录被访问的字段路径。
        return original(path, *args, **kwargs)  # 保持 Qlib 原有的读取结果。

    monkeypatch.setattr(minute_reader, "_read_field_slice", record_read)  # 开始观测实际的读取分支。
    cached = cached_library.session_bars(STOCK, FIRST_DAY, TRADING_MINUTES)  # 通过固定网格读取首个股票日。
    assert cached.dtype == np.float64 and cached.shape == first_day.shape  # 保持模型采样前的浮点类型和矩阵形状。
    assert np.isnan(cached[2]).all() and np.signbit(cached[0, minute_reader.OPEN])  # 对照日历缺失行和有符号零。
    np.testing.assert_array_equal(cached.view(np.uint64), first_day.view(np.uint64))  # 字节级核对全部行情和 NaN。
    assert not reads  # 缓存命中时完全跳过多字段文件读取。
    missed = cached_library.session_bars(STOCK, SECOND_DAY, TRADING_MINUTES)  # 请求未写入缓存的另一日。
    np.testing.assert_array_equal(missed, second_day)  # 回退到 Qlib 并返回正确的日期内容。
    assert len(reads) == len(minute_reader.FIELDS)  # 未缓存的股票日通过原读取器处理所有字段。
    alternate = cached_library.session_bars(STOCK, FIRST_DAY, reordered_minutes)  # 对缓存日期使用另一种有序分钟请求。
    np.testing.assert_array_equal(alternate, reordered)  # 仍满足原接口的重排与重复语义。
    assert len(reads) == 2 * len(minute_reader.FIELDS)  # 非缓存网格同样应访问原 Qlib 行情。


def test_stock_day_cache_rejects_stale_source(qlib_provider, tmp_path):  # 源字段更新后拒绝打开旧缓存。
    """输入合成行情库与临时路径，在同长度源字段改变时检查缓存的来源校验。"""
    library = minute_reader.MinuteLibrary(qlib_provider)  # 按原始字段和日历装载源行情。
    output = tmp_path / "cache"  # 为本轮提供独立的发布目录。
    build_stock_day_cache(library, [(STOCK, FIRST_DAY)], TRADING_MINUTES, minute_reader.FIELDS, output)  # 构建针对原始源文件的版本。
    source = qlib_provider / FIELD  # 找到与缓存内容相关的 Qlib 文件。
    old = source.stat()  # 保留原始文件时标。
    values = np.fromfile(source, dtype="<f4")  # 保持原文件布局、数据类型和长度。
    values[2] = 1.5  # 在原本已缓存的分钟上改一个价格。
    values.tofile(source)  # 同长度原地改写源字段。
    os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))  # 确保时标变化可被缓存校验观察到。
    with pytest.raises(ValueError):  # 对同名但已变动的数据源禁止加载旧缓存。
        minute_reader.MinuteLibrary(qlib_provider, stock_day_cache=output)  # 创建读取器时应识别到过时的源签名。


def test_stock_day_cache_failure_keeps_output_unpublished(qlib_provider, tmp_path, monkeypatch):  # 中断构建时不留下正式产物。
    """在部分股票日已读取之后注入文件错误，确认缓存尚未发布。"""
    library = minute_reader.MinuteLibrary(qlib_provider)  # 使用实际 Qlib 读取器构建第一天。
    read_bars = library.session_bars  # 保存要在注入错误前调用的正式方法。
    requests = [(STOCK, FIRST_DAY), (STOCK, SECOND_DAY)]  # 确保中断发生在第一条写入之后。
    output = tmp_path / "cache"  # 预期只在全部数据处理完之后出现。
    seen = []  # 记录构建器依次读取的交易日。

    def interrupted(code, day, minutes):  # 在第二个股票日模拟文件系统失败。
        """输入实际请求，在第二次读取抛出文件错误。"""
        seen.append(day)  # 记录本次已尝试读取的日期。
        if len(seen) == 2:  # 第一块已读入之后才制造失败。
            raise OSError("interrupted in test")  # 让构建器进入真实的失败分支。
        return read_bars(code, day, minutes)  # 第一块走实际 Qlib。

    monkeypatch.setattr(library, "session_bars", interrupted)  # 注入确定性的中途失败。
    with pytest.raises(OSError, match="interrupted in test"):  # 确认原始读取异常传至调用方。
        build_stock_day_cache(library, requests, TRADING_MINUTES, minute_reader.FIELDS, output)  # 运行真正的缓存构建和发布流程。
    assert len(seen) == 2 and not output.exists()  # 部分读取完成后正式目录仍不存在。


def test_stock_day_cache_refuses_existing_output(qlib_provider, tmp_path):  # 检查用户已有文件不会被改写。
    """输入合成行情库和临时路径，尝试向已存在目录发布缓存并检查其中的原文件。"""
    library = minute_reader.MinuteLibrary(qlib_provider)  # 构造正常的源读取器。
    output = tmp_path / "cache"  # 模拟已经被占用的输出路径。
    output.mkdir()  # 提前创建用户目录。
    original = output / "keep.txt"  # 模拟原目录中用户需要保留的文件。
    original.write_text("keep\n", encoding="utf-8")  # 写入一段可在失败后检查的内容。
    with pytest.raises((ValueError, FileExistsError)):  # 要求构建器拒绝发布到已有目录。
        build_stock_day_cache(library, [(STOCK, FIRST_DAY)], TRADING_MINUTES, minute_reader.FIELDS, output)  # 尝试占用现有路径。
    assert original.read_text(encoding="utf-8") == "keep\n"  # 确认原文件内容保持完整。
