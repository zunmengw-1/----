"""验收 Qlib 分钟读取的对齐、缓存及错误处理。

测试在临时目录构造两只股票、两个日期的行情。首先核对原始二进制数值
与缺失位置，再检查日历查找复用以及无命中位置下的文件结构校验。
"""

import numpy as np  # 写入 Qlib 使用的小端单精度数值并比较读取结果。
import pytest  # 构造临时行情夹具和参数化的错误断言。

from src.data import (  # 引入项目的 Qlib 分钟行情读取模块。
    minutes as minute_reader,  # 使用模块接口观测日期与分钟的转换。
)  # 完成正式读取模块的导入。

FIRST_DAY = "2024-06-03"  # 设置第一天的日历日期。
SECOND_DAY = "2024-06-04"  # 设置有相同时钟位置的另一个日期。
FIRST_STOCK = "SH600001"  # 使用覆盖范围较短的沪市股票。
SECOND_STOCK = "SZ000001"  # 使用覆盖完整测试日历的深市股票。
M31, M32, M33, M34, M35 = range(571, 576)  # 以北京时间零点后的分钟数表示 09:31 至 09:35。


@pytest.fixture  # 将两只股票的合成行情作为各项测试共享的构造步骤。
def tiny_provider(tmp_path):  # 按 Qlib 布局建立小型行情库。
    """输入 pytest 的临时目录路径，返回可供正式读取器使用的行情库路径。

    日历故意缺少 09:33，首只股票的覆盖期从第一天 09:32 起到第二天
    09:34。文件写入失败时由文件系统报告异常。
    """
    calendar = tmp_path / "calendars"  # 保存全库分钟时间戳。
    calendar.mkdir()  # 按 Qlib 约定创建日历目录。
    stamps = [  # 生成严格递增的跨日日历。
        f"{day} 09:{minute}:00"  # 同一日期内按时钟顺序生成分钟戳。
        for day in (FIRST_DAY, SECOND_DAY)  # 每个交易日各生成四个时点。
        for minute in (31, 32, 34, 35)  # 有意留下 09:33 的日历缺口。
    ]  # 整理完成八个全库日历位置。
    (calendar / "1min.txt").write_text("\n".join(stamps), encoding="utf-8")  # 写入分钟频率的全库交易日历。
    instruments = tmp_path / "instruments"  # 保存各证券的首末分钟。
    instruments.mkdir()  # 按 Qlib 约定创建证券目录。
    (instruments / "all.txt").write_text(  # 写入以制表符分隔的证券清单。
        f"{FIRST_STOCK}\t{stamps[1]}\t{stamps[6]}\n"  # 首只股票不覆盖日历首末分钟。
        f"{SECOND_STOCK}\t{stamps[0]}\t{stamps[-1]}\n",  # 第二只股票覆盖完整日历。
        encoding="utf-8",  # 使用与正式读取器兼容的文本编码。
    )  # 完成清单写入。
    first = tmp_path / "features" / FIRST_STOCK.lower()  # 指定首只股票的小写代码目录。
    second = tmp_path / "features" / SECOND_STOCK.lower()  # 指定第二只股票的小写代码目录。
    first.mkdir(parents=True)  # 创建首只股票的字段目录。
    second.mkdir(parents=True)  # 创建第二只股票的字段目录。
    np.array([1, 12, 14, 15, 21, 22, 24], dtype="<f4").tofile(first / "open.1min.bin")  # 写入起始位置 1 和六个开盘价。
    np.array([1, 112, np.nan, 115, 121, 122, 124], dtype="<f4").tofile(first / "close.1min.bin")  # 保留一个覆盖内记录的真实 NaN。
    np.array([0, 101, 102, 104, 105, 201, 202, 204, 205], dtype="<f4").tofile(second / "open.1min.bin")  # 写入覆盖完整日历的另一组值。
    (first / "factor.1min.bin").mkdir()  # 模拟以行情字段命名但不可作为普通文件读取的目录。
    return tmp_path  # 向各测试返回已准备好的 Qlib 根目录。


def test_minute_reader_preserves_order_spans_and_missing_data(tiny_provider):  # 核对重排、重复和各类缺失处理。
    """输入合成 Qlib 行情库路径，检查两个股票的矩阵内容；无返回值。

    如果请求顺序、缺失保留、跨日定位或数值类型改变，抛出断言异常。
    """
    library = minute_reader.MinuteLibrary(tiny_provider)  # 初始化正式的行情读取器。
    requested = (M34, M31, M33, M35, M32, M34)  # 刻意乱序，包含重复、范围外和日历缺失位置。
    result = library.session_bars(FIRST_STOCK, FIRST_DAY, requested)  # 读取首只股票第一天的字段。
    assert result.shape == (6, len(minute_reader.FIELDS)) and result.dtype == np.float64  # 保持读取器公开的行列和数值类型。
    assert library.has_coverage(FIRST_STOCK, FIRST_DAY)  # 按覆盖期确认首只股票当天在库中。
    np.testing.assert_array_equal(result[:, minute_reader.OPEN], [14, np.nan, np.nan, 15, 12, 14])  # 对齐乱序和重复的价格位置。
    np.testing.assert_array_equal(result[:, minute_reader.CLOSE], [np.nan, np.nan, np.nan, 115, 112, np.nan])  # 保留二进制文件中的原始 NaN。
    assert np.isnan(result[:, minute_reader.FACTOR]).all()  # 对应路径是目录，应当按缺失字段保留 NaN。
    assert np.isnan(result[:, minute_reader.AMOUNT]).all()  # 成交额文件不存在，也应当保留 NaN。
    next_result = library.session_bars(FIRST_STOCK, SECOND_DAY, requested)  # 用同一序列读取另一交易日。
    np.testing.assert_array_equal(next_result[:, minute_reader.OPEN], [24, 21, np.nan, np.nan, 22, 24])  # 核对覆盖期结束之后及日历缺口的位置。
    other = library.session_bars(SECOND_STOCK, FIRST_DAY, requested)  # 读取同日期同序列下的另一只股票。
    np.testing.assert_array_equal(other[:, minute_reader.OPEN], [104, 101, np.nan, 105, 102, 104])  # 同一个时间位置仍取自不同股票的文件。


def test_minute_reader_reuses_only_matching_calendar_request(tiny_provider, monkeypatch):  # 确认日历定位按日期和请求顺序复用。
    """输入合成行情库和 pytest 的临时替换工具，观测同日期同序列的复用；无返回值。

    重复转换时钟，或改换日期与顺序后仍使用不适用的日期位置时，抛出断言异常。
    """
    calls = []  # 保留发生的每一次分钟到时钟文本转换。
    convert = minute_reader.clock_text  # 保存真实时钟转换函数，以便在记录后原样调用。

    def record_clock(minute):  # 包装时钟转换并记录所收到的指令。
        """输入当天零点后的整数分钟数，记录后返回时钟文本。

        输入无效时由正式时钟转换函数报告原有异常。
        """
        calls.append(minute)  # 追加当前转换请求。
        return convert(minute)  # 保持正式的时钟文本结果。

    monkeypatch.setattr(minute_reader, "clock_text", record_clock)  # 在读库入口前开始观测转换。
    library = minute_reader.MinuteLibrary(tiny_provider)  # 初始化一个独立的新读取器。
    requested = [M32, M34]  # 使用列表作为首个日期的原始请求。
    first = library.session_bars(FIRST_STOCK, FIRST_DAY, requested)  # 首次解析日期和序列。
    count_after_first = len(calls)  # 记录处理首个请求后已发生的工作。
    second = library.session_bars(SECOND_STOCK, FIRST_DAY, tuple(requested))  # 相同内容改用元组并换股票。
    assert len(calls) == count_after_first  # 同日期同序列应复用日历定位，避免重复转换。
    np.testing.assert_array_equal(first[:, minute_reader.OPEN], [12, 14])  # 首只股票利用共享定位读取自己的价格。
    np.testing.assert_array_equal(second[:, minute_reader.OPEN], [102, 104])  # 另一只股票不应混用原股票的价格。
    requested.reverse()  # 原地调换同一个列表里的顺序，验证缓存依据内容。
    reordered = library.session_bars(SECOND_STOCK, FIRST_DAY, requested)  # 在同一天按反序查询。
    np.testing.assert_array_equal(reordered[:, minute_reader.OPEN], [104, 102])  # 返回顺序按当前请求对齐。
    another_date = library.session_bars(SECOND_STOCK, SECOND_DAY, requested)  # 换日期但保留反序请求。
    np.testing.assert_array_equal(another_date[:, minute_reader.OPEN], [204, 202])  # 不得把上一交易日的日历索引用于新日期。


@pytest.mark.parametrize("requested", ((), (M31,)))  # 覆盖空序列和只有证券范围外位置的请求。
@pytest.mark.parametrize(  # 分别核对原有的两类字段结构错误。
    ("damage", "expected"),  # 为各场景提供文件变更方式和异常消息关键词。
    (("length", "field length"), ("header", "field start index")),  # 验证错误长度和错误的起点索引。
)  # 完成文件损坏场景的参数化。
def test_minute_reader_validates_file_even_without_matching_rows(tiny_provider, requested, damage, expected):  # 检查无命中请求仍核对文件结构。
    """输入临时行情库、无命中分钟序列、损坏方式和应匹配的提示；无返回值。

    读取错误文件时必须抛出带指定提示的 ValueError，否则测试失败。
    """
    path = tiny_provider / "features" / FIRST_STOCK.lower() / "open.1min.bin"  # 定位首只股票的正常开盘价文件。
    if damage == "length":  # 制造字段字节数错误。
        path.write_bytes(path.read_bytes()[:-1])  # 删掉一个字节，防止只比较整浮点数的数量。
    else:  # 文件大小保持正常，只破坏文件头的日历起点。
        contents = np.fromfile(path, dtype="<f4")  # 完整读出原始测试文件。
        contents[0] = 0  # 在文件头写入与证券覆盖范围不一致的起始位置。
        contents.tofile(path)  # 用保持相同字节数的内容覆盖原文件。
    library = minute_reader.MinuteLibrary(tiny_provider)  # 用正式读取器加载目录和证券范围。
    with pytest.raises(ValueError, match=expected):  # 无行情行可读时也要报告字段结构错误。
        library.session_bars(FIRST_STOCK, FIRST_DAY, requested)  # 访问损坏字段并触发原有校验。
