"""按股票和时段读取 Qlib 分钟行情。

输入为 Qlib 行情库目录、证券代码、北京时间交易日与分钟序列。先读取全库日历和
各股票覆盖范围，再按分钟索引读取二进制字段，返回保留缺失位置的行情矩阵。
"""

from errno import EBADF, ELOOP, ENOENT, ENOTDIR  # 识别缺失或无法定位的字段路径。
from stat import S_ISREG  # 检查字段是否为可读取的普通文件。

# 用于读取行情二进制文件并对齐成数值矩阵。
import numpy as np

from src.data.stock_day_cache import StockDayCache  # 可选地加载按股票日排列的连续行情。

# 保存行情矩阵的七列字段名及其固定顺序。
FIELDS = ("open", "high", "low", "close", "volume", "amount", "factor")
# 赋予每个字段对应的列索引，供读取和后续特征计算共用。
OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, FACTOR = range(len(FIELDS))
_FIELD_DTYPE = np.dtype("<f4")  # Qlib 文件头和行情值均为小端 32 位浮点数。
_CACHE_LIMIT = 1024  # 限制每个读取器缓存的日期和分钟序列数量。



# 将从当日零点累计的分钟数转换为时钟文本。
def clock_text(minute):
    """将北京时间的分钟偏移量转为 `HH:MM` 字符串。

    minute 为当天零点起累计的整数分钟数。返回补齐前导零的小时和分钟。
    参数不支持整数运算时会抛出 TypeError。
    """
    # 除以 60 得到小时，取余数得到分钟，两项均显示为两位。
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _read_field_slice(path, first_index, expected_bytes, byte_offset, byte_length):  # 读取一个字段的头部和所需区间。
    """接收字段路径、全库起始索引、预期文件字节数、读取起点字节及读取长度。

    返回小端 float32 数组。缺失或非普通文件返回 None，长度或文件头不符时抛出 ValueError。
    无覆盖分钟时读取长度为零，仍校验已有文件的头部与长度。
    """
    try:  # 查询路径类型和长度，避免尝试打开目录或管道作为行情文件。
        info = path.stat()  # 允许普通文件或指向它的符号链接。
    except OSError as error:  # 判断字段路径缺失或无效时的处理方式。
        if error.errno in (ENOENT, ENOTDIR, EBADF, ELOOP):  # 与原来的文件存在性检查对齐。
            return None  # 此字段缺失，由调用方保留整列 NaN。
        raise  # 其他访问错误仍交给调用方处理。
    if not S_ISREG(info.st_mode):  # 非普通文件不参与分钟行情读取。
        return None  # 目录、管道等路径按缺失字段处理。
    if info.st_size != expected_bytes:  # 核对包含文件头的完整字段长度。
        raise ValueError(f"Unexpected Qlib field length in {path}")  # 保留原有异常提示。
    with path.open("rb", buffering=0) as stream:  # 每个字段打开一次并按字节精确读取。
        header = stream.read(_FIELD_DTYPE.itemsize)  # 先读取记录证券起点的文件头。
        if len(header) != _FIELD_DTYPE.itemsize:  # 读取中发现文件头不完整。
            raise ValueError(f"Unexpected Qlib field length in {path}")  # 截断文件按长度异常报告。
        if np.frombuffer(header, dtype=_FIELD_DTYPE)[0] != first_index:  # 校验索引与覆盖清单一致。
            raise ValueError(f"Unexpected Qlib field start index in {path}")  # 报告无法对齐的起点。
        stream.seek(byte_offset)  # 跳到请求的首个行情值。
        payload = stream.read(byte_length)  # 读取能覆盖本次请求的最小连续区间。
    if len(payload) != byte_length:  # 检查读取期间文件是否缺少所需内容。
        raise ValueError(f"Unexpected Qlib field length in {path}")  # 避免使用截断片段。
    return np.frombuffer(payload, dtype=_FIELD_DTYPE)  # 按既有二进制格式解释所需数据。


# 管理 Qlib 分钟日历、证券覆盖范围和字段读取。
class MinuteLibrary:
    """为调用方提供股票覆盖判断和按时段读取分钟行情的接口。"""

    # 从行情库建立日历索引与证券覆盖范围。
    def __init__(self, provider, *, stock_day_cache=None):  # stock_day_cache 为可选的缓存目录。
        """读取 Qlib 日历和证券覆盖清单，初始化查询所需的索引。

        provider 为 Qlib 行情库根目录 Path。stock_day_cache 为可选缓存目录 Path。
        初始化不返回值。缓存来源不符、日历重复或乱序、
        证券重复或清单格式无效时抛出 ValueError，文件读取失败时抛出 OSError。
        """
        # 保存行情库根目录供后续定位分钟字段文件。
        self.provider = provider
        # 从日历文件逐行读取全库分钟时间戳。
        calendar = (provider / "calendars/1min.txt").read_text().splitlines()
        # 确认时间戳严格递增，避免同一时刻映射到多个位置。
        if calendar != sorted(set(calendar)):
            # 重复或无序的日历无法用于对齐二进制字段。
            raise ValueError("Qlib minute calendar must be unique and chronological")
        # 为每个时间戳建立其在日历中的从零起算位置。
        self.positions = {stamp: index for index, stamp in enumerate(calendar)}
        # 从分钟时间戳提取交易日，供调用者检查是否在日历内。
        self.dates = {stamp[:10] for stamp in calendar}
        self._minute_cache = {}  # 在读取器生命周期内复用日期和对应的分钟位置。
        # 以证券代码为键保存其首末分钟时间戳。
        self.spans = {}
        # 逐行读取全部证券及其行情覆盖范围。
        for line in (provider / "instruments/all.txt").read_text().splitlines():
            # 证券清单每行按制表符分为代码、起点和终点。
            code, first, last = line.split("\t")
            # 每个证券只能声明一次覆盖范围。
            if code in self.spans:
                # 阻止后出现的证券范围覆盖同一代码的既有范围。
                raise ValueError(f"Duplicate Qlib instrument {code}")
            # 登记证券的两端时间戳，保留原始分钟精度。
            self.spans[code] = (first, last)
        self._stock_day_cache = (
            StockDayCache(stock_day_cache, provider, FIELDS)  # 校验后加载指定的索引和行情映射。
            if stock_day_cache is not None else None  # 省略参数时沿用原始 Qlib 读取。
        )  # 保存会在日内行情读取前查询的缓存。

    # 判断指定交易日是否位于该证券的行情覆盖期。
    def cached_stock_days(self):  # 为训练入口提供可以每轮重新遍历的缓存清单。
        """返回规范股票代码和北京时间 ISO 日期的迭代器。未配置缓存时抛出 ValueError。"""
        if self._stock_day_cache is None:  # 检查调用方是否指定过股票日缓存目录。
            raise ValueError("No stock-day cache configured")  # 提示使用缓存索引前先加载缓存。
        return self._stock_day_cache.stock_days()  # 每次调用取得一个新迭代器。

    def has_coverage(self, code, day):
        """判断 Qlib 证券代码 code 在日期 day 是否处于覆盖范围。

        day 为北京时间的 YYYY-MM-DD 字符串。返回布尔值，首末日均计入。
        清单中没有该代码时返回 False。参数类型无法比较时可能抛出 TypeError。
        """
        # 获取指定证券的首末时间戳，未收录时得到 None。
        span = self.spans.get(code)
        # 比较首末时间戳的日期部分，避免把日内分钟混入日期判断。
        return span is not None and span[0][:10] <= day <= span[1][:10]

    def _positions_for_minutes(self, day, minutes):  # 取得同一日期和同一请求顺序共用的日历位置。
        """接收北京时间日期和分钟序列，返回按请求顺序排列的全库索引。

        日历中不存在的分钟返回 -1。索引缓存由当前读取器持有。
        """
        requested = tuple(minutes)  # 固定分钟内容和顺序，使列表和元组可以命中同一结果。
        key = (day, requested)  # 日期和分钟序列共同区分每种请求。
        cached = self._minute_cache.get(key)  # 查找已经计算的日历位置。
        if cached is not None:  # 同一个请求无需再次格式化时间戳。
            return cached  # 返回只用于读取的索引数组。
        positions = np.array(  # 首次请求时按顺序查找每个分钟的全库位置。
            [self.positions.get(f"{day} {clock_text(minute)}:00", -1) for minute in requested],  # 保留缺失分钟。
            dtype=np.intp,  # 使用 NumPy 数组索引的整数类型。
        )  # 完成分钟索引构建。
        if len(self._minute_cache) < _CACHE_LIMIT and len(requested) <= 24 * 60:  # 仅缓存日内常用长度的请求。
            self._minute_cache[key] = positions  # 留给当日其他证券重复使用。
        return positions  # 返回真实的请求位置与缺失标记。

    # 读取一个证券在指定交易日的一组分钟。
    def session_bars(self, code, day, minutes):
        """将请求的分钟顺序对齐到 Qlib 行情字段。

        code 为 Qlib 证券代码，day 为北京时间日期字符串。minutes 是可取长度
        的分钟序列，元素为从当日零点起累计的整数分钟。返回形状为
        (分钟数, 7) 的 NumPy 矩阵，字段缺失与范围外位置保留 NaN。
        已指定缓存且股票日和分钟表命中时，从缓存读取。
        证券或覆盖边界不存在时抛出 KeyError，字段文件结构错误时抛出
        ValueError，文件访问失败时抛出 OSError。
        """
        # 取出证券覆盖范围的首末分钟时间戳。
        first, last = self.spans[code]
        if self._stock_day_cache is not None:  # 仅配置缓存时查询该股票日。
            cached = self._stock_day_cache.read(code, day, minutes)  # 读取与请求分钟表一致的缓存块。
            if cached is not None:  # 判断该股票日是否包含在已建缓存中。
                return cached  # 保持原读取器的浮点精度和缺失分钟表示。
        # 将证券覆盖范围转换成全库日历位置。
        first_index, last_index = self.positions[first], self.positions[last]
        positions = self._positions_for_minutes(day, minutes)  # 复用股票之间相同的分钟索引。
        in_span = (positions >= first_index) & (positions <= last_index)  # 标记在证券覆盖范围内的位置。
        relative = 1 + positions[in_span] - first_index  # 计算七个字段共用的相对位置，并跳过一个文件头。
        begin = int(relative.min()) if relative.size else 0  # 取所需片段的起始位置。
        end = int(relative.max()) + 1 if relative.size else 0  # 取所需片段的结束位置，右端不含。
        field_positions = relative - begin  # 将请求位置换算为片段中的下标，保留乱序和重复分钟。
        expected_bytes = (last_index - first_index + 2) * _FIELD_DTYPE.itemsize  # 计算包含文件头的预期字节数。
        byte_offset = begin * _FIELD_DTYPE.itemsize  # 计算片段的起点字节。
        byte_length = (end - begin) * _FIELD_DTYPE.itemsize  # 计算片段的读取字节数。
        rows = np.full((len(minutes), len(FIELDS)), np.nan)  # 按原形状初始化，留下缺失的分钟与字段。
        directory = self.provider / "features" / code.lower()  # 一次定位七个字段所在的证券目录。
        for column, field in enumerate(FIELDS):  # 按行情矩阵规定的列顺序逐字段读取。
            path = directory / f"{field}.1min.bin"  # 定位当前字段文件。
            values = _read_field_slice(path, first_index, expected_bytes, byte_offset, byte_length)  # 只读取所需区间。
            if values is not None:  # 字段文件存在且通过原有基础结构检查。
                rows[in_span, column] = values[field_positions]  # 按原请求顺序依次填入矩阵。
        return rows  # 返回包括缺失值的行情矩阵。
