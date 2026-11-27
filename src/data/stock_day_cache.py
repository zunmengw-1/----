"""把已对齐的 Qlib 行情按股票日写到连续文件，供训练时随机读取。

输入为原始行情读取器、股票日列表、固定分钟表和字段次序。构建时只保存原始
float32 行情和索引，加载时按需映射文件，逐日恢复原读取器交付的 float64 数组。
"""

import json  # 读取和写入缓存版本、字段和来源信息。
from errno import EBADF, ELOOP, ENOENT, ENOTDIR  # 识别原读取器视作缺失的字段路径。
from hashlib import sha256  # 核对来源的日历、证券清单和字段文件状态。
from itertools import pairwise  # 就地比较相邻的股票日键。
from pathlib import Path  # 组织缓存和来源文件路径。
from shutil import disk_usage  # 写入前检查目标磁盘空间。
from stat import S_IFMT  # 提取文件类型并忽略无关的权限位。
from tempfile import TemporaryDirectory  # 在同一文件系统完整构建后再发布。

import numpy as np  # 用可按需映射的数组保存连续行情和有序索引。

_CACHE_VERSION = 1  # 标识当前缓存的文件结构和存储精度。
_KEY_DTYPE = np.dtype("S18")  # 每个键依次保存 8 字节证券代码和 10 字节 ISO 日期。
_BAR_DTYPE = np.dtype("<f4")  # 沿用 Qlib 行情文件中的小端单精度。


def _stock_day_key(code, day):  # 将一个股票日编码成固定长度的有序键。
    """接收 8 字符证券代码和 10 字符北京时间 ISO 日期，返回 ASCII 字节。"""
    if len(code) != 8 or len(day) != 10:  # 固定长度能避免两个不同股票日映射到同一个键。
        raise ValueError(f"Invalid stock-day for cache: {code} {day}")  # 报出不符合缓存键格式的输入。
    return (code + day).encode("ascii")  # 按证券代码、日期排列，供二分查找。


def _source_signature(provider, codes, fields):  # 为缓存记录依赖的行情来源状态。
    """接收 Qlib 目录、证券代码和字段，返回日历内容及字段文件状态的摘要。"""
    checksum = sha256()  # 累计影响分钟对齐和原始数据读取的来源标记。
    for name in ("calendars/1min.txt", "instruments/all.txt"):  # 核对来源的分钟位置与覆盖范围。
        checksum.update(sha256((provider / name).read_bytes()).digest())  # 用定长摘要记录每份索引的内容。
    for code in codes:  # 每只缓存股票仅检查一次它的七个字段文件。
        for field in fields:  # 只检查缓存完整性所依赖的来源字段。
            path = provider / "features" / code.lower() / f"{field}.1min.bin"  # 复用 Qlib 的字段路径格式。
            try:  # 查找字段当前的文件类型、字节数和修改时刻。
                info = path.stat()  # 与正式读取器一样跟随有效的符号链接。
            except OSError as error:  # 与正式读取器对齐，允许部分字段文件缺失。
                if error.errno not in (ENOENT, ENOTDIR, EBADF, ELOOP):  # 其他访问错误仍交给调用方。
                    raise  # 避免把无法验证的来源默认为可用。
                stamp = None  # 记录该字段在来源中没有可定位的文件。
            else:  # 按状态识别字段的新建、替换或修改。
                stamp = (S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns)  # 使用类型、大小和纳秒修改时刻。
            checksum.update(json.dumps((code, field, stamp)).encode("ascii"))  # 同时记下证券和字段身份。
    return checksum.hexdigest()  # 返回供构建和加载阶段核对的来源签名。


class StockDayCache:  # 按固定分钟表读取磁盘中已对齐的原始行情。
    """加载只读股票日索引和数据映射，校验文件形状与 Qlib 来源版本。"""

    def __init__(self, directory: Path, provider: Path, fields):  # 核对缓存与当前读取器的兼容性。
        """directory 为缓存目录，provider 为原始 Qlib 根目录，fields 为有序字段名。

        缓存结构、字段或来源状态变化时报 ValueError。文件访问错误向调用方传递。
        """
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))  # 读取构建时保存的参数。
        if manifest["version"] != _CACHE_VERSION or manifest["fields"] != list(fields):  # 验证当前程序认识该版本和字段。
            raise ValueError("Incompatible stock-day cache format or fields")  # 要求使用当前版本重新构建。
        self._minutes = tuple(manifest["minutes"])  # 保留缓存生成时的分钟顺序。
        self._keys = np.load(directory / "keys.npy", mmap_mode="r", allow_pickle=False)  # 仅映射索引文件。
        self._bars = np.load(directory / "bars.npy", mmap_mode="r", allow_pickle=False)  # 仅按请求从行情文件取一块。
        count = manifest["stock_days"]  # 股票日数决定索引长度和行情块数。
        if self._keys.shape != (count,) or self._keys.dtype != _KEY_DTYPE:  # 防止索引格式或行数异常。
            raise ValueError("Invalid stock-day cache index")  # 拒绝可能把行情关联到错误证券的索引。
        if self._bars.shape != (count, len(self._minutes), len(fields)) or self._bars.dtype != _BAR_DTYPE:  # 核对行情尺寸与精度。
            raise ValueError("Invalid stock-day cache bars")  # 拒绝不完整或不兼容的行情块。
        if np.any(self._keys[1:] <= self._keys[:-1]):  # 二分查找要求索引严格递增。
            raise ValueError("Stock-day cache index must be unique and sorted")  # 拒绝重复或乱序的键。
        if _source_signature(provider, manifest["codes"], fields) != manifest["source_signature"]:  # 检查来源仍与构建时相符。
            raise ValueError("Qlib source changed since the stock-day cache was built")  # 避免使用过期行情。

    def stock_days(self):  # 每次按索引顺序遍历缓存包含的股票日。
        """逐项返回规范股票代码和北京时间 ISO 交易日，供构造各轮训练请求。"""
        for key in self._keys:  # 从磁盘索引逐项取出固定长度的键。
            yield key[:8].decode("ascii"), key[8:].decode("ascii")  # 分开恢复证券代码和日期。

    def read(self, code, day, minutes):  # 根据请求在索引中查找单个股票日。
        """接收证券代码、北京时间日期和分钟表，命中时返回浮点行情，未命中时返回 None。"""
        if tuple(minutes) != self._minutes:  # 缓存按一套固定分钟表存放，其他分钟请求交给原读取器。
            return None  # 跳过不适用的缓存。
        key = _stock_day_key(code, day)  # 对齐构建缓存时所用的股票日键。
        row = self._keys.searchsorted(key)  # 二分查找，仅访问磁盘索引所需页面。
        if row == len(self._keys) or self._keys[row] != key:  # 避免把未覆盖的日期关联到邻近股票日。
            return None  # 交给 Qlib 读取缺少的股票日。
        return self._bars[row].astype(np.float64)  # 恢复正式样本生成器使用的精度。


def build_stock_day_cache(library, requests, minutes, fields, output: Path):  # 一次性把原始 Qlib 数据转为股票日缓存。
    """library 为直接读取 Qlib 的接口，requests 为唯一的规范股票日键。

    minutes 按北京时间分钟偏移量给出固定网格，fields 按读取器列顺序给出字段名。
    output 为新的结果目录。成功时返回缓存股票日数，重复键或目标路径已存在时抛出 ValueError。
    """
    if output.exists() or output.is_symlink():  # 避免覆盖现有缓存或其他结果。
        raise ValueError(f"Output directory already exists: {output}")  # 要求调用方选择新路径。
    stock_days = sorted(requests)  # 使行情块能够通过证券及日期顺序的键查找。
    count = len(stock_days)  # 取得本次需要落盘的股票日数。
    if not count or any(left == right for left, right in pairwise(stock_days)):  # 缓存需要非空且不重复。
        raise ValueError("Stock-day cache requires a nonempty list of unique stock-days")  # 报出无效的清单。
    minute_grid = tuple(minutes)  # 构建和读取使用相同顺序的固定分钟。
    codes = sorted({code for code, _ in stock_days})  # 只核对确实用于构建缓存的证券。
    signature = _source_signature(library.provider, codes, fields)  # 记录开始时来源状态。
    shape = (count, len(minute_grid), len(fields))  # 为每个股票日分配一块连续的日内行情。
    manifest = {  # 保存声明缓存适用范围和来源的参数。
        "version": _CACHE_VERSION, "stock_days": count,  # 记录文件版本及股票日数。
        "fields": list(fields), "minutes": list(minute_grid),  # 记录字段及连续竞价分钟次序。
        "codes": codes, "source_signature": signature,  # 记录来源检查所需的证券和摘要。
    }  # 结束缓存清单。
    document = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"  # 先生成清单，便于计量文件大小。
    output.parent.mkdir(parents=True, exist_ok=True)  # 在结果路径所在的文件系统构建。
    required_bytes = count * (len(minute_grid) * len(fields) * _BAR_DTYPE.itemsize + _KEY_DTYPE.itemsize)  # 估算两份数组的载荷。
    if disk_usage(output.parent).free < required_bytes + len(document.encode("utf-8")) + 8_192:  # 为清单和数组头保留空间。
        raise OSError(f"Not enough disk space for {required_bytes:,} bytes of stock-day cache data")  # 避免在映射写入中耗尽磁盘。
    with TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:  # 临时内容和最终路径位于同一个磁盘。
        staged = Path(temporary) / "cache"  # 在临时目录中组织齐全的缓存。
        staged.mkdir()  # 创建仅在构建完成后才发布的结果目录。
        bars = np.lib.format.open_memmap(staged / "bars.npy", mode="w+", dtype=_BAR_DTYPE, shape=shape)  # 逐日写入，避免读取全量到内存。
        keys = np.lib.format.open_memmap(staged / "keys.npy", mode="w+", dtype=_KEY_DTYPE, shape=(count,))  # 顺序写入对应键。
        for row, (code, day) in enumerate(stock_days):  # 把每个股票日写入一份索引和一个连续行情块。
            keys[row] = _stock_day_key(code, day)  # 在索引中保存行情块所属的股票日。
            bars[row] = library.session_bars(code, day, minute_grid)  # 保存原字段值和缺失分钟位置。
        bars.flush()  # 在发布前落盘行情数据。
        keys.flush()  # 在发布前落盘股票日索引。
        del bars, keys  # 释放写入映射，使加载器只看到完整的文件。
        if _source_signature(library.provider, codes, fields) != signature:  # 避免构建过程中混用不同版本的原始行情。
            raise ValueError("Qlib source changed while building the stock-day cache")  # 发布前报告变化。
        (staged / "manifest.json").write_text(document, encoding="utf-8")  # 完整写入版本和来源声明。
        if output.exists() or output.is_symlink():  # 构建期间可能已有另一进程准备好结果。
            raise ValueError(f"Output directory already exists: {output}")  # 保留其他运行已经写好的内容。
        staged.rename(output)  # 完成后在目标位置一次发布，调用方不会读到半个缓存。
    return count  # 告知调用方真实写入的股票日数。
