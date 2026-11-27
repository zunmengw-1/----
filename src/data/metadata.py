"""关联 BaoStock 日线和已核对的交易所事件。

输入为来源清单、SQLite 日线快照和交易所事件 CSV。先按清单装载事件与代码映射，
再按证券和交易日查日线，返回前收参考价、交易标记与停复牌等事件。来源覆盖的
日期范围包括两端，可限定交易所。在声明范围内未找到事件和没有来源覆盖分别处理。
"""

# 读取标准 CSV 事件表和深交所原始导出表。
import csv

# 规范证券代码和原始公告中的时间文本。
import re

# 以只读方式查询 BaoStock 日线快照。
import sqlite3

# 按事件类别和证券日自动创建索引容器。
from collections import defaultdict

# 定义构造后不可修改的股票日与来源记录。
from dataclasses import dataclass

# 解析日期、时刻和时间戳，并逐日展开退市整理区间。
from datetime import date, datetime, time, timedelta

# 用十进制数保存原始前收参考价并捕获无效价格。
from decimal import Decimal, InvalidOperation

# 使用统一路径对象定位清单和数据文件。
from pathlib import Path


# 将元数据无法确认的情况归为标明原因的异常。
class MetadataError(ValueError):
    """日线记录、来源覆盖或事件值无法确认时抛出的错误。"""


# 让股票日记录在创建后保持不可变。
@dataclass(frozen=True)
# 汇总某一证券在某一交易日的日线和官方事件。
class StockDay:
    """保存股票日的参考价、交易状态与交易所披露事件。"""
    # 保存未复权日线提供的当日前收参考价，单位为元。
    preclose: Decimal
    # 标记 BaoStock 日线中的风险警示状态。
    st: bool
    # 保存 BaoStock 日线中的交易状态。
    trading: bool
    # 标记上市初期需排除的交易日。
    ipo: bool
    # 标记处于退市整理期的交易日。
    delist: bool
    # 标记交易所记录的全天停牌。
    full_halt: bool
    # 保存当日已核对的盘中停牌与复牌时刻。
    intraday_halts: tuple[tuple[time, time], ...]


# 让每个来源的路径和声明范围在加载后保持不可变。
@dataclass(frozen=True)
# 描述一个日线或事件数据文件及其声明范围。
class _Source:
    """保存数据路径和适用的交易日及交易所范围。"""
    # 指向清单声明的数据文件。
    path: Path
    # 记录声明覆盖范围的起始日期，含当日。
    start: date
    # 记录声明覆盖范围的结束日期，含当日。
    end: date
    # 记录来源限定的交易所，None 表示不限交易所。
    exchange: str | None

    # 检查某个交易日和交易所是否在来源声明范围内。
    def covers(self, day, exchange):
        """判断日期 day 和交易所 exchange 是否受当前来源覆盖。

        day 为北京时间对应的 date，exchange 是 SH、SZ 或 BJ。
        返回布尔值。日期对象不可比较时可能抛出 TypeError。
        """
        # 日期必须处于闭区间，且来源须适用于该交易所。
        return self.start <= day <= self.end and self.exchange in (None, exchange)


# 接受交易所前缀和后缀两种常见的六位证券代码。
_EQUITY_CODE = re.compile(r"(?:(SH|SZ|BJ)\.?([0-9]{6})|([0-9]{6})\.(SH|SZ|BJ))")
# 列出清单中必须声明的全部日线与交易所事件来源。
_SOURCE_KINDS = ("daily", "ipo", "delist", "full_halt", "intraday_halt")
# 指定从未复权日线快照读取的列和顺序。
_DAILY_COLUMNS = "preclose, isst, tradestatus, adjustflag"
# 为不同事件类别列出可接受的必需列组合。
_EVENT_COLUMNS = {
    # 上市表可逐日列出，也可用一栏列出多个日期。
    "ipo": ({"date"}, {"no_limit_trading_dates_in_2025"}),
    # 退市表可逐日列出，也可声明起止日期。
    "delist": ({"date"}, {"first_period_date", "last_period_date"}),
    # 全天停牌表要求提供停牌日期。
    "full_halt": ({"date"},),
    # 盘中停牌表要求起点，复牌时刻可用 end 或 resume 命名。
    "intraday_halt": ({"date", "start", "end"}, {"date", "start", "resume"}),
# 结束事件类别与必需列的对应表。
}


# 将外部证券代码转换成 Qlib 可用的统一格式。
def _code(value):
    """规范来源中的证券代码 value。

    value 可转换为字符串，接受沪、深、北交易所与六位代码的常见排列。
    返回交易所前缀紧接六位号码的大写字符串。格式无效时抛出 MetadataError。
    """
    # 去除首尾空格并统一为大写，再匹配支持的代码格式。
    match = _EQUITY_CODE.fullmatch(str(value).strip().upper())
    # 无法识别交易所和六位代码时拒绝该输入。
    if match is None:
        # 在异常中注明无法解析的原始证券代码。
        raise MetadataError(f"invalid exchange and security code {value!r}")
    # 区分正则表达式识别出的前缀形式和后缀形式。
    prefix, number, suffix_number, suffix = match.groups()
    # 将交易所代码统一放到证券号码前面。
    return prefix + number if prefix else suffix + suffix_number


# 按严格的年月日文本读取日期。
def _date(value):
    """验证日期字符串 value 并返回 date 对象。

    value 为北京时间的 YYYY-MM-DD 字符串。类型、格式或日历值不合法时
    抛出 MetadataError。
    """
    # 尝试将日期字符串解析为日期对象。
    try:
        # 使用标准库处理固定年月日文本。
        result = date.fromisoformat(value)
    # 将错误类型或无效日期归入元数据错误。
    except (TypeError, ValueError) as exc:
        # 保留原始异常和收到的日期，方便定位问题。
        raise MetadataError(f"expected date YYYY-MM-DD, found {value!r}") from exc
    # 回写为标准字符串，以排除其他 ISO 日期写法。
    if result.isoformat() != value:
        # 对解析成功但格式不一致的字符串报告同类错误。
        raise MetadataError(f"expected date YYYY-MM-DD, found {value!r}")
    # 返回通过格式校验的日期对象。
    return result


# 核对来源或事件的起止日期并返回两个端点。
def _date_range(start, end):
    """校验起始日期 start 和结束日期 end，返回两个 date 端点。

    两个参数均为按北京时间解释的 YYYY-MM-DD 字符串，端点均含。
    端点格式错误或结束早于起始时抛出 MetadataError。
    """
    # 将两个端点分别验证为标准日期。
    first, last = _date(start), _date(end)
    # 结束日期不能早于起始日期。
    if first > last:
        # 报告端点颠倒的原始日期范围。
        raise MetadataError(f"invalid date range {start} through {end}")
    # 返回已验证的起始日期与结束日期。
    return first, last


# 将公告中的停复牌时刻解析为本地时钟值。
def _event_time(value, day):
    """解析事件时间 value 及其所属交易日 day。

    value 为本地时间字符串或完整的 ISO 格式时间戳，day 为北京时间日期对象。
    返回无时区的 time。格式无法解析时抛出 ValueError，日期不一致或存在时区时
    抛出 MetadataError。
    """
    # 含日期和时钟分隔符时按完整时间戳解析。
    if "T" in value or " " in value:
        # 读取停复牌时间戳中的日期和时刻。
        timestamp = datetime.fromisoformat(value)
        # 公告的完整时间戳应与事件所属交易日一致。
        if timestamp.date() != day:
            # 拒绝落在其他日期的停复牌时刻。
            raise MetadataError(f"halt timestamp {value} is outside {day}")
        # 提取时钟值并保留时区信息，供后面统一校验。
        clock = timestamp.timetz()
    # 只有时钟文本时直接解析时分秒。
    else:
        # 读取公告中的本地时钟值。
        clock = time.fromisoformat(value)
    # 仅接受未标注时区的交易所当地时间。
    if clock.tzinfo is not None:
        # 拒绝带时区的值，避免直接使用未经转换的时钟。
        raise MetadataError(f"expected local exchange time, found {value!r}")
    # 返回可供调用者比较的本地时钟值。
    return clock


# 读取事件原始行并按格式统一其字段。
def _csv_rows(file, kind, source_format):
    """逐条产出标准事件字典。

    file 为已打开的文本文件，kind 是事件类别，source_format 是清单规定的
    输入格式。缺少必需列或格式不受支持时抛出 MetadataError。原始行格式
    错误可抛出 ValueError，文件读取错误由调用者处理。
    """
    # 对带标准表头的事件表逐行读取命名字段。
    if source_format == "csv":
        # 使用首行表头建立列名到数据值的对应。
        reader = csv.DictReader(file)
        # 提取全部表头并允许后续检查空表头。
        columns = set(reader.fieldnames or ())
        # 证券代码和一种受支持的事件列组合都须存在。
        if "code" not in columns or not any(required <= columns for required in _EVENT_COLUMNS[kind]):
            # 在异常中指出缺少必需列的事件类型和文件。
            raise MetadataError(f"missing {kind} columns in {file.name}")
        # 按文件顺序产出剩余的事件行。
        yield from reader
        # 标准格式读完后结束生成器。
        return
    # 除标准表外，仅盘中停牌可使用深交所指定原始格式。
    if source_format != "szse_2025" or kind != "intraday_halt":
        # 报告当前类别不支持的来源格式。
        raise MetadataError(f"unsupported metadata format {source_format!r}")
    # 按深交所原始表的列顺序解包，忽略公告文件名和标题。
    for raw_date, _, _, raw_code, start, end in csv.reader(file):
        # 产出供后续统一读取的事件字典。
        yield {
            # 补上深交所前缀得到 Qlib 风格的证券代码。
            "code": "SZ" + raw_code,
            # 把无分隔符的日期变成标准日期字符串。
            "date": datetime.strptime(raw_date, "%Y%m%d").date().isoformat(),
            # 将停牌时刻的中文时间单位转换为标准分隔形式。
            "start": re.sub(r"[时分]", ":", start.removesuffix("秒")),
            # 同样规范复牌时刻。
            "end": re.sub(r"[时分]", ":", end.removesuffix("秒")),
        # 结束当前已规范的深交所事件行。
        }


# 按证券代码和交易日关联未复权日线和交易所事件。
class Metadata:
    """关联同一证券日在未复权日线快照和交易所事件表中的记录。"""

    # 装载元数据清单并准备按股票日查询的索引。
    def __init__(self, spec, base_dir=None):
        """按清单 spec 装载数据来源、事件和别名。

        spec 为 JSON 读取后的字典，base_dir 是解析相对路径的目录，默认 None
        时使用当前工作目录。初始化不返回值。必需来源缺失、范围或事件格式
        无效时抛出 MetadataError，清单缺少必填键时可抛出 KeyError。
        """
        # 未指定根目录时，以运行时的当前工作目录解析相对路径。
        base = Path.cwd() if base_dir is None else Path(base_dir)
        # 按类别登记清单声明的来源。
        self._sources = defaultdict(list)
        # 按文件路径缓存按需打开的只读日线连接。
        self._daily = {}
        # 各类逐日事件按证券代码和日期存入集合，便于去重查询。
        self._events = defaultdict(set)
        # 按证券日保存公告中的盘中停复牌时刻区间。
        self._intraday = defaultdict(set)
        # 按代码保存生效范围内的历史与标准代码对应关系。
        self._aliases = defaultdict(list)
        # 按代码保存清单指定的板块例外。
        self._board_overrides = defaultdict(list)

        # 按清单所需的五种数据类别逐项装载。
        for kind in _SOURCE_KINDS:
            # 取得当前类别的一个或多个来源声明。
            declarations = spec.get(kind)
            # 缺少任一来源类别时停止加载，避免误认为没有事件。
            if not declarations:
                # 将缺少声明的类别写入异常。
                raise MetadataError(f"manifest is missing {kind} source declarations")
            # 同一类别可声明多个文件，逐个校验并加载。
            for declaration in declarations:
                # 按基准目录得到文件的绝对路径。
                path = (base / declaration["path"]).resolve()
                # 确认声明的位置确实存在一个文件。
                if not path.is_file():
                    # 报告缺失的来源类别和路径。
                    raise MetadataError(f"{kind} source does not exist: {path}")
                # 解析文件涵盖的首末交易日期，两个端点均含。
                start, end = _date_range(declaration["start"], declaration["end"])
                # 读取可选的市场限制。
                exchange = declaration.get("exchange")
                # 限制只能是沪、深、北交易所或不限。
                if exchange not in (None, "SH", "SZ", "BJ"):
                    # 在异常中给出无效的交易所及其文件。
                    raise MetadataError(f"invalid source exchange {exchange!r} for {path}")
                # 将已检查的路径、日期和交易所打包为来源记录。
                source = _Source(path, start, end, exchange)
                # 把该记录登记到对应类别，供按日检查覆盖范围。
                self._sources[kind].append(source)
                # 日线连接稍后按需打开，先装载其他类别的事件行。
                if kind != "daily":
                    # 未指定特殊格式时按标准 CSV 装载。
                    self._read_events(kind, source, declaration.get("format", "csv"))

        # 读取清单中可选的历史代码映射。
        for entry in spec.get("aliases", []):
            # 统一新旧代码格式，后续可以用任一代码检索。
            canonical, historical = _code(entry["code"]), _code(entry["historical_code"])
            # 验证别名适用日期范围。
            start, end = _date_range(entry["start"], entry["end"])
            # 将该日期范围和对应关系记到标准代码下。
            self._aliases[canonical].append((start, end, canonical, historical))
            # 两个代码不同时为历史代码也登记同一关系。
            if historical != canonical:
                # 允许查询历史代码时直接得到其标准代码。
                self._aliases[historical].append((start, end, canonical, historical))
        # 读取清单中按生效日期确认的板块例外。
        for entry in spec.get("board_overrides", []):
            # 验证该例外适用的日期范围。
            start, end = _date_range(entry["start"], entry["end"])
            # 保证板块状态由布尔值明确表示。
            if not isinstance(entry["mainboard"], bool):
                # 拒绝字符串或数字等可能导致板块歧义的声明。
                raise MetadataError("board override mainboard must be a boolean")
            # 以规范证券代码登记板块覆盖范围。
            self._board_overrides[_code(entry["code"])].append((start, end, entry["mainboard"]))

    # 进入上下文时提供当前查询对象。
    def __enter__(self):
        """进入 with 上下文，返回当前查询对象。此方法无附加参数或校验。"""
        # 返回当前对象供 with 语句中的变量使用。
        return self

    # 离开上下文时释放本对象打开的日线连接。
    def __exit__(self, exc_type, exc_value, traceback):
        """退出 with 上下文时关闭连接并返回 None。

        exc_type、exc_value 和 traceback 分别来自上下文中的异常类型、对象与
        回溯。它们为 None 时表示正常退出。原异常继续传播，连接关闭错误也会传播。
        """
        # 统一调用连接清理逻辑，让原异常继续传播。
        self.close()

    # 关闭已经按需打开的日线数据库连接。
    def close(self):
        """关闭并清空缓存的日线连接。

        此方法无附加参数，返回 None。连接关闭错误由 SQLite 抛出。
        """
        # 枚举本对象缓存的每个 SQLite 连接。
        for connection in self._daily.values():
            # 关闭当前连接以释放文件句柄。
            connection.close()
        # 清空连接缓存，避免后续检查误用已关闭的连接。
        self._daily.clear()

    # 逐条读取并登记单个来源文件中的交易所事件。
    def _read_events(self, kind, source, source_format):
        """读取来源 source 中属于 kind 的全部事件。

        source 为路径及覆盖信息，kind 是事件类别，source_format 来自清单，
        标准表使用 csv。登记事件后返回 None。读取或解析失败时抛出 MetadataError，
        并附上能够定位问题的文件路径或记录序号。
        """
        # 捕获文件打开或 CSV 读取失败。
        try:
            # 以兼容 UTF-8 文件头的编码打开事件表。
            with source.path.open(newline="", encoding="utf-8-sig") as file:
                # 从第一条数据记录起编号，以便在失败时指出问题所在。
                for row_number, row in enumerate(_csv_rows(file, kind, source_format), start=1):
                    # 单独定位每条事件行的解析错误。
                    try:
                        # 按事件类别将当前原始行写入内存索引。
                        self._read_event(kind, source, row)
                    # 捕获字段缺失、参数类型或事件值不合法的情况。
                    except (KeyError, TypeError, ValueError) as exc:
                        # 把文件路径、记录序号和原始错误一并写入异常。
                        raise MetadataError(f"{source.path}, record {row_number}: {exc}") from exc
        # 捕获发生在文件访问或 CSV 读取阶段的错误。
        except (OSError, csv.Error) as exc:
            # 为读取错误补充具体来源路径。
            raise MetadataError(f"cannot read {source.path}: {exc}") from exc

    # 按来源范围登记一条上市、退市或停牌记录。
    def _read_event(self, kind, source, row):
        """解析并登记事件来源中的一行 row。

        kind 是事件类别，source 描述数据文件及覆盖范围，row 是字段字典。
        登记事件后返回 None。时间、日期或证券代码无效时抛出 ValueError
        或它的子类 MetadataError，字段缺失时可抛出 KeyError。
        """
        # 从当前行取出并规范证券代码。
        code = _code(row["code"])
        # 没有逐日列的上市记录使用 2025 年免限幅日期列表。
        if kind == "ipo" and "date" not in row:
            # 将用竖线连接的日期拆成独立值。
            for value in row["no_limit_trading_dates_in_2025"].split("|"):
                # 忽略列表里没有填写日期的位置。
                if value:
                    # 校验日期后按来源的声明范围登记该日。
                    self._add_event_day(kind, source, code, _date(value))
            # 当前上市行处理完毕，结束本次调用。
            return
        # 没有逐日列的退市记录由整理期首末端点表示。
        if kind == "delist" and "date" not in row:
            # 解析该证券退市整理期的起始日和结束日。
            start, end = _date_range(row["first_period_date"], row["last_period_date"])
            # 将事件区间裁剪到该文件声明的生效日期范围。
            day, end = max(start, source.start), min(end, source.end)
            # 逐个自然日展开裁剪后的闭区间。
            for offset in range((end - day).days + 1):
                # 将当前展开日期登记到退市事件索引。
                self._add_event_day(kind, source, code, day + timedelta(days=offset))
            # 当前退市区间处理完毕，结束本次调用。
            return
        # 其余来源每行只提供一个标准日期。
        day = _date(row["date"])
        # 上市、退市和全天停牌逐日记录只需要证券代码与日期。
        if kind != "intraday_halt":
            # 经过来源范围检查后登记该证券日。
            self._add_event_day(kind, source, code, day)
            # 当前逐日事件处理完毕，结束本次调用。
            return
        # 读取盘中停牌起点，完整时间戳必须属于该交易日。
        begin = _event_time(row["start"], day)
        # 优先读 end 列，没有有效值时使用 resume 列。
        resume = _event_time(row.get("end") or row["resume"], day)
        # 停牌开始必须早于复牌，才形成有效区间。
        if begin >= resume:
            # 在异常中同时报告证券日、开始和复牌时刻。
            raise MetadataError(f"invalid intraday halt {code} {day} {begin}–{resume}")
        # 仅登记来源日期和交易所范围内的盘中停牌。
        if source.covers(day, code[:2]):
            # 通过集合保留当前证券日不重复的停复牌区间。
            self._intraday[code, day].add((begin, resume))

    # 当逐日事件属于声明范围时写入索引。
    def _add_event_day(self, kind, source, code, day):
        """把给定证券日登记进事件类别 kind 的索引。

        source 是声明范围，code 是已规范的证券代码，day 为已验证的日期对象。
        超出范围的记录不登记，方法返回 None，无额外输入校验。
        """
        # 检查日期和证券代码前两位代表的交易所是否符合来源声明。
        if source.covers(day, code[:2]):
            # 将证券日加入对应事件集合并自动去重。
            self._events[kind].add((code, day))

    # 查找指定代码在交易日内生效的别名关系。
    def _resolve(self, code, day):
        """查询 code 在 day 生效的代码对应关系。

        code 已规范，day 是已验证的日期对象。返回标准代码和历史代码二元组，
        没有别名时两项均为 code。多项不同映射同时生效时抛出 MetadataError。
        """
        # 收集该代码关联的标准代码和历史代码组合。
        matches = {(canonical, historical) for start, end, canonical, historical in self._aliases[code]
                   # 仅保留日期处于生效闭区间的映射。
                   if start <= day <= end}
        # 同时存在多个不同代码关系时，映射无法确定。
        if len(matches) > 1:
            # 报告发生冲突的证券代码和交易日。
            raise MetadataError(f"conflicting code aliases for {code} on {day}")
        # 返回唯一映射，没有映射时两种代码都沿用输入。
        return next(iter(matches), (code, code))

    # 查询证券在给定交易日对应的 Qlib 代码。
    def canonical_code(self, code, day):
        """取得 code 在交易日 day 对应的 Qlib 证券代码。

        code 可使用交易所代码与证券号的常见排列，day 是北京时间的 YYYY-MM-DD。
        返回标准代码字符串。日期、代码无效或别名冲突时抛出 MetadataError。
        """
        # 规范日期与代码，再从配对结果中取出标准代码。
        canonical, _ = self._resolve(_code(code), _date(day))
        # 返回可与 Qlib 分钟库关联的代码。
        return canonical

    # 查询交易所曾在给定交易日使用的证券代码。
    def historical_code(self, code, day):
        """取得 code 在交易日 day 对应的交易所历史代码。

        code 可使用交易所代码与证券号的常见排列，day 是北京时间的 YYYY-MM-DD。
        返回当日实际代码字符串。日期、代码无效或别名冲突时抛出 MetadataError。
        """
        # 规范查询值后取出当日生效的历史代码。
        _, historical = self._resolve(_code(code), _date(day))
        # 返回可与当日交易所公告关联的代码。
        return historical

    # 判定指定证券日是否属于沪深主板。
    def is_mainboard(self, code, day):
        """判断 code 在交易日 day 是否属于沪深主板。

        code 为外部或 Qlib 证券代码，day 是北京时间的 YYYY-MM-DD 字符串。
        返回布尔值。日期、代码无效，或别名、板块声明冲突时抛出 MetadataError。
        """
        # 将交易日文本转换为日期对象供区间比较。
        day = _date(day)
        # 解析当日生效的标准代码与历史代码。
        canonical, historical = self._resolve(_code(code), day)
        # 同时查找新旧代码对应的板块例外。
        overrides = {mainboard for key in {canonical, historical}
                     # 只保留交易日位于有效范围内的主板判断结果。
                     for start, end, mainboard in self._board_overrides[key] if start <= day <= end}
        # 同时得到主板和非主板两个结果时无法确认板块。
        if len(overrides) > 1:
            # 报告板块冲突发生的证券代码和交易日。
            raise MetadataError(f"conflicting board declarations for {code} on {day}")
        # 优先采用例外声明，否则按历史代码的沪深主板前缀判断。
        return next(iter(overrides), historical.startswith(("SH60", "SZ00")))

    # 查询目标股票日的唯一有效日线记录。
    def _daily_rows(self, day, canonical, historical):
        """从覆盖 day 的日线来源查询 canonical 和 historical 两种代码。

        day 是已验证的日期对象，后两项是标准及历史证券代码。返回以元为单位的
        前收参考价、ST 标记和交易标记。查询失败或数据缺失、冲突、格式无效时
        抛出 MetadataError。
        """
        # 用集合保存结构化查询结果并合并完全相同的重复记录。
        rows = set()
        # 若发生代码变更则查询历史与标准两个代码。
        codes = (historical,) if historical == canonical else (historical, canonical)
        # 逐个检查日线清单中的来源文件。
        for source in self._sources["daily"]:
            # 跳过未覆盖该日期或历史代码所属交易所的来源。
            if not source.covers(day, historical[:2]):
                # 当前来源不适用时继续检查下一条声明。
                continue
            # 捕获连接数据库或查询日线失败的情况。
            try:
                # 重用该文件先前建立的 SQLite 连接。
                connection = self._daily.get(source.path)
                # 首次访问文件时按需打开一个只读连接。
                if connection is None:
                    # 使用 SQLite 文件 URI 指定只读模式。
                    connection = sqlite3.connect(source.path.as_uri() + "?mode=ro", uri=True)
                    # 缓存连接供下一次证券日查询使用。
                    self._daily[source.path] = connection
                # 对可能使用的历史和标准证券代码分别查询。
                for code in codes:
                    # 通过带参数的语句读取指定股票日。
                    results = connection.execute(
                        # 查询的条件来自交易日期与当前证券代码。
                        f"SELECT {_DAILY_COLUMNS} FROM daily WHERE date = ? AND code = ?", (day.isoformat(), code)
                    # 结束查询调用，得到可迭代的日线查询结果。
                    )
                    # 逐条处理当前来源返回的数据。
                    for result in results:
                        # 规范每条日线值后加入去重集合。
                        rows.add(self._parse_daily_row(result, code, day))
            # 拦截 SQLite 连接和查询错误。
            except sqlite3.Error as exc:
                # 在异常中补充证券日、文件路径和数据库错误。
                raise MetadataError(f"cannot read daily metadata for {historical} on {day} from {source.path}: {exc}") from exc
        # 结果须有且仅有一个可用值。
        if len(rows) != 1:
            # 区分没有结果与不同来源存在不同值。
            issue = "missing" if not rows else "conflicting"
            # 把缺失或冲突连同证券日写入异常。
            raise MetadataError(f"{issue} daily metadata for {historical} on {day}")
        # 返回唯一的前收参考价、风险警示状态和交易状态。
        return rows.pop()

    # 解析一条日线记录时不需要访问当前实例。
    @staticmethod
    # 验证未复权日线的价格字段及两个状态字段。
    def _parse_daily_row(row, code, day):
        """验证并转换来自日线查询的 row。

        row 依次包含前收参考价、风险警示状态、交易状态和复权标记。code 与 day
        用于定位错误。返回 Decimal 价格和两个布尔值。复权标记不为 3、状态值
        不是 0 或 1、价格无效时抛出 MetadataError。
        """
        # 按 SQL 查询所用顺序解包四个字段。
        preclose, st, trading, adjustment = row
        # 只接受未复权记录以及明确的二值状态标记。
        if str(adjustment) != "3" or str(st) not in ("0", "1") or str(trading) not in ("0", "1"):
            # 在异常中附上无效的日线记录。
            raise MetadataError(f"invalid adjustment or trading flags for {code} on {day}: {row!r}")
        # 将前收参考价转换成用于价格计算的十进制类型。
        try:
            # 先使用文本形式读取，避免引入额外的浮点近似。
            preclose = Decimal(str(preclose))
        # 捕获前收参考价不能转换成十进制数的情况。
        except InvalidOperation as exc:
            # 对无效价格报告证券、日期和原始值。
            raise MetadataError(f"invalid preclose for {code} on {day}: {preclose!r}") from exc
        # 前收参考价必须为正且是有限数值。
        if not preclose.is_finite() or preclose <= 0:
            # 停止使用非正价格、无穷或无效数值。
            raise MetadataError(f"invalid preclose for {code} on {day}: {preclose!r}")
        # 返回价格与转换后的两个布尔状态。
        return preclose, str(st) == "1", str(trading) == "1"

    # 汇总一个股票日的已核对日线和所有相关事件。
    def day(self, code, day):
        """查询 code 在交易日 day 的日线状态与交易所事件。

        code 为可识别的证券代码，day 是北京时间的 YYYY-MM-DD 字符串。返回
        StockDay。来源未声明覆盖、元数据缺失或无效、来源相互冲突时抛出
        MetadataError。
        """
        # 校验交易日期后统一使用日期对象。
        day = _date(day)
        # 取得关联 Qlib 和历史公告所需的两个代码。
        canonical, historical = self._resolve(_code(code), day)
        # 逐项确认每种来源在指定证券日均声明覆盖。
        for kind, sources in self._sources.items():
            # 仅在该类别有来源覆盖日期和实际交易所时继续。
            if not any(source.covers(day, historical[:2]) for source in sources):
                # 来源有覆盖缺口时明确报告类别和证券日。
                raise MetadataError(f"{kind} source has no declared coverage for {historical} on {day}")
        # 读取唯一符合条件的 BaoStock 前收参考价和交易标记。
        preclose, st, trading = self._daily_rows(day, canonical, historical)
        # 允许用公告历史代码或 Qlib 标准代码匹配已登记的事件。
        keys = {(historical, day), (canonical, day)}
        # 合并同一证券日两种代码下的盘中停牌区间并去重。
        halts = {interval for key in keys for interval in self._intraday.get(key, ())}
        # 将查到的日线状态和公告区间装入不可变结果。
        return StockDay(
            # 前收参考价直接来自已校验的未复权日线。
            preclose=preclose,
            # 风险警示标记直接来自已校验的日线。
            st=st,
            # 交易状态用于与正式停牌记录交叉核对。
            trading=trading,
            # 标记该证券日是否命中上市初期限幅排除表。
            ipo=bool(keys & self._events["ipo"]),
            # 标记该证券日是否命中退市整理期表。
            delist=bool(keys & self._events["delist"]),
            # 标记该证券日是否命中全天停牌表。
            full_halt=bool(keys & self._events["full_halt"]),
            # 将去重后的停牌区间按时钟排序并固定为元组。
            intraday_halts=tuple(sorted(halts)),
        # 结束股票日结果对象的构造。
        )
