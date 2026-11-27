"""将分钟量价窗口和可选的时钟条件转换为对数收益率预测。

模型用因果卷积提取局部时序特征，按 30→15→5→1 压缩时间维，
用回归头将每个窗口的编码映射为一个预测值。
网络结构来自 GitHub 提交 2878cc27b4c2052d93e427f01985ea3c77dcdaf5
的 src/intraday_tcn/training/model.py。
"""

from math import isfinite  # 检查模型输出尺度能否用有限浮点数表示。

import torch  # 使用 PyTorch 的整数索引类型。
from torch import Tensor, float32, nn, tensor  # 使用 PyTorch 张量和网络层。
from torch.nn import functional as F  # 使用 PyTorch 的时序填充函数。
from torch.nn.utils.parametrizations import weight_norm  # 对卷积核应用权重归一化。

from src.features.price_scale import PRICE_CHANNELS  # 统一识别已定稿的四列价格。
from src.features.samples import (  # 复用正式采样的时间与量价口径。
    FEATURES,  # 确认后融合的卷积仅接收量价通道。
    SESSIONS,  # 列出有完整历史的预测分钟。
    TIME_FEATURES,  # 取得预测头的时钟维数。
    WINDOW,  # 使用与采样一致的窗口长度。
    time_features,  # 取得和原八通道一致的时钟值。
)  # 完成样本特征接口导入。

_MINUTES_PER_DAY = 24 * 60  # 尺度表以北京时间自午夜起的分钟数为索引。
_DEFAULT_HIDDEN_CHANNELS = 16  # 默认及旧检查点使用的卷积宽度。
_HEAD_HIDDEN_CHANNELS = 16  # 非线性预测头隐藏层的固定宽度。
_NEGATIVE_SLOPE = 0.01  # LeakyReLU 在负值区域的斜率。


def _main_conv(  # 构造带权重归一化和激活的时序卷积。
    in_channels: int,  # 输入的特征通道数。
    out_channels: int,  # 输出的特征通道数。
    kernel_size: int,  # 卷积核包含的时间位置数。
    *,  # 步长和膨胀率通过名称指定。
    stride: int = 1,  # 默认每个时间位置计算一次卷积。
    dilation: int = 1,  # 默认使用连续的时间位置。
) -> nn.Sequential:  # 返回按顺序执行卷积、激活的 PyTorch 模块。
    """按通道数、卷积核长度、步长和膨胀率构造一维卷积及激活。

    参数必须符合 PyTorch Conv1d 的要求。需要因果填充时由调用方在左侧补零。
    """
    conv = nn.Conv1d(  # 使用 PyTorch 标准的一维卷积。
        in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,  # 应用本层的时序尺寸。
    )  # 保留默认的零填充长度，由残差块处理因果填充。
    nn.init.kaiming_normal_(  # 先初始化卷积核，再注册权重归一化。
        conv.weight, a=_NEGATIVE_SLOPE, mode="fan_in", nonlinearity="leaky_relu",  # 按激活函数及输入规模初始化。
    )  # 卷积核初始化完成。
    nn.init.zeros_(conv.bias)  # 将卷积的初始偏置置零。
    return nn.Sequential(weight_norm(conv, dim=0), nn.LeakyReLU(_NEGATIVE_SLOPE))  # 逐输出通道归一化，再激活。


class _ResidualBlock(nn.Module):  # 定义带捷径的因果时序卷积块。
    """使用膨胀率为 1 和 2 的两层因果卷积，保留输入的时间长度。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:  # 按输入和输出通道数初始化卷积与捷径。
        """in_channels 和 out_channels 为输入、输出通道数，须为正整数。"""
        super().__init__()  # 初始化 PyTorch 模块以注册子层。
        self.conv1 = _main_conv(in_channels, out_channels, kernel_size=3)  # 读取当前及此前两个相邻位置。
        self.conv2 = _main_conv(out_channels, out_channels, kernel_size=3, dilation=2)  # 每隔一个位置取值。
        if in_channels == out_channels:  # 通道相同时可直接相加。
            self.shortcut = nn.Identity()  # 捷径按原样传递输入。
        else:  # 通道不同时先对齐每个时间位置的通道数。
            projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 只在同一时间位置投影。
            nn.init.xavier_uniform_(projection.weight)  # 初始化捷径的投影权重。
            nn.init.zeros_(projection.bias)  # 将投影的初始偏置置零。
            self.shortcut = projection  # 将投影层保存为捷径。

    def forward(self, x: Tensor) -> Tensor:  # 计算输入批次的因果卷积与残差和。
        """x 为 [批大小, 输入通道, 时间长度]，返回时间长度相同的输出通道张量。"""
        features = self.conv1(F.pad(x, (2, 0)))  # 在左侧补两个位置，使卷积只访问当前及更早的输入。
        features = self.conv2(F.pad(features, (4, 0)))  # 按膨胀率在左侧补四个位置，保留时间长度。
        return features + self.shortcut(x)  # 将卷积结果和时间对齐后的捷径相加。


class ReturnTCN(nn.Module):  # 定义用于预测单一收益率的 TCN。
    """接收训练加载器产生的 [批大小, 通道数, 30] 单精度特征张量。"""

    return_scale: Tensor  # 保存全局尺度，便于使用统一单位计算验证损失。
    minute_return_scales: Tensor | None  # 可选的逐分钟训练尺度表，随模型保存和迁移。
    minute_price_multipliers: Tensor | None  # 可选的价格输入倍率表，与检查点一同保存。
    position_means: Tensor | None  # 可选的逐预测分钟、窗内位置均值表，与检查点一同保存。
    clock_rows: Tensor | None  # 按时钟分钟定位预测头时钟特征的行。
    head_clock: Tensor | None  # 完整窗口的预测分钟对应的两个时钟值。
    window_clock: Tensor | None  # 门控模型的逐根时钟查表。

    # 保留已公开的构造参数，允许分别配置门控和卷积宽度。
    def __init__(  # noqa: PLR0913
        self,  # 当前 TCN 实例。
        *,  # 模型配置以名称指定。
        return_scale: float = 1.0,  # 用正有限尺度把模型分数还原为物理对数收益率。
        input_channels: int = 6,  # 默认只用六个量价通道，加入时间特征时设为八。
        minute_return_scales: Tensor | None = None,  # 可选的正有限单精度尺度表，形状为 [1440]。
        time_at_head: bool = False,  # 只对六通道启用末端时钟条件。
        minute_price_multipliers: Tensor | None = None,  # 可选的正有限 float32 价格输入倍率表，形状为 [1440]。
        position_means: Tensor | None = None,  # 可选的有限 float32 [1440, 30, 3] 表，依次存储收盘、成交量和成交额均值。
        window_clock_mode: str = "none",  # gate 只在首个残差块输出后施加逐根时钟乘法。
        hidden_channels: int = _DEFAULT_HIDDEN_CHANNELS,  # 卷积阶段使用的共同宽度，仅门控模式允许配置。
    ) -> None:  # 保存所选输入规格和对应网络。
        """以指定的输入通道数和收益率尺度初始化模型。

        return_scale 为全局训练标签标准差。input_channels 可为 6 或 8，八通道
        沿用逐分钟时钟输入。time_at_head 为 True 时，卷积使用六列量价，预测头
        追加预测分钟的两个时钟值。提供 minute_return_scales 时，模型按预测分钟
        将分数还原为物理收益率。minute_price_multipliers 按北京时间预测分钟提供
        输入倍率，对整个窗口的四列价格生效。提供 position_means 时先减去按
        北京时间预测分钟和窗口位置固定的训练集均值。window_clock_mode 为 gate
        时须用六列量价且关闭时钟头，可指定正整数 hidden_channels。卷积统一使用该
        宽度，预测头隐藏层固定十六维。配置或表无效时抛出 ValueError。
        """
        super().__init__()  # 初始化 PyTorch 模块以注册网络参数及缓冲区。
        self._validate_structure(input_channels, time_at_head, window_clock_mode, hidden_channels)  # 检查前向通道和合法的时钟组合。
        self.input_channels = input_channels  # 保存可供加载器和验证入口读取的输入规格。
        self.hidden_channels = hidden_channels  # 保存所有卷积阶段使用的共同宽度。
        self.window_clock_mode = window_clock_mode  # 保存可重建的时钟模式。
        use_gate = window_clock_mode == "gate"  # 识别唯一的逐根时钟模式。
        scale = tensor(float(return_scale), dtype=float32)  # 将输出尺度保存为单精度标量。
        scale_value = float(scale)  # 读取单精度表示以检查是否溢出或下溢。
        if not isfinite(scale_value) or scale_value <= 0:  # 有效尺度须为正有限值。
            raise ValueError("收益率尺度须为单精度可表示的正有限值")  # 报告无效尺度。
        self.register_buffer("return_scale", scale)  # 随模型移动设备并与权重一同保存全局尺度。
        if minute_return_scales is not None:  # 按分钟模式仅在构造时检查表的全部数值。
            if minute_return_scales.shape != (_MINUTES_PER_DAY,) or minute_return_scales.dtype != float32:  # 检查表的形状和类型。
                raise ValueError("分钟尺度表须为形状 [1440] 的 float32 张量")  # 防止错误的时点索引布局。
            minute_return_scales = minute_return_scales.detach().clone()  # 保存独立于调用方张量的固定尺度。
            if not (minute_return_scales.isfinite() & (minute_return_scales > 0)).all():  # 每日全部分钟均须有可用尺度。
                raise ValueError("分钟尺度表的所有值须为正有限值")  # 避免损失归一化产生无效数。
        self.register_buffer("minute_return_scales", minute_return_scales)  # 保存已有分钟表；空表不进入模型状态。
        if minute_price_multipliers is not None:  # 新检查点带已拟合的价格输入倍率。
            if minute_price_multipliers.shape != (_MINUTES_PER_DAY,) or minute_price_multipliers.dtype != float32:  # 校验时钟索引和精度。
                raise ValueError("价格输入倍率表须为形状 [1440] 的 float32 张量")  # 拒绝不完整的倍率表。
            minute_price_multipliers = minute_price_multipliers.detach().clone()  # 保持模型中的倍率独立且固定。
            if not (minute_price_multipliers.isfinite() & (minute_price_multipliers > 0)).all():  # 只允许正有限倍率。
                raise ValueError("价格输入倍率表的所有值须为正有限值")  # 防止静态输入变换产生无效值。
        self.register_buffer("minute_price_multipliers", minute_price_multipliers)  # 训练和检查点重载复用同一表。
        if position_means is not None:  # 逐位置去均值使用拟合后固定的训练表。
            if position_means.shape != (_MINUTES_PER_DAY, WINDOW, 3) or position_means.dtype != float32:  # 检查分钟、位置和均值列布局。
                raise ValueError("位置均值表须为形状 [1440, 30, 3] 的 float32 张量")  # 拒绝无效形状或精度。
            position_means = position_means.detach().clone()  # 固定保存独立于调用方的均值表。
            if not position_means.isfinite().all():  # 带入非有限均值会污染卷积窗口。
                raise ValueError("位置均值表的所有值须有限")  # 在训练启动前报告无效均值。
        self.register_buffer("position_means", position_means)  # 随模型保存和移动，旧模型不多出状态字段。
        clock_rows = head_clock = window_clock = None  # 常规卷积模式无需准备独立时钟表。
        if time_at_head or use_gate:  # 两个互斥的时钟位置共用同样的合法分钟定义。
            valid_minutes = torch.tensor(  # 在 CPU 上列出有完整窗口的预测分钟。
                [minute for session in SESSIONS.values() for minute in session[WINDOW - 1:]], device="cpu",  # 与既有时段网格一致。
            )  # 完成预测分钟索引。
            clock_rows = torch.full((_MINUTES_PER_DAY,), -1, dtype=torch.long, device="cpu")  # 无效分钟指向越界行。
            clock_rows[valid_minutes] = torch.arange(len(valid_minutes), device="cpu")  # 将有效分钟映射到时钟行。
            clock_rows = clock_rows.to(device=scale.device)  # 将索引表放在模型初始设备上。
            clock = torch.from_numpy(time_features(valid_minutes.numpy())).to(device=scale.device)  # 从预测分钟构造所有位置的既定时钟。
            if use_gate:  # 门控模式使用完整窗口时钟。
                window_clock = clock  # 每根分钟都可以形成一个通道乘数。
            else:  # 原后融合只从窗口末根读取时钟。
                head_clock = clock[:, :, -1].contiguous()  # 保留原检查点的头部输入逻辑。
        self.register_buffer("clock_rows", clock_rows, persistent=False)  # 使索引表随模型移动设备。
        self.register_buffer("head_clock", head_clock, persistent=False)  # 由模型配置与时段重新构造后融合表。
        self.register_buffer("window_clock", window_clock, persistent=False)  # 门控时钟由同一规则重建，无需占用检查点。
        self.minute_block = _ResidualBlock(self.input_channels, self.hidden_channels)  # 用配置宽度编码每根分钟输入。
        self.downsample_pairs = _main_conv(  # 将相邻的两个分钟位置压缩成一个位置。
            self.hidden_channels, self.hidden_channels, kernel_size=2, stride=2,  # 每一对输出对应其最后一分钟。
        )  # 时间长度从三十减为十五。
        self.two_minute_block = _ResidualBlock(self.hidden_channels, self.hidden_channels)  # 提取同宽度的两分钟尺度因果特征。
        self.downsample_triples = _main_conv(  # 将相邻的三个两分钟位置压缩成一个位置。
            self.hidden_channels, self.hidden_channels, kernel_size=3, stride=3,  # 每个输出对齐到这组三个位置中的最后一个。
        )  # 时间长度从十五减为五。
        self.collapse_time = _main_conv(  # 聚合经过压缩的全部时间位置。
            self.hidden_channels, self.hidden_channels, kernel_size=5,  # 将时间维聚合为一个位置，保留所选的卷积宽度。
        )  # 时间长度从五减为一。
        output = nn.Linear(_HEAD_HIDDEN_CHANNELS, 1)  # 将固定十六维表示映射为一个收益率分数。
        nn.init.xavier_uniform_(output.weight)  # 初始化收益率回归权重。
        nn.init.zeros_(output.bias)  # 将回归头的初始偏置置零。
        if time_at_head or use_gate:  # 按已验证的旧门控构造顺序初始化公共非线性层。
            hidden = nn.Linear(self.hidden_channels + len(TIME_FEATURES), _HEAD_HIDDEN_CHANNELS)  # 按原口径初始化量价及两个时钟列。
            nn.init.xavier_uniform_(hidden.weight)  # 初始化时钟与量价共同使用的隐藏层。
            nn.init.zeros_(hidden.bias)  # 隐藏层初始偏置置零。
            self.head = nn.Sequential(hidden, nn.LeakyReLU(_NEGATIVE_SLOPE), output)  # 门控模式随后只保留对应量价的头部权重。
        else:  # 兼容纯量价和原八通道模型的检查点结构。
            self.head = output  # 原模型直接用十六维编码预测收益。
        self._configure_clock_gate(use_gate)  # 保留旧头部兼容性，只为单位置门控补充结构。

    @staticmethod  # 架构约束只依赖构造参数。
    def _validate_structure(input_channels: int, time_at_head: bool, mode: str, width: int) -> None:  # 拒绝未经确认的架构组合。
        """核对量价输入通道、头部时钟、门控模式及卷积宽度。配置无效时抛出 ValueError。"""
        if not isinstance(input_channels, int) or input_channels not in (6, 8):  # 检查项目定义的通道数。
            raise ValueError("输入通道数须为 6 或 8")  # 明确报告两种输入布局。
        if time_at_head and input_channels != len(FEATURES):  # 后融合的卷积只接受已有量价特征。
            raise ValueError("时钟后融合要求六个量价输入通道")  # 保持原检查点的输入约定。
        if mode not in ("none", "gate"):  # 正式模型仅提供一种逐根时钟接入方式。
            raise ValueError("逐根时钟模式须为 none 或 gate")  # 拒绝未知模式。
        if mode == "gate" and (time_at_head or input_channels != len(FEATURES)):  # 保持单位置时钟和六通道原始输入。
            raise ValueError("时钟门控要求六列量价并关闭预测头时钟")  # 拒绝和旧时钟架构混合。
        if type(width) is not int or width <= 0:  # 正整数卷积宽度不接受布尔值或非正数。
            raise ValueError("卷积宽度须为正整数")  # 阻止无效卷积维数。
        if mode != "gate" and width != _DEFAULT_HIDDEN_CHANNELS:  # 仅对正式门控实验增加可配宽度。
            raise ValueError("只有时钟门控模式可以改变卷积宽度")  # 保留旧模型十六通道结构。

    def _configure_clock_gate(self, use_gate: bool) -> None:  # 初始化门控模式新增的逐根乘法层。
        """use_gate 为 True 时，首块输出可乘以逐根时钟生成的门值。"""
        self.clock_gate: nn.Linear | None = None  # 无门控的模型没有额外参数。
        if not use_gate:  # 原线性头和后融合头照常使用。
            return  # 保持旧状态结构和前向接口。
        with torch.random.fork_rng(devices=[]), torch.no_grad():  # 保存公共初始化消耗的 CPU 随机状态。
            assert isinstance(self.head, nn.Sequential)  # 门控模型使用同样的非线性回归头模板。
            source = self.head[0]  # 读取原方式初始化的公共隐藏层权重和偏置。
            hidden = nn.Linear(self.hidden_channels, _HEAD_HIDDEN_CHANNELS)  # 只接受卷积输出，不向头部传时钟。
            hidden.weight.copy_(source.weight[:, :self.hidden_channels])  # 使用原初始化中与量价对应的列。
            hidden.bias.copy_(source.bias)  # 保留相同的隐藏层偏置。
            self.head[0] = hidden  # 确保预测头没有时钟列或占位值。
            gate = nn.Linear(len(TIME_FEATURES), self.hidden_channels)  # 每根时钟映射到全部卷积通道。
            nn.init.zeros_(gate.weight)  # 门控权重固定从零开始。
            nn.init.zeros_(gate.bias)  # 加一后的初始乘数恒等于一。
            self.clock_gate = gate  # 保存逐根可训练的乘法门。

    @classmethod  # 供训练自检和独立预测共同恢复相同的模型输入口径。
    def from_checkpoint(cls, checkpoint: dict) -> "ReturnTCN":  # 传入已读取的正式检查点。
        """按已保存的配置和张量严格恢复模型。预测前由调用者切换到 eval 模式。"""
        config, state = checkpoint["config"], checkpoint["model"]  # 选取模型规格与持久化权重。
        gate_mode = config.get("window_clock_mode", "none")  # 旧检查点没有新增的卷积分支。
        if gate_mode == "gate" and (config.get("time_fusion") != "gate_only" or config.get("head_features", [])):  # 核对单位置门控的声明。
            raise ValueError("时钟门控检查点须声明门控融合并关闭预测头时钟")  # 拒绝双接入元数据。
        prices = state.get("minute_price_multipliers")  # 旧格式缺少倍率表时使用单位倍率。
        if "center_positions" in config:  # 新格式明确记录倍率模式，要求声明的表存在。
            prices = None if config["price_input_scale_mode"] == "none" else state["minute_price_multipliers"]  # 单位倍率无需缓冲区。
        scales = state["minute_return_scales"] if config.get("label_scale_mode") == "prediction_minute" else state.get("minute_return_scales")  # 恢复已声明的标签尺度。
        model = cls(  # 按训练时声明的结构构造，确保多余状态也会在严格加载时被发现。
            input_channels=config.get("input_channels", len(FEATURES)),  # 早期检查点默认为六列量价。
            time_at_head=config.get("time_fusion") == "late",  # 恢复时钟接入位置。
            position_means=state["position_means"] if config.get("center_positions", False) else None,  # 新口径必须带逐位置均值。
            minute_price_multipliers=prices, minute_return_scales=scales,  # 重用训练阶段冻结的输入和标签表。
            window_clock_mode=gate_mode,  # 恢复卷积分支的时钟模式。
            hidden_channels=config.get("hidden_channels", _DEFAULT_HIDDEN_CHANNELS),  # 旧十六宽度门控也可严格加载。
        )  # 完成模型结构重建。
        model.load_state_dict(state, strict=True)  # 核对所有权重和持久化缓冲区。
        return model  # 返回可用于进一步训练或预测的模型。

    def encode(self, x: Tensor, minutes: Tensor | None = None) -> Tensor:  # 提取每个窗口的时序表示。
        """x 为 [批大小, input_channels, 30] 的特征，按分钟从早到晚排列。

        启用位置均值、分时价格倍率或时钟门控时，minutes 为与 x 对齐的一维北京时间预测分钟整数。
        返回 [批大小, hidden_channels] 张量。通道、时间轴、维数或分钟不符合约定时抛出 ValueError。
        """
        if x.ndim != 3 or x.shape[1:] != (self.input_channels, WINDOW):  # 在进入卷积前检查批次布局。
            raise ValueError(  # 报告当前模型要求的通道数与实际形状。
                f"预期输入形状为 [批大小, {self.input_channels}, {WINDOW}]，实际为 {tuple(x.shape)}"  # 列出两种形状。
            )  # 终止通道数或分钟数不符的调用。

        clock = None  # 只在时钟门控模式准备逐根时钟。
        transform_inputs = self.position_means is not None or self.minute_price_multipliers is not None  # 判断是否需要正式量价变换。
        if transform_inputs or self.window_clock is not None:  # 输入变换及时钟门控使用同一预测分钟索引。
            indices = self._minute_indices(minutes)  # 查询同一设备上的日内分钟索引。
            if indices.shape != x.shape[:1]:  # 每个窗口必须有且仅有一个预测分钟。
                raise ValueError("预测分钟须与输入批次的样本数一致")  # 拒绝不对齐的分钟。
            if transform_inputs:  # 保留既有的先去均值再缩放顺序。
                prices = x[:, :PRICE_CHANNELS]  # 按既定顺序取得四列价格。
                turnover = x[:, PRICE_CHANNELS:len(FEATURES)]  # 取得成交量和成交额。
                if self.position_means is not None:  # 按预测分钟选择各窗内位置的期望值。
                    means = self.position_means.index_select(0, indices).transpose(1, 2)  # 对齐为 [批大小, 3, 30]。
                    prices = prices - means[:, :1]  # 四列价格共用收盘位置均值。
                    turnover = turnover - means[:, 1:]  # 量额各自减去相同位置的均值。
                if self.minute_price_multipliers is not None:  # 在去均值完成后应用可选的价格倍率。
                    factors = self.minute_price_multipliers.index_select(0, indices)  # 按预测分钟取实际倍率。
                    prices = prices * factors[:, None, None]  # 整个窗口的四列价格共用乘数。
                x = torch.cat((prices, turnover, x[:, len(FEATURES):]), dim=1)  # 重新合并量价和未作变换的时钟通道。
            if self.window_clock is not None:  # 为门控查出窗口各根的两个时钟值。
                assert self.clock_rows is not None  # 构造门控时已计算合法预测分钟映射。
                rows = self.clock_rows.index_select(0, indices)  # 找出对应窗口的时钟行。
                clock = self.window_clock.index_select(0, rows)  # 得到 [批大小, 2, 30] 的固定时钟。
        features = self.minute_block(x)  # 先执行整个首块，得到主路和捷径之和。
        if self.clock_gate is not None:  # 在首块之后、下采样之前乘以逐根门值。
            assert clock is not None  # 已配置门层时一定先查出时钟。
            gate = self.clock_gate(clock.transpose(1, 2)).transpose(1, 2)  # 将各根两个时钟变成与隐藏通道对齐的线性项。
            features = features * (1 + gate)  # 纯乘法调节，不向量价编码叠加时钟。
        features = self.downsample_pairs(features)  # 每两个位置汇总一次，保留十五个位置。
        features = self.two_minute_block(features)  # 保留十五个位置并计算该时间尺度的残差特征。
        features = self.downsample_triples(features)  # 每三个位置汇总一次，得到五个位置。
        return self.collapse_time(features).squeeze(-1)  # 聚合窗口并去掉长度为一的时间轴。

    @property  # 统一训练和推理对分钟索引的要求。
    def requires_prediction_minutes(self) -> bool:  # 判断前向是否需要每条样本的钟表分钟。
        """时钟、输入去均值、价格倍率或收益率还原需要预测分钟时返回 True。"""
        tables = (self.head_clock, self.window_clock, self.position_means, self.minute_price_multipliers, self.minute_return_scales)  # 收集所有按预测分钟查询的固定表。
        return any(table is not None for table in tables)  # 任何一张表可用即要求调用方提供时点。

    def _minute_indices(self, minutes: Tensor | None) -> Tensor:  # 校验分钟张量并转换为模型设备上的索引。
        """minutes 须为一维整数张量，无效输入抛出 ValueError。"""
        if minutes is None or minutes.ndim != 1 or minutes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):  # 分钟须为一维整数。
            raise ValueError("按时点处理时须提供一维整数预测分钟张量")  # 明确报告索引输入要求。
        return minutes.to(device=self.return_scale.device, dtype=torch.int64)  # 在同一设备复用时钟和尺度索引。

    def scale_for(self, minutes: Tensor | None = None) -> Tensor:  # 取得预测所需的收益率还原尺度。
        """minutes 为一维整数张量，按样本给出北京时间自午夜起的分钟数，范围 0–1439。

        全局模式返回单个标量，无须提供分钟。按分钟模式返回同长度的尺度，缺少
        分钟或类型和维数错误时抛出 ValueError，越界位置由 PyTorch 索引拒绝。
        """
        table = self.minute_return_scales  # 读取构造时已校验的分钟尺度表。
        if table is None:  # 旧模式用同一全局尺度适配所有样本。
            return self.return_scale  # 返回无需展开的标量张量。
        return table.index_select(0, self._minute_indices(minutes))  # 按样本顺序取值，负索引不会回绕。

    def forward(self, x: Tensor, minutes: Tensor | None = None) -> Tensor:  # 输出每条样本的物理收益率。
        """x 为 [批大小, input_channels, 30] 的特征，返回 [批大小] 的对数收益率预测。

        使用时钟、位置均值、价格倍率或收益率分钟尺度时，须提供与 x 逐行对齐的北京时间
        预测分钟。缺少时点或批次形状不匹配时抛出 ValueError。
        """
        if self.requires_prediction_minutes:  # 所有按时点处理都使用同一组索引。
            minutes = self._minute_indices(minutes)  # 提前校验并将分钟与模型放在同一设备。
            if minutes.shape != x.shape[:1]:  # 防止把某样本的时钟或分钟尺度用在另一条样本上。
                raise ValueError("预测分钟须与输入批次的样本数一致")  # 报告特征与时点未逐行对齐。
        if self.position_means is not None or self.minute_price_multipliers is not None or self.window_clock is not None:  # 输入变换或时钟门控需要预测分钟。
            features = self.encode(x, minutes)  # 先完成输入变换，再卷积和时序汇总。
        else:  # 旧配置继续使用不接收时点的编码调用。
            features = self.encode(x)  # 直接提取量价时序表示。
        if self.head_clock is not None:  # 后融合模型在预测头补充两个时钟值。
            assert self.clock_rows is not None and minutes is not None  # 构造时同时提供查表索引与时钟值。
            rows = self.clock_rows.index_select(0, minutes)  # 取得各条样本的时钟行。
            clock = self.head_clock.index_select(0, rows)  # 读取时钟值，索引拒绝缺少完整窗口的分钟。
            features = torch.cat((features, clock), dim=1)  # 拼接十六个量价表示和两个时钟值。
        scores = self.head(features).squeeze(-1)  # 每个窗口输出一个归一化收益率分数。
        return scores * self.scale_for(minutes)  # 将分数映射为物理对数收益率。
