"""用北京时间连续竞价时段核对两个新增时间通道的数值和模型布局。"""

import numpy as np  # 构造预测分钟数组并核对时间特征。
import pytest  # 对无效时钟输入执行参数化检查。
import torch  # 构造量价张量并检查模型输入拼接。

from src.features.samples import (  # 使用采样侧统一的时钟特征接口。
    TIME_FEATURES,  # 核对两个时间通道的顺序。
    time_features,  # 按预测时点构造每根分钟线的时间特征。
)  # 完成样本接口导入。
from src.models.training_data import (  # 使用模型输入组装模块。
    prepare_model_inputs,  # 复用训练和验证共用的输入拼接接口。
)  # 完成模型接口导入。


def test_time_features_follow_each_minute_and_reset_after_lunch():  # 核对两个时段首个和较晚的预测窗口。
    """无需输入，从手算的窗口时点核对列顺序、分钟顺序和每个时段的比例。"""
    minutes = np.array([13 * 60 + 30, 10 * 60, 14 * 60 + 52, 11 * 60 + 25])  # 故意让样本时间打乱并包含两个时段。
    clock = time_features(minutes)  # 按原样本顺序构造每分钟的时间特征。
    assert TIME_FEATURES == ("is_afternoon", "session_progress")  # 下午标记位于时段进度之前。
    assert clock.shape == (4, 2, 30) and clock.dtype == np.float32  # 时间轴、通道轴和类型与模型约定一致。
    np.testing.assert_array_equal(clock[:, 0], np.repeat([[1], [0], [1], [0]], 30, axis=1))  # 一个窗口的三十根行情共用时段标记。
    expected = np.stack(  # 分别计算从每个窗口的首根至末根行情的时段进度。
        (  # 采用手算时段长度和窗口位置。
            np.arange(1, 31) / 117,  # 下午首个有效历史窗口从 13:01 起重新计数。
            np.arange(1, 31) / 120,  # 上午首个有效历史窗口从 09:31 起计数。
            np.arange(83, 113) / 117,  # 下午 14:52 对应本时段第 112 根分钟线。
            np.arange(86, 116) / 120,  # 上午 11:25 对应本时段第 115 根分钟线。
        ),  # 完成四个窗口的期望进度。
    ).astype(np.float32)  # 使用与模型输入相同的单精度对比。
    np.testing.assert_array_equal(clock[:, 1], expected)  # 确认窗口内逐分钟前进，午休后重新开始。


@pytest.mark.parametrize("minute", [-1, 9 * 60 + 59, 11 * 60 + 31, 13 * 60, 13 * 60 + 29, 14 * 60 + 58, 1440])  # 覆盖缺历史、非连续竞价和日期外分钟。
def test_time_features_reject_invalid_clock(minute):  # 检查无效预测时点不会生成跨时段窗口。
    """给定无法匹配完整历史窗口的时钟分钟，要求抛出 ValueError。"""
    with pytest.raises(ValueError, match="Prediction minutes"):  # 时间无效时报告清晰的异常。
        time_features(np.array([10 * 60, minute]))  # 即使同批还有有效分钟，也必须及时发现错误。


@pytest.mark.parametrize(  # 时间输入必须是对齐样本行的一维整数数组。
    "minutes",  # 作为时钟特征的预测分钟输入。
    [np.array([[600]]), np.array([600.0]), np.array(["2025-01-02T10:00"], dtype="datetime64[m]")],  # 检查多维、浮点和未换算的时间戳。
)  # 分别执行不同的输入类型检查。
def test_time_features_requires_vector_of_integer_minutes(minutes):  # 保持时钟输入定义明确。
    """传入非一维整数的分钟数组，要求抛出 ValueError。"""
    with pytest.raises(ValueError, match="one-dimensional integer array"):  # 拒绝会隐式改变时间精度的输入。
        time_features(minutes)  # 将错误形状或精度的数组交给正式接口。


def test_prepare_model_inputs_preserves_market_channels_and_appends_clock():  # 核对新旧模型输入顺序和长度。
    """无需输入，比较和真实 DataLoader 同布局的量价批次及其时间通道。"""
    market = torch.arange(2 * 30 * 6, dtype=torch.float32).reshape(2, 30, 6).transpose(1, 2)  # 构造非连续的六通道张量。
    minutes = np.array([10 * 60, 13 * 60 + 30])  # 依次让两个样本处于上午和下午。
    assert prepare_model_inputs(market, np.empty(0, dtype=np.intp), 6) is market  # 六通道保持同一个张量，无需时钟信息。
    expanded = prepare_model_inputs(market, minutes, 8)  # 拼接两个已确认的时间通道。
    assert expanded.shape == (2, 8, 30) and expanded.device == market.device and expanded.dtype == market.dtype  # 核对拼接后的规格。
    torch.testing.assert_close(expanded[:, :6], market, rtol=0, atol=0)  # 六个量价通道的数值和顺序保持原样。
    torch.testing.assert_close(expanded[:, 6:], torch.from_numpy(time_features(minutes)), rtol=0, atol=0)  # 追加下午标记和逐分钟进度。
    with pytest.raises(ValueError, match="same batch size"):  # 防止预测时间与特征的行数错位。
        prepare_model_inputs(market, minutes[:1], 8)  # 两条量价输入只提供一个预测时点。
    with pytest.raises(ValueError, match="must be 6 or 8"):  # 拒绝未经定义的通道数。
        prepare_model_inputs(market, minutes, 7)  # 无法按七通道口径组装输入。
