"""核对训练入口默认采用通道加权、宽度传递以及历史结构的选择。"""

import sys  # 控制一次参数解析看到的命令行。

import pytest  # 构造默认、增宽和历史配置的回归用例。

from scripts.train_tcn import _parse_args  # 验证实际训练命令使用的入口。


@pytest.mark.parametrize(("options", "fusion", "width"), [([], "gate_only", 16), (["--hidden-channels", "32"], "gate_only", 32), (["--clock-mode", "head"], "late", 16), (["--channels", "6"], "none", 16), (["--channels", "8"], "early", 16)])  # 覆盖默认和可复现的历史调用。
def test_clock_mode_and_width(monkeypatch, options, fusion, width):  # 核对 CLI 模式、宽度和输入特征开关。
    """options 为附加命令参数，fusion 和 width 分别为期望的时钟接入位置与卷积宽度。"""
    monkeypatch.setattr(sys, "argv", ["train_tcn.py", "--pilot", "--output", "trial", *options])  # 使用真实命令布局。
    args = _parse_args()  # 仅解析参数，无需训练数据。
    assert (args.time_fusion, args.hidden_channels) == (fusion, width)  # 核对控制模型构造的两个参数。
    assert args.center_positions and args.price_scale == "minute"  # 增宽继续使用既定的特征处理。


@pytest.mark.parametrize("options", [["--hidden-channels", "0"], ["--channels", "8", "--hidden-channels", "32"]])  # 拒绝不支持的架构宽度。
def test_invalid_width(monkeypatch, options):  # 验证训练前即可定位错误。
    """options 为请求不合法宽度的一组命令参数。"""
    monkeypatch.setattr(sys, "argv", ["train_tcn.py", "--output", "trial", *options])  # 送入实际入口。
    with pytest.raises(SystemExit):  # argparse 在读取数据前终止。
        _parse_args()  # 使用正式参数校验。
