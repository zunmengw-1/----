"""Temporal convolutional model for one return prediction per sample."""

from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


_NEGATIVE_SLOPE = 0.01


def _main_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    conv = nn.Conv1d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
    )
    nn.init.kaiming_normal_(
        conv.weight, a=_NEGATIVE_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
    )
    nn.init.zeros_(conv.bias)
    return nn.Sequential(weight_norm(conv, dim=0), nn.LeakyReLU(_NEGATIVE_SLOPE))


class _ResidualBlock(nn.Module):
    """Two causal convolutions with dilations of one and two time steps."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = _main_conv(in_channels, out_channels, kernel_size=3)
        self.conv2 = _main_conv(out_channels, out_channels, kernel_size=3, dilation=2)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.shortcut = projection

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv1(F.pad(x, (2, 0)))
        features = self.conv2(F.pad(features, (4, 0)))
        return features + self.shortcut(x)


class ReturnTCN(nn.Module):
    """Map 30 time-ordered, 20-channel observations to one return score."""

    def __init__(self) -> None:
        super().__init__()
        self.minute_block = _ResidualBlock(20, 16)
        self.downsample_pairs = _main_conv(16, 16, kernel_size=2, stride=2)
        self.two_minute_block = _ResidualBlock(16, 16)
        self.downsample_triples = _main_conv(16, 16, kernel_size=3, stride=3)
        self.collapse_time = _main_conv(16, 16, kernel_size=5)
        self.head = nn.Linear(16, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, x: Tensor) -> Tensor:
        """Return one 16-dimensional feature vector per input window."""
        if x.ndim != 3 or x.shape[1:] != (30, 20):
            raise ValueError(f"expected input shape [B, 30, 20], got {tuple(x.shape)}")

        features = self.minute_block(x.transpose(1, 2))
        # Non-overlapping pairs end at minutes 2, 4, ..., 30 of the window.
        features = self.downsample_pairs(features)
        features = self.two_minute_block(features)
        features = self.downsample_triples(features)
        return self.collapse_time(features).squeeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x)).squeeze(-1)
