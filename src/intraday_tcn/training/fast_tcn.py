"""Time-major matrix multiplication for the TCN's short CPU convolutions."""

from torch import Tensor, nn
from torch.nn import functional as F

from intraday_tcn.training.model import ReturnTCN


def _temporal_layer(x: Tensor, layer: nn.Sequential, *, causal: bool = False) -> Tensor:
    conv, activate = layer
    (kernel,) = conv.kernel_size
    (stride,) = conv.stride
    (dilation,) = conv.dilation
    reach = (kernel - 1) * dilation
    if causal:
        x = F.pad(x, (0, 0, reach, 0))

    # Put channels next to each tap so each window becomes one GEMM row.
    taps = x.unfold(1, reach + 1, stride)[..., ::dilation].transpose(-1, -2)
    weight = conv.weight.transpose(1, 2).flatten(1)
    return activate(F.linear(taps.flatten(-2), weight, conv.bias))


def _causal_path(x: Tensor, first: nn.Sequential, second: nn.Sequential) -> Tensor:
    x = _temporal_layer(x, first, causal=True)
    return _temporal_layer(x, second, causal=True)


class FastReturnTCN(ReturnTCN):
    """Evaluate the same TCN weights and residuals with a CPU-friendly layout."""

    def encode(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (30, 20):
            raise ValueError(f"expected input shape [B, 30, 20], got {tuple(x.shape)}")

        first = self.minute_block
        shortcut = first.shortcut
        features = _causal_path(x, first.conv1, first.conv2) + F.linear(
            x, shortcut.weight.squeeze(-1), shortcut.bias
        )
        features = _temporal_layer(features, self.downsample_pairs)

        second = self.two_minute_block
        features = _causal_path(features, second.conv1, second.conv2) + features
        features = _temporal_layer(features, self.downsample_triples)
        return _temporal_layer(features, self.collapse_time).squeeze(1)
