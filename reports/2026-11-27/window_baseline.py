"""Frozen 30-bar flat-window models. Calling the model returns physical log return."""
import torch
from torch import nn

TABLES = ("position_means", "minute_price_multipliers", "minute_return_scales",
          "return_scale", "clock_rows", "window_clock")


class WindowBaseline(nn.Module):
    input_channels = 6
    requires_prediction_minutes = True

    def __init__(self, kind, tables):
        super().__init__()
        self.kind = kind
        for name in TABLES:
            self.register_buffer(name, tables[name].detach().clone())
        if kind == "mlp24":
            self.readout = nn.Sequential(nn.Linear(240, 24), nn.LeakyReLU(0.01), nn.Linear(24, 1))
        elif kind == "linear":
            self.readout = nn.Linear(240, 1)
        else:
            raise ValueError(f"Unknown baseline {kind}")
        for layer in self.readout.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def features(self, x, minutes):
        if x.shape[1:] != (6, 30) or minutes.shape != x.shape[:1]:
            raise ValueError("Expected [B,6,30] and [B] aligned input tensors")
        means = self.position_means.index_select(0, minutes).transpose(1, 2)
        prices = (x[:, :4] - means[:, :1]) * self.minute_price_multipliers.index_select(0, minutes)[:, None, None]
        turnover = x[:, 4:] - means[:, 1:]
        clock = self.window_clock.index_select(0, self.clock_rows.index_select(0, minutes))
        return torch.cat((prices, turnover, clock), dim=1).flatten(1)

    def scale_for(self, minutes):
        return self.minute_return_scales.index_select(0, minutes)

    def forward(self, x, minutes):
        return self.readout(self.features(x, minutes)).squeeze(1) * self.scale_for(minutes)


def load_baseline(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = WindowBaseline(saved["kind"], saved["model"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()
