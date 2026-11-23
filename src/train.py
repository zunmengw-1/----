"""Train the shared minute TCN and fit regularized stock-specific heads."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn import functional as F
from torch.nn.utils.parametrize import cached

from data import history_mainboard_eligibility, load_bars, pilot_eligibility, read_dates, read_eligibility
from evaluation import daily_metrics, summarize_daily, write_daily
from features import FEATURE_NAMES, build_samples, encode_windows, find_targets
from model import ReturnTCN


ROOT = Path(__file__).resolve().parents[1]
RAW_HISTORY = ROOT / "data/questdb-trial/dbroot/db/stock_minute_history~16"
DAILY_SUMMARY = ROOT / "市场微观结构研究/cache/daily_summary.parquet"
CONTINUOUS = np.array([*range(5), *range(7, 17)])
WINDOW, SIGNALS_PER_STOCK_DAY = 30, 169


@dataclass(frozen=True)
class Scaler:
    center: np.ndarray
    scale: np.ndarray
    label_scale: float

    @classmethod
    def fit(cls, windows, label_count, label_sum, label_square_sum):
        features = np.concatenate(windows).reshape(-1, len(FEATURE_NAMES))
        center = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        scale = np.ones_like(center)
        for column in CONTINUOUS:
            values = features[:, column]
            values = values[np.isfinite(values)]
            if len(values):
                low, median, high = np.quantile(values, (0.25, 0.5, 0.75))
                center[column] = median
                width = high - low
                if width == 0:
                    width = values.std(dtype=np.float64)
                scale[column] = width or 1.0
        variance = label_square_sum / label_count - (label_sum / label_count) ** 2
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("训练标签需要正且有限的方差")
        return cls(center, scale, float(np.sqrt(variance)))

    def transform(self, x):
        x = (x - self.center) / self.scale
        return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def as_dict(self):
        return {
            "feature_names": list(FEATURE_NAMES), "center": self.center.tolist(),
            "scale": self.scale.tolist(), "label_scale_bp": self.label_scale,
        }


@dataclass(frozen=True)
class Shard:
    symbol: str
    start: int
    n: int

    @property
    def rows(self):
        return slice(self.start, self.start + self.n)


def sample_training_rows(coords, cap, by_day, rng):
    size = len(coords)
    if not cap or size <= cap:
        return np.arange(size)
    if not by_day:
        return np.sort(rng.choice(size, cap, replace=False))

    _, first, counts = np.unique(coords[:, 0], return_index=True, return_counts=True)
    if len(first) > cap:
        days = rng.choice(len(first), cap, replace=False)
        first, counts = first[days], counts[days]
    core = first + rng.integers(counts)
    remaining = cap - len(core)
    if remaining:
        candidates = np.ones(size, dtype=bool)
        candidates[core] = False
        extra = rng.choice(np.flatnonzero(candidates), remaining, replace=False)
        core = np.r_[core, extra]
    return np.sort(core)


def prepare_training(provider, universe, training_dates, scratch, cap, by_day, rng):
    shards, scaling_windows = [], []
    kept, total, total_sum, total_square_sum = 0, 0, 0.0, 0.0
    represented_stock_days = 0
    scaling_windows_per_stock = max(1, 4_096 // len(universe))
    eligible_days = [int(mask[:len(training_dates)].sum()) for mask in universe.values()]
    capacity = sum(min(n * SIGNALS_PER_STOCK_DAY, cap) if cap else n * SIGNALS_PER_STOCK_DAY
                   for n in eligible_days)
    if not capacity:
        raise ValueError("没有合资格的训练股票日")
    # Keep a packed disk-backed sample matrix so mixed-stock batches need one read.
    x_store = np.lib.format.open_memmap(
        scratch / "features.npy", mode="w+", dtype=np.float32,
        shape=(capacity, WINDOW, len(FEATURE_NAMES)),
    )
    y_store = np.lib.format.open_memmap(scratch / "labels.npy", mode="w+", dtype=np.float32, shape=(capacity,))

    for index, (symbol, eligible) in enumerate(universe.items(), 1):
        dates = [day for day, valid in zip(training_dates, eligible) if valid]
        bars = load_bars(provider, symbol, dates)
        y, coords = find_targets(bars)
        candidates = len(y)
        if candidates:
            total += candidates
            total_sum += float(y.sum(dtype=np.float64))
            total_square_sum += float(np.square(y, dtype=np.float64).sum())
            scaling_idx = rng.choice(candidates, min(candidates, scaling_windows_per_stock), replace=False)
            chosen = sample_training_rows(coords, cap, by_day, rng)
            represented_stock_days += int(np.unique(coords[chosen, 0]).size)
            needed = np.union1d(chosen, scaling_idx)
            encoded = encode_windows(bars, coords[needed])
            scaling_windows.append(encoded[np.searchsorted(needed, scaling_idx)])
            shard = Shard(symbol, kept, len(chosen))
            x_store[shard.rows] = encoded[np.searchsorted(needed, chosen)]
            y_store[shard.rows] = y[chosen]
            kept += shard.n
            shards.append(shard)

        if index == 1 or index % 250 == 0 or index == len(universe):
            print(f"准备训练样本 {index}/{len(universe)}，累计有效窗口 {total:,}，入训 {kept:,}", flush=True)

    if not shards:
        raise ValueError("没有符合条件的训练样本")
    scaler = Scaler.fit(scaling_windows, total, total_sum, total_square_sum)
    return shards, x_store[:kept], y_store[:kept], scaler, total, represented_stock_days


def shared_fit(x_store, y_store, scaler, args, device, rng):
    if args.fast_cpu:
        from fast_tcn import FastReturnTCN

        model = FastReturnTCN()
    else:
        model = ReturnTCN()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses = []
    for epoch in range(args.epochs):
        shuffled = rng.permutation(len(y_store))
        total_loss = torch.zeros((), device=device)
        for first in range(0, len(shuffled), args.batch_size):
            index = shuffled[first : first + args.batch_size]
            xb = torch.from_numpy(scaler.transform(x_store[index])).to(device)
            yb = torch.from_numpy(y_store[index] / scaler.label_scale).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.huber_loss(model(xb), yb, delta=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(index)
            if first and first % (200 * args.batch_size) == 0:
                print(f"共享训练 {epoch + 1}/{args.epochs}，{first:,}/{len(shuffled):,} 窗口", flush=True)
        mean_loss = float(total_loss.cpu()) / len(shuffled)
        if not np.isfinite(mean_loss):
            raise RuntimeError(f"第 {epoch + 1} 轮训练出现非有限损失")
        losses.append(mean_loss)
        print(f"共享训练 {epoch + 1}/{args.epochs}，Huber {mean_loss:.6f}", flush=True)
    return model.eval(), losses


@torch.inference_mode()
def encode(model, x, scaler, batch_size, device):
    features = np.empty((len(x), 16), dtype=np.float64)
    for first in range(0, len(x), batch_size):
        batch = slice(first, first + batch_size)
        features[batch] = model.encode(torch.from_numpy(scaler.transform(x[batch])).to(device)).cpu().numpy()
    return features


def fit_stock_head(design, labels, shared, penalty):
    """Fit the 17 head parameters against a frozen feature matrix."""
    def objective(theta):
        error = design @ theta - labels
        magnitude = np.abs(error)
        huber = np.mean(np.where(magnitude <= 1, 0.5 * error**2, magnitude - 0.5))
        diff = theta - shared
        loss = huber + penalty * np.dot(diff, diff)
        gradient = design.T @ np.clip(error, -1, 1) / len(labels) + 2 * penalty * diff
        return loss, gradient

    result = minimize(objective, shared, method="L-BFGS-B", jac=True, options={"maxiter": 200})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("单股预测头拟合产生非有限参数")
    return result.x, bool(result.success)


def evaluate_development(model, shards, x_store, y_store, scaler, universe, dates, train_count, args, device):
    shared = np.r_[model.head.weight.detach().cpu().numpy().ravel(), model.head.bias.item()].astype(np.float64)
    shared_daily = []
    penalties = [] if args.shared_only else args.penalties
    adapted_daily = {penalty: [] for penalty in penalties}
    heads = {penalty: {} for penalty in penalties}
    fit_warnings = {penalty: [] for penalty in penalties}
    by_symbol = {shard.symbol: shard for shard in shards}
    symbols = [symbol for symbol, mask in universe.items() if mask[train_count:].any()]
    if args.eval_stock_cap and len(symbols) > args.eval_stock_cap:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
        symbols = sorted(rng.choice(symbols, args.eval_stock_cap, replace=False).tolist())
    zero_abs, zero_square, zero_n = 0.0, 0.0, 0
    for index, symbol in enumerate(symbols, 1):
        shard = by_symbol.get(symbol)
        if penalties and shard is not None:
            # All updates to stock weights use this stock's training-period windows.
            x_train, y_train = x_store[shard.rows], y_store[shard.rows]
            encoded = encode(model, x_train, scaler, args.batch_size, device)
            design = np.column_stack((encoded, np.ones(len(y_train))))
            labels = np.asarray(y_train, dtype=np.float64) / scaler.label_scale
            for penalty in penalties:
                theta, converged = fit_stock_head(design, labels, shared, penalty)
                heads[penalty][symbol] = theta
                if not converged:
                    fit_warnings[penalty].append(symbol)

        test_dates = [day for day, valid in zip(dates[train_count:], universe[symbol][train_count:]) if valid]
        bars = load_bars(args.provider, symbol, test_dates)
        x, actual, coords = build_samples(bars, sample_every=args.eval_sample_every)
        z = np.column_stack((encode(model, x, scaler, args.batch_size, device), np.ones(len(x))))
        shared_daily.extend(daily_metrics(symbol, test_dates, coords, (z @ shared) * scaler.label_scale, actual))
        zero_abs += float(np.abs(actual).sum(dtype=np.float64))
        zero_square += float(np.square(actual, dtype=np.float64).sum())
        zero_n += len(actual)
        for penalty in penalties:
            predicted = (z @ heads[penalty].get(symbol, shared)) * scaler.label_scale
            adapted_daily[penalty].extend(daily_metrics(symbol, test_dates, coords, predicted, actual))
        if index == 1 or index % 100 == 0 or index == len(symbols):
            print(f"开发测试 {index}/{len(symbols)} 只，累计 {zero_n:,} 条预测", flush=True)

    eligible_days = sum(int(universe[symbol][train_count:].sum()) for symbol in symbols)
    shared_summary = summarize_daily(shared_daily, eligible_days, sample_every=args.eval_sample_every)
    write_daily(args.out / "shared_daily.csv", shared_daily)
    zero = {"n": zero_n, "mae_bp": zero_abs / zero_n if zero_n else None,
            "rmse_bp": float(np.sqrt(zero_square / zero_n)) if zero_n else None}
    if not penalties:
        return shared_summary, zero, symbols, {}, None, {}, []

    summary = {param: summarize_daily(rows, eligible_days, sample_every=args.eval_sample_every)
               for param, rows in adapted_daily.items()}
    has_ic = [param for param, result in summary.items() if result["mean_rank_ic"] is not None]
    if not has_ic:
        raise ValueError("开发测试期没有足够的有效样本计算 Rank IC")
    selected = max(has_ic, key=lambda param: (summary[param]["mean_rank_ic"], param))
    write_daily(args.out / "adapted_daily.csv", adapted_daily[selected])
    return shared_summary, zero, symbols, summary, selected, heads[selected], fit_warnings[selected]


def save_checkpoint(path, model, scaler, heads):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "stock_heads": {symbol: torch.as_tensor(theta, dtype=torch.float64) for symbol, theta in heads.items()},
        "scaler": scaler.as_dict(),
    }, temporary)
    temporary.replace(path)


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("需要正整数")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, default=ROOT / "data/qlib_cn_1min")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--eligibility", type=Path, help="外部核验的 date,symbol 逐日股票池 CSV")
    source_group.add_argument("--history-mainboard", action="store_true", help="用历史简称和成交近似筛选广覆盖主板")
    parser.add_argument("--daily-source", type=Path, default=DAILY_SUMMARY if DAILY_SUMMARY.is_file() else RAW_HISTORY)
    parser.add_argument("--pilot-stocks", type=positive_int, default=12)
    parser.add_argument("--out", type=Path, required=True, help="新的输出目录")
    parser.add_argument("--epochs", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument(
        "--train-cap-per-stock", type=int,
        help="每股训练窗口上限，默认小样本 8000、广覆盖 256，0 表示全量",
    )
    parser.add_argument("--shared-only", action="store_true", help="只联合训练共享模型")
    parser.add_argument("--eval-sample-every", type=positive_int, default=1, help="两个交易时段各自锚定的开发测试步长")
    parser.add_argument("--eval-stock-cap", type=int, default=0, help="开发测试的固定抽样股票数，0 表示全池")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--penalties", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--threads", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--fast-cpu", action="store_true", help="CPU 上使用等价的展开矩阵乘法执行卷积")
    args = parser.parse_args()
    if args.train_cap_per_stock is None:
        args.train_cap_per_stock = 256 if args.history_mainboard or args.eligibility else 8_000
    if args.train_cap_per_stock < 0 or args.eval_stock_cap < 0:
        parser.error("训练窗口上限和开发股票上限须为非负整数")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate 须为正有限数")
    if any(not np.isfinite(x) or x <= 0 for x in args.penalties):
        parser.error("--penalties 须为正有限数")
    args.penalties = sorted(set(args.penalties))

    dates = [day for day in read_dates(args.provider) if day.year == 2025]
    train_count, development_end = round(len(dates) * 0.7), round(len(dates) * 0.9)
    if not 0 < train_count < development_end < len(dates):
        raise ValueError("2025 年分钟日历不足以分成三段")
    active_dates = dates[:development_end]
    universe_audit = None
    if args.eligibility:
        universe = read_eligibility(args.eligibility, args.provider, active_dates)
    elif args.history_mainboard:
        universe, universe_audit = history_mainboard_eligibility(args.provider, args.daily_source, active_dates)
    else:
        universe = pilot_eligibility(args.provider, args.daily_source, active_dates, args.pilot_stocks)
    universe = {symbol: np.asarray(mask, dtype=bool) for symbol, mask in sorted(universe.items())}
    if not universe:
        raise ValueError("没有合资格的股票")
    if any(mask.shape != (len(active_dates),) for mask in universe.values()):
        raise ValueError("股票池掩码和交易日不匹配")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    if args.fast_cpu and device.type != "cpu":
        parser.error("--fast-cpu 须与 CPU 设备一起使用")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"研究股票 {len(universe)} 只，训练设备 {device}；最终验证期保留 {len(dates)-development_end} 日", flush=True)
    if universe_audit is not None:
        (args.out / "universe_audit.json").write_text(json.dumps(universe_audit, ensure_ascii=False, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix=".samples-", dir=args.out) as scratch:
        shards, x_store, y_store, scaler, total_windows, trained_stock_days = prepare_training(
            args.provider, universe, dates[:train_count], Path(scratch),
            args.train_cap_per_stock, args.history_mainboard, rng
        )
        model, losses = shared_fit(x_store, y_store, scaler, args, device, rng)
        save_checkpoint(args.out / "model.pt", model, scaler, {})
        print("共享权重已保存，进入开发测试", flush=True)
        with cached():
            shared, zero, evaluated, by_penalty, selected, heads, fit_warnings = evaluate_development(
                model, shards, x_store, y_store, scaler, universe, active_dates, train_count, args, device
            )
        del x_store, y_store

    def period(first, last):
        return {"start": dates[first].isoformat(), "end": dates[last - 1].isoformat(), "days": last - first}

    run = {
        "universe_mode": "external_eligibility" if args.eligibility else (
            "historical_mainboard_proxy" if args.history_mainboard else "fixed_mainboard_pilot"
        ),
        "universe_source": str(args.eligibility or args.daily_source),
        "symbols": list(universe), "trained_symbols": [shard.symbol for shard in shards],
        "split": {
            "train": period(0, train_count), "development_test": period(train_count, development_end),
            "sealed_final_validation": period(development_end, len(dates)),
        },
        "seed": args.seed, "device": str(device), "epochs": args.epochs,
        "threads": args.threads, "shared_only": args.shared_only, "fast_cpu": args.fast_cpu,
        "batch_size": args.batch_size, "train_cap_per_stock": args.train_cap_per_stock,
        "learning_rate": args.learning_rate, "huber_delta": 1.0,
        "training_candidate_windows": total_windows, "training_windows_used": sum(shard.n for shard in shards),
        "eligible_training_stock_days": sum(int(mask[:train_count].sum()) for mask in universe.values()),
        "training_stock_days_represented": trained_stock_days,
        "train_sample_by_day": args.history_mainboard,
        "eval_sample_every": args.eval_sample_every, "eval_stock_cap": args.eval_stock_cap,
        "development_evaluated_symbols": evaluated,
        "label_scale_bp": scaler.label_scale, "epoch_mean_huber": losses,
        "development_shared": shared, "development_zero": zero,
    }
    if selected is not None:
        run.update({
            "development_adapted_by_penalty": {str(p): s for p, s in by_penalty.items()},
            "development_selected_penalty": selected, "development_adapted_selected": by_penalty[selected],
            "selected_head_fit_warnings": fit_warnings,
        })
        save_checkpoint(args.out / "model.pt", model, scaler, heads)
    (args.out / "metrics.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"development_shared": shared, "development_zero": zero, "metrics": str(args.out / 'metrics.json')},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
