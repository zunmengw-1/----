"""Train the fixed linear and MLP baselines on the same batches as the saved Gate16."""
import json
import sys
from hashlib import sha256
from itertools import islice
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from window_baseline import TABLES, WindowBaseline, load_baseline

OUT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
GATE = Path("/private/tmp/tcn-clock-exclusive-20261127/gate")
sys.path.insert(0, str(PROJECT))
from src.data.metadata import Metadata
from src.data.minutes import MinuteLibrary
from src.features.batches import TrainingSampleCounts
from src.models.tcn import ReturnTCN
from src.models.training_data import make_training_loader


def tensor_hash(tensor):
    return sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def params_hash(model):
    h = sha256()
    for name, p in model.named_parameters():
        b = name.encode("utf-8")
        v = p.detach().cpu().numpy()
        for value in (np.int64(len(b)).tobytes(), b, np.int64(v.ndim).tobytes(),
                      np.asarray(v.shape, dtype=np.int64).tobytes(), v.dtype.str.encode("ascii"), v.tobytes()):
            h.update(value)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Matched MPS device unavailable")
    checkpoint = torch.load(GATE / "model.pt", map_location="cpu", weights_only=True)
    plan = json.loads((OUT / "evaluation_plan.json").read_text())
    assert sha256((GATE / "model.pt").read_bytes()).hexdigest() == plan["gate16_sha256"]
    cfg, old, sources = (checkpoint[k] for k in ("config", "training", "sources"))
    gate = ReturnTCN.from_checkpoint(checkpoint).eval()
    tables = {name: getattr(gate, name) for name in TABLES}
    with np.load(PROJECT / "output/tcn_position_centered_features/feature_tables.npz") as independent:
        for name in ("minute_price_multipliers", "position_means"):
            np.testing.assert_array_equal(tables[name].numpy(), independent[name])
    models = {}
    for kind in ("mlp24", "linear"):
        torch.manual_seed(cfg["seed"])
        models[kind] = WindowBaseline(kind, tables).to("mps")
    opt = {k: torch.optim.AdamW(m.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]) for k, m in models.items()}
    hashes = {k: params_hash(m) for k, m in models.items()}
    metadata_path = Path(sources["metadata"])
    assert sha256(metadata_path.read_bytes()).hexdigest() == sources["metadata_sha256"]
    cache = Path(sources["stock_day_cache"])
    assert sha256((cache / "keys.npy").read_bytes()).hexdigest() == sources["stock_day_index_sha256"]
    assert json.loads((cache / "manifest.json").read_text())["source_signature"] == sources["qlib_source_signature"]
    library = MinuteLibrary(Path(sources["qlib"]), stock_day_cache=cache)
    counts = TrainingSampleCounts()
    digest = sha256()
    total = 0
    cumulative = np.zeros(2)
    last_losses = cumulative.copy()
    last_total = 0
    segments = []
    reference = models["mlp24"]
    cpu_sample = None
    started = monotonic()
    with Metadata(json.loads(metadata_path.read_text()), metadata_path.parent) as meta:
        loader = make_training_loader(library, meta, library.cached_stock_days, rng=np.random.default_rng(cfg["seed"]), counts=counts)
        for idx, batch in enumerate(islice(loader, old["batches"]), 1):
            n = len(batch.y)
            for v in (np.int64(n).tobytes(), batch.code.tobytes(), batch.t.tobytes(), batch.y.numpy().tobytes()):
                digest.update(v)
            minutes = torch.from_numpy(batch.t.astype(np.int64) % 1440).to("mps")
            x = batch.x.to("mps")
            features = reference.features(x, minutes)
            scale = reference.scale_for(minutes)
            y = batch.y.to("mps") / scale
            if idx == 1:
                cpu_sample = (x[:128].cpu(), minutes[:128].cpu())
                gate_inputs = []
                handle = gate.minute_block.register_forward_pre_hook(lambda _, inputs: gate_inputs.append(inputs[0]))
                try:
                    with torch.no_grad():
                        gate(*cpu_sample)
                finally:
                    handle.remove()
                market_clock = features[:128].cpu().reshape(-1, 8, 30)
                torch.testing.assert_close(market_clock[:, :6], gate_inputs[0], atol=0, rtol=0)
                torch.testing.assert_close(market_clock[:, 6:], gate.window_clock[gate.clock_rows[cpu_sample[1]]], atol=0, rtol=0)
            losses = []
            for kind, model in models.items():
                opt[kind].zero_grad(set_to_none=True)
                physical = model.readout(features).squeeze(1) * scale
                loss = torch.nn.functional.huber_loss(physical / scale, y, delta=cfg["huber_delta"])
                loss.backward()
                opt[kind].step()
                losses.append(loss.detach())
            batch_losses = torch.stack(losses).cpu().numpy().astype(float)
            assert np.isfinite(batch_losses).all()
            total += n
            cumulative += n * batch_losses
            if idx % 500 == 0:
                segment_losses = (cumulative - last_losses) / (total - last_total)
                seg = {"end_batch": idx, "samples": total - last_total, **dict(zip(models, segment_losses))}
                assert old["loss_history"][idx // 500 - 1]["samples"] == seg["samples"]
                segments.append(seg)
                last_total, last_losses = total, cumulative.copy()
                print(idx, total, seg, round(monotonic()-started, 1), flush=True)
    assert (idx, total, digest.hexdigest(), counts.candidates, counts.labels_available) == (
        old["batches"], old["samples"], old["training_sha256"], old["candidates_prepared"], old["labels_available_prepared"])
    saved_hashes = {}
    for kind, model in models.items():
        with torch.no_grad():
            expected = model(cpu_sample[0].to("mps"), cpu_sample[1].to("mps")).cpu()
        state = {"kind": kind, "model": model.cpu().state_dict(), "optimizer": opt[kind].state_dict(),
                 "seed": cfg["seed"], "batches": idx, "training_sha256": digest.hexdigest()}
        fname = OUT / f"{kind}.pt"
        tmp = fname.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        reloaded = load_baseline(tmp, device="mps")
        with torch.no_grad():
            actual = reloaded(cpu_sample[0].to("mps"), cpu_sample[1].to("mps")).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for k in TABLES:
            assert torch.equal(getattr(reloaded, k).cpu(), tables[k])
        assert all(entry["step"].item() == idx for entry in opt[kind].state.values())
        tmp.replace(fname)
        saved_hashes[kind] = sha256(fname.read_bytes()).hexdigest()
    result = {
        "device": "mps", "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "reference_model": str(GATE / "model.pt"), "gate_sha256": plan["gate16_sha256"],
        "gate_trainable_parameters": sum(p.numel() for p in gate.parameters()),
        "trainable_parameters": {k: sum(p.numel() for p in v.parameters()) for k, v in models.items()},
        "initial_named_parameters_sha256": hashes, "weight_file_sha256": saved_hashes, "seed": cfg["seed"],
        "input": "30 positions; channel-major OHLCVA and two per-bar clocks, flattened to 240",
        "initialization": "Xavier uniform per dense layer, zero biases, reseed each model independently",
        "structure": {"mlp24": "240 -> 24 -> LeakyReLU(0.01) -> 1", "linear": "240 -> 1"},
        "training": {"batches": idx, "samples": total, "candidates_prepared": counts.candidates,
                     "labels_available_prepared": counts.labels_available, "training_sha256": digest.hexdigest(),
                     "per_500_batches_huber": segments, "train_mean_huber": dict(zip(models, cumulative/total)),
                     "batch_size": cfg["batch_size"], "stock_days_per_group": cfg["stock_days_per_group"],
                     "source_index_sha256": sources["stock_day_index_sha256"], "metadata_sha256": sources["metadata_sha256"],
                     "gate_batch_order_hash_matched": True, "elapsed_seconds": monotonic()-started},
        "optimization": {"algorithm":"AdamW", "lr":cfg["learning_rate"], "weight_decay":cfg["weight_decay"],
                         "delta": cfg["huber_delta"], "betas":[0.9,0.999], "eps":1e-8, "schedule":"constant",
                         "normalized_training_error":"prediction / minute_scale - y / minute_scale"},
        "label": cfg["return_label"], "frozen_tables_sha256": {k:tensor_hash(v) for k,v in tables.items()},
        "checks": {"strict_reload_128_mps_predictions_bitwise_equal":True, "all_gate_tables_identical":True,
                   "independent_position_and_price_npz_equal":True, "real_batch_128_gate_market_and_clocks_equal":True,
                   "training_only_no_validation_monitor_or_tuning":True}
    }
    (OUT / "training_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("Published both checkpoints and training_report.json", flush=True)


if __name__ == "__main__":
    main()
