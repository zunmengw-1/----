"""Recover both frozen flat models and regenerate the exact prior holdout panel."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
OLD = Path("/private/tmp/tcn-signal-date-holdout-20261127")
sys.path.insert(0, str(OLD))
import predict_dates as panel

KEYS = ("code", "t", "label", "label_valid")
COLS = ["code", "t", "p_mlp", "p_linear", "label", "label_valid"]
OUTPUT = HERE / "predictions"


def atomic_audit(data, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def original_day(day, signatures):
    reference = pq.read_table(OLD / "daily" / f"{day}.parquet", columns=list(KEYS))
    meta = reference.schema.metadata
    for key in ("model16_sha256", "metadata_sha256", "qlib_source_signature", "return_label"):
        if meta[key.encode()].decode() != signatures[key]:
            raise ValueError(f"Gate cache signature mismatch: {day} {key}")
    if meta[b"date"].decode() != day or not len(reference):
        raise ValueError(f"Gate cache date mismatch: {day}")
    return reference


def check(table, reference, expected):
    if table.column_names != COLS or table.num_rows != reference.num_rows:
        raise ValueError("Columns or row count differ")
    if table.schema.metadata != {k.encode(): v.encode() for k, v in expected.items()}:
        raise ValueError("Day metadata mismatch")
    for key in KEYS:
        if not table[key].combine_chunks().equals(reference[key].combine_chunks()):
            raise ValueError(f"{key} differs from original panel")
    for name in ("p_mlp", "p_linear"):
        if not np.isfinite(table[name].to_numpy()).all():
            raise ValueError(f"Nonfinite prediction: {name}")


@torch.inference_mode()
def smoke(library, manifest, gate_checkpoint, day):
    model = panel.ReturnTCN.from_checkpoint(gate_checkpoint).eval()
    with panel.Metadata(json.loads(manifest.read_text()), manifest.parent) as meta:
        sessions = panel._date_sessions(library, meta, day, 2)
        scores, minutes, codes, ys = [], [], [], []
        for _, code, columns, s in sessions:
            if not len(columns):
                continue
            m = panel.CLOCK[columns]
            x = panel.prepare_model_inputs(torch.from_numpy(s.x.copy()).transpose(1, 2), m, len(panel.FEATURES))
            p = model(x, torch.as_tensor(m, dtype=torch.long)).numpy()
            scores.extend(p.tolist())
            codes.extend([code] * len(m))
            minutes.extend(m.tolist())
            ys.extend(np.where(s.label_valid, s.y, np.nan).tolist())
    old = pq.read_table(OLD / "daily" / f"{day}.parquet").slice(0, len(scores))
    assert len(scores) > 0
    np.testing.assert_array_equal(codes, old["code"].to_numpy())
    np.testing.assert_array_equal(minutes, (old["t"].cast("int64").to_numpy() // 60000 + 480) % 1440)
    np.testing.assert_allclose(ys, old["label"].to_numpy(), atol=0, rtol=0, equal_nan=True)
    np.testing.assert_allclose(scores, old["p16"].to_numpy(), atol=2e-8, rtol=1e-5)
    return {"date": day, "sample_rows": len(scores), "gate16_cpu_vs_saved_mps_max_abs_diff":
            float(np.max(np.abs(np.array(scores) - old["p16"].to_numpy()))), "keys_and_labels_exact": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    gate_file = panel.CHECKPOINTS[16]
    gate = torch.load(gate_file, map_location="cpu", weights_only=True)
    library, manifest = panel.source_provenance(gate["sources"])
    plan = json.loads((HERE / "evaluation_plan.json").read_text())
    days = plan["evaluation_dates"]
    frozen = OLD / "frozen_plan.json"
    assert panel.sha256(gate_file) == plan["gate16_sha256"]
    assert panel.sha256(frozen) == plan["panel_sha256"]
    assert days == json.loads(frozen.read_text())["evaluation_dates"]
    assert len(days) == len(set(days)) == 67 and max(days) < "2025-07-01"
    reproduction = smoke(library, manifest, gate, days[0])
    atomic_audit(reproduction, HERE / "inference_smoke.json")
    print(json.dumps(reproduction), flush=True)
    if args.smoke_only:
        return
    from window_baseline import TABLES, load_baseline
    paths = [HERE / f"{name}.pt" for name in ("mlp24", "linear")]
    models = [load_baseline(p, device=device).eval() for p in paths]
    gate_model = panel.ReturnTCN.from_checkpoint(gate).eval()
    for path, model in zip(paths, models, strict=True):
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["kind"] == path.stem and state["training_sha256"] == gate["training"]["training_sha256"]
        assert state["batches"] == gate["training"]["batches"] == 5000
        for name in TABLES:
            assert torch.equal(getattr(model, name), getattr(gate_model, name)), (path, name)
    expected = {
        "format": "flat_window_holdout_v1", "model16_sha256": plan["gate16_sha256"],
        "mlp_sha256": panel.sha256(paths[0]), "linear_sha256": panel.sha256(paths[1]),
        "inference_script_sha256": panel.sha256(Path(__file__)),
        "baseline_module_sha256": panel.sha256(HERE / "window_baseline.py"),
        "training_sha256": gate["training"]["training_sha256"], "batches": "5000",
        "holdout_inference_module_sha256": panel.sha256(OLD / "predict_dates.py"),
        "return_label": panel.RETURN_LABEL, "device": str(device), "batch_size": "2048",
        "metadata_sha256": gate["sources"]["metadata_sha256"],
        "qlib_source_signature": gate["sources"]["qlib_source_signature"]
    }
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    with panel.Metadata(json.loads(manifest.read_text()), manifest.parent) as metadata:
        for day in days:
            start = time.monotonic()
            ref = original_day(day, expected)
            daily_expected = {**expected, "date": day,
                              "gate_day_file_sha256": panel.sha256(OLD / "daily" / f"{day}.parquet")}
            dst = OUTPUT / f"{day}.parquet"
            if dst.exists():
                check(pq.read_table(dst), ref, daily_expected)
                status = "reused"
            else:
                table = panel.predict_shared_date(models, library, metadata, day, batch_size=2048, device=device)
                table = table.rename_columns(COLS).replace_schema_metadata(daily_expected)
                check(table, ref, daily_expected)
                pending = dst.with_suffix(".partial")
                pq.write_table(table, pending, compression="zstd", compression_level=1)
                check(pq.read_table(pending), ref, daily_expected)
                pending.replace(dst)
                status = "written"
            rows.append({"date": day, "rows": len(ref), "valid": pc.sum(ref["label_valid"]).as_py(),
                         "status": status, "seconds": round(time.monotonic() - start,2)})
            print(json.dumps(rows[-1]), flush=True)
    atomic_audit({"provenance": expected, "gate_reproduction": reproduction, "rows": sum(d["rows"] for d in rows),
                  "valid": sum(d["valid"] for d in rows), "daily": rows,
                  "exact_columns_vs_original_every_date": list(KEYS), "all_predictions_finite": True,
                  "scope": "Predeclared 67 validation dates through 2025-06-27. No 2025H2 evaluation."},
                 HERE / "inference_audit.json")


if __name__ == "__main__":
    main()
