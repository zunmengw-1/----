"""用正式取样规则对每个指定交易日一次采样、并行交付两种宽度的预测。"""

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

PROJECT = Path(__file__).resolve().parents[2].resolve()
sys.path.insert(0, str(PROJECT))

from src.data.metadata import Metadata
from src.data.minutes import MinuteLibrary
from src.features.samples import (
    FEATURES,
    RETURN_LABEL,
    TRADING_MINUTES,
    WINDOW,
    split_for,
)
from src.models.evaluation import _date_sessions
from src.models.tcn import ReturnTCN
from src.models.training_data import prepare_model_inputs

CHECKPOINTS = {
    16: Path("/private/tmp/tcn-clock-exclusive-20261127/gate/model.pt"),
    32: Path("/private/tmp/tcn-gate-width-20261127/wide/model.pt"),
}
CUTS_FILE = Path("/private/tmp/tcn-huber-signal-review-20261127/by_signal.json")
ZONE = ZoneInfo("Asia/Shanghai")
CLOCK = np.asarray(TRADING_MINUTES)
SCHEMA = pa.schema([
    ("code", pa.string()),
    ("t", pa.timestamp("ms", tz="Asia/Shanghai")),
    ("p16", pa.float64()),
    ("p32", pa.float64()),
    ("label", pa.float64()),
    ("label_valid", pa.bool_()),
])


def sha256(path):
    """给固定数据来源与模型文件计算内容指纹。"""
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def load_models(device):
    """按检查点构造并严格恢复两种正式门控模型，并核对共享来源。"""
    models, signatures, sources = [], {}, []
    for width, path in CHECKPOINTS.items():
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        if (config.get("return_label"), config.get("time_fusion")) != (RETURN_LABEL, "gate_only"):
            raise ValueError(f"模型的标签或时钟设置不匹配：{path}")
        model = ReturnTCN.from_checkpoint(checkpoint)
        if (model.input_channels, model.hidden_channels, model.window_clock_mode) != (len(FEATURES), width, "gate"):
            raise ValueError(f"模型的量价输入或卷积宽度不匹配：{path}")
        models.append(model.eval().to(device))
        signatures[f"model{width}_sha256"] = sha256(path)
        sources.append(checkpoint["sources"])
    for key in ("qlib", "qlib_source_signature", "metadata", "metadata_sha256", "metadata_files_sha256", "stock_day_cache"):
        if sources[0][key] != sources[1][key]:
            raise ValueError(f"两份训练检查点引用的行情或元数据不同：{key}")
    return models, signatures, sources[0]


def source_provenance(sources):
    """读取冻结的元数据并核验为两份已训练模型使用的版本。"""
    manifest_path = Path(sources["metadata"])
    if sha256(manifest_path) != sources["metadata_sha256"]:
        raise ValueError(f"固定元数据清单已变更：{manifest_path}")
    for path, expected in sources["metadata_files_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"固定元数据来源已变更：{path}")
    cache = Path(sources["stock_day_cache"])
    cache_spec = json.loads((cache / "manifest.json").read_text())
    if cache_spec["source_signature"] != sources["qlib_source_signature"]:
        raise ValueError("行情缓存来源签名已变更")
    library = MinuteLibrary(Path(sources["qlib"]), stock_day_cache=cache)
    return library, manifest_path


@torch.inference_mode()
def predict_shared_date(models, library, metadata, day, *, batch_size, device):
    """用正式 _date_sessions 一次取样，两模型逐批使用同一输入及对齐时点。"""
    window_buffer = np.empty((batch_size, WINDOW, len(FEATURES)), dtype=np.float32)
    codes = np.empty(batch_size, dtype="U8")
    minutes = np.empty(batch_size, dtype=np.intp)
    labels = np.empty(batch_size, dtype=np.float64)
    masks = np.empty(batch_size, dtype=bool)
    batches = []

    def flush(size):
        clock = minutes[:size]
        x = torch.from_numpy(window_buffer[:size]).transpose(1, 2)
        x = prepare_model_inputs(x, clock, len(FEATURES)).to(device)
        t = torch.as_tensor(clock, dtype=torch.long, device=device)
        predictions = torch.stack([model(x, t) for model in models], dim=1).cpu().numpy().astype(np.float64)
        if predictions.shape != (size, 2) or not np.isfinite(predictions).all():
            raise FloatingPointError(f"{day} 出现未对齐或非有限的模型预测")
        wall_times = np.datetime64(day, "m") + clock.astype("timedelta64[m]")
        timestamps = pc.assume_timezone(pa.array(wall_times.astype("datetime64[ms]")), "Asia/Shanghai")
        valid = masks[:size]
        batches.append(pa.RecordBatch.from_arrays([
            pa.array(codes[:size].tolist()), timestamps, pa.array(predictions[:, 0]), pa.array(predictions[:, 1]),
            pa.array(labels[:size].copy(), mask=~valid), pa.array(valid.copy()),
        ], schema=SCHEMA))

    buffered = 0
    for _, code, columns, session in _date_sessions(library, metadata, day, None):
        start = 0
        while start < len(columns):
            copied = min(batch_size - buffered, len(columns) - start)
            selected = slice(start, start + copied)
            pending = slice(buffered, buffered + copied)
            window_buffer[pending] = session.x[selected]
            codes[pending] = code
            minutes[pending] = CLOCK[columns[selected]]
            labels[pending] = session.y[selected]
            masks[pending] = session.label_valid[selected]
            start += copied
            buffered += copied
            if buffered == batch_size:
                flush(buffered)
                buffered = 0
    if buffered:
        flush(buffered)
    return pa.Table.from_batches(batches, schema=SCHEMA)


def check_reproduction(table, day, *, atol, rtol):
    """对照正式历史 MPS 输出，核对完整候选、标签、阈值档位和数值。"""
    start = datetime.combine(date.fromisoformat(day), time(), tzinfo=ZONE)
    stop = start + timedelta(days=1)
    thresholds = json.loads(CUTS_FILE.read_text())
    info = {}
    for width, checkpoint in CHECKPOINTS.items():
        original = pq.read_table(
            checkpoint.with_name("predictions.parquet"),
            filters=[("t", ">=", start), ("t", "<", stop)],
        )
        if original.num_rows != table.num_rows or table.num_rows == 0:
            raise ValueError(f"宽度 {width} 的旧验证日预测与新候选行数不一致")
        for name in ("code", "t", "label", "label_valid"):
            if not original[name].combine_chunks().equals(table[name].combine_chunks()):
                raise ValueError(f"宽度 {width} 的 {name} 列与旧验证日不同")
        old_prediction = original["prediction"].to_numpy()
        new_prediction = table[f"p{width}"].to_numpy()
        cuts_bp = np.asarray(thresholds[str(width)]["cut_bp"])
        if cuts_bp.shape != (2,) or not np.isfinite(cuts_bp).all() or not 0 < cuts_bp[0] < cuts_bp[1]:
            raise ValueError(f"旧宽度 {width} 的两个强度阈值无效")
        cuts = cuts_bp / 10_000
        old_bins = np.searchsorted(cuts, np.abs(old_prediction), side="right")
        new_bins = np.searchsorted(cuts, np.abs(new_prediction), side="right")
        np.testing.assert_array_equal(new_bins, old_bins, err_msg=f"Gate{width} 固定强度档位不同")
        np.testing.assert_allclose(new_prediction, old_prediction, atol=atol, rtol=rtol, err_msg=f"Gate{width} 预测未复现")
        info[str(width)] = {
            "max_abs_prediction_difference": float(np.abs(new_prediction - old_prediction).max()),
            "fixed_thresholds_bp": cuts_bp.tolist(),
            "group_counts": np.bincount(new_bins, minlength=3).tolist(),
        }
    return {
        "check_date": day,
        "rows": table.num_rows,
        "labels_available": int(pc.sum(table["label_valid"]).as_py()),
        "atol_log_return": atol,
        "rtol": rtol,
        "checks_by_width": info,
    }


def is_complete(path, expected):
    """核对先前发布当天产物的完整元信息、行数和时间范围。"""
    if not path.is_file():
        return False
    try:
        file = pq.ParquetFile(path)
        if not file.schema_arrow.equals(SCHEMA, check_metadata=False):
            return False
        metadata = {k.decode(): v.decode() for k, v in (file.schema_arrow.metadata or {}).items()}
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if file.metadata.num_rows != int(metadata["rows"]) or file.metadata.num_rows == 0:
            return False
        table = file.read()
        wall_dates = pc.local_timestamp(table["t"]).to_numpy().astype("datetime64[D]")
        if not np.all(wall_dates == np.datetime64(expected["date"], "D")):
            return False
        valid = table["label_valid"].to_numpy()
        return bool(
            np.isfinite(table["p16"].to_numpy()).all()
            and np.isfinite(table["p32"].to_numpy()).all()
            and np.array_equal(valid, np.isfinite(table["label"].to_numpy()))
            and int(valid.sum()) == int(metadata["labels_available"])
        )
    except (OSError, TypeError, ValueError, KeyError, pa.ArrowException):
        return False


def write_atomically(table, destination, expected):
    """验证写出的文件符合续跑口径后，在同一目录原子发布。"""
    if not table.num_rows:
        raise ValueError(f"{expected['date']} 没有合格预测候选")
    metadata = {
        **expected,
        "rows": str(table.num_rows),
        "labels_available": str(pc.sum(table["label_valid"]).as_py()),
    }
    table = table.replace_schema_metadata(metadata)
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}-", suffix=".tmp", delete=False) as file:
        pending = Path(file.name)
    try:
        pq.write_table(table, pending)
        if not is_complete(pending, expected):
            raise OSError(f"新写出的当日产物未通过完整性核对：{destination}")
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="JSON 中的 evaluation_dates 定义正式预测日期")
    parser.add_argument("--check-date", help="只复现一个旧验证日，在内存中对照两个历史预测文件")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--check-atol", type=float, default=1e-9, help="预测对照的绝对误差容差，单位对数收益率")
    parser.add_argument("--check-rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.plan is None and args.check_date is None:
        parser.error("需要提供 --plan，或使用 --check-date 单独复现旧验证日")
    if args.threads <= 0 or args.batch_size <= 0:
        parser.error("线程数和批次大小必须为正")
    if not np.isfinite([args.check_atol, args.check_rtol]).all() or min(args.check_atol, args.check_rtol) < 0:
        parser.error("对照误差容差应为有限非负数值")
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前 Python 进程无法访问 MPS")
    device = torch.device(args.device)
    models, signatures, sources = load_models(device)
    library, manifest = source_provenance(sources)
    if args.check_date:
        dates = [args.check_date]
    else:
        plan = args.plan.resolve()
        plan_spec = json.loads(plan.read_text())
        dates = plan_spec["evaluation_dates"]
        if not isinstance(dates, list) or not dates or not all(isinstance(day, str) for day in dates) or len(set(dates)) != len(dates):
            raise ValueError("计划的 evaluation_dates 须为非空且无重复的日期字符串列表")
        for width, path in CHECKPOINTS.items():
            if path != Path(plan_spec["model_files"][str(width)]) or signatures[f"model{width}_sha256"] != plan_spec["model_sha256"][str(width)]:
                raise ValueError(f"宽度 {width} 的实际模型文件或内容与冻结计划不同")
        if set(dates).intersection(plan_spec["excluded_known_tcn_evaluation_dates"]):
            raise ValueError("新评估日与已评估日期重叠")
        output = plan.parent / "daily"
        if not output.resolve().is_relative_to(Path("/private/tmp")):
            raise ValueError("预测结果只可写入临时计划目录")
        output.mkdir(exist_ok=True)
    for day in dates:
        split_for(day)
        if date.fromisoformat(day).isoformat() != day or day not in library.dates:
            raise ValueError(f"不是研究期内的 Qlib 交易日：{day}")
    expected = {
        **signatures,
        "format": "tcn_gate_width_shared_predictions_v1",
        "return_label": RETURN_LABEL,
        "qlib_source_signature": sources["qlib_source_signature"],
        "metadata_sha256": sources["metadata_sha256"],
        "device": args.device,
        "batch_size": str(args.batch_size),
    }
    with Metadata(json.loads(manifest.read_text()), manifest.parent) as metadata:
        for day in dates:
            if args.check_date is None:
                target = output / f"{day}.parquet"
                expected_for_day = {**expected, "date": day}
                if is_complete(target, expected_for_day):
                    print(json.dumps({"date": day, "status": "reused"}, ensure_ascii=False), flush=True)
                    continue
            table = predict_shared_date(models, library, metadata, day, batch_size=args.batch_size, device=device)
            if args.check_date:
                print(json.dumps(check_reproduction(table, day, atol=args.check_atol, rtol=args.check_rtol), ensure_ascii=False), flush=True)
            else:
                write_atomically(table, target, expected_for_day)
                print(json.dumps({"date": day, "status": "written", "rows": table.num_rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
