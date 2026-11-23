"""Default paths relative to the directory where the pipeline is launched."""

from pathlib import Path

ROOT = Path.cwd()
DEFAULT_PROVIDER = ROOT / "data" / "qlib_cn_1min"
DEFAULT_DAILY_SOURCE = ROOT / "data" / "daily_summary.parquet"
DEFAULT_RUN = ROOT / "output" / "tcn_mainboard_joint_01"
