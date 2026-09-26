"""Per-bucket performance breakdown from an already-completed run's saved
CSVs -- no new API calls, no new cost. Reuses harness.py's own
summarize_by() so the numbers are computed exactly the same way a live
--report-by bucket run would compute them.

Run from eval/:  python analysis/bucket_breakdown.py [timestamp]
If no timestamp given, uses the most recent run found in results/.
"""
import csv
import glob
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
RESULTS = os.path.join(HERE, "..", "results")

import harness  # noqa: E402


def latest_timestamp():
    files = glob.glob(os.path.join(RESULTS, "*.csv"))
    files = [f for f in files if "summary" not in f and "by_bucket" not in f]
    if not files:
        sys.exit(f"No run CSVs found in {RESULTS}")
    ts_list = sorted({os.path.basename(f).split("_")[0] for f in files})
    return ts_list[-1]


BOOL_KEYS = set(harness.HARD) | {"all_hard_pass"}


PASSTHROUGH_KEYS = {"id", "system", "bucket", "contact", "model", "backend", "error"}


def _convert(key, val):
    """CSV round-trips everything as strings; aggregate() requires real
    bools/floats or it silently drops the column. error must stay an
    empty string (not "N/A") since summarize() checks `not r["error"]`
    to decide whether a row is ok -- turning "" into "N/A" would make
    every valid row look like an error."""
    if key in PASSTHROUGH_KEYS:
        return val
    if val == "N/A":
        return "N/A"
    if key in BOOL_KEYS:
        return val == "True"
    try:
        return float(val)
    except ValueError:
        return val


def load_system_rows(ts, system):
    path = os.path.join(RESULTS, f"{ts}_{system}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: _convert(k, v) for k, v in row.items()} for row in rows]


if __name__ == "__main__":
    ts = sys.argv[1] if len(sys.argv) > 1 else latest_timestamp()
    systems = [s for s in ["simple", "oss", "layer0"]
               if os.path.exists(os.path.join(RESULTS, f"{ts}_{s}.csv"))]
    if not systems:
        sys.exit(f"No result CSVs found for timestamp {ts}")

    breakdowns = []
    for system in systems:
        rows = load_system_rows(ts, system)
        breakdowns.extend(harness.summarize_by("bucket", system, rows, ts))

    harness.print_table(breakdowns)
    out_path = os.path.join(RESULTS, f"{ts}_by_bucket.csv")
    harness.write_rows(Path(out_path), breakdowns)
    print(f"\nsaved: {out_path}")
