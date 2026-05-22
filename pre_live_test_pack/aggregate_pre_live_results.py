"""
Aggregate pre-live test results from summary JSON files.

This scans result directories for summary_results*.json and metrics.json files,
extracts strategy results, and produces a single CSV.

Run after your universe/feature/seed tests.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/lokaverkefni_bs")
SEARCH_ROOTS = [
    BASE_DIR / "results" / "top20_trade_replay",
    BASE_DIR / "results" / "universe_tests",
    BASE_DIR / "results" / "pre_live_tests",
]
OUT_DIR = BASE_DIR / "results" / "pre_live_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def flatten(prefix: str, obj: Any, out: dict):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = obj

def extract_rows_from_json(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = []

    # Common case: list of result dicts.
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                row = {"source_file": str(path)}
                row.update(item)
                rows.append(row)
        return rows

    # Common case: dict with baseline/test/val results.
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and any(metric in value for metric in ["total_return", "sharpe", "max_drawdown"]):
                row = {"source_file": str(path), "block": key}
                row.update(value)
                rows.append(row)

        # If the dict itself looks like one metric row.
        if any(metric in data for metric in ["total_return", "sharpe", "max_drawdown"]):
            row = {"source_file": str(path)}
            row.update(data)
            rows.append(row)

    return rows

def main():
    all_rows = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*.json"):
            if any(x in path.name.lower() for x in ["summary", "metrics"]):
                all_rows.extend(extract_rows_from_json(path))

    if not all_rows:
        print("No summary/metrics JSON files found.")
        return

    df = pd.DataFrame(all_rows)

    # Try to order useful columns first.
    preferred = [
        "name", "block", "total_return", "sharpe", "max_drawdown",
        "avg_turnover", "avg_cash", "final_equity",
        "seed", "min_exposure", "lambda_bench", "learning_rate",
        "source_file",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_path = OUT_DIR / "all_pre_live_results.csv"
    df.to_csv(out_path, index=False)

    print("Saved aggregate results:")
    print(out_path)
    print(df.sort_values(by=[c for c in ["sharpe", "total_return"] if c in df.columns], ascending=False).head(20))

if __name__ == "__main__":
    main()
