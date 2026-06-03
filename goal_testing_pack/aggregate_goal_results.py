"""
aggregate_goal_results.py

Aggregates summary_results*.json from results/goal_tests and creates:
    results/goal_test_summary.csv

Run:
    !python aggregate_goal_results.py
"""

from pathlib import Path
import json
import pandas as pd

RESULTS_ROOTS = [Path("results/goal_tests"), Path("results/experiments"), Path("results")]
OUT = Path("results/goal_test_summary.csv")

def load_rows(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = []
    def add_item(item):
        if isinstance(item, dict) and "name" in item:
            row = dict(item)
            row["source_file"] = str(path)
            row["experiment"] = path.parent.name
            rows.append(row)

    if isinstance(data, list):
        for item in data:
            add_item(item)
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    add_item(item)
    return rows

def main():
    rows = []
    seen = set()
    for root in RESULTS_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("summary_results*.json"):
            if str(path) in seen:
                continue
            seen.add(str(path))
            rows.extend(load_rows(path))

    if not rows:
        print("No summary files found.")
        return

    df = pd.DataFrame(rows)
    enhanced = []

    for exp, group in df.groupby("experiment"):
        eq = group[group["name"].astype(str).str.contains("Baseline_EQ", case=False, na=False)]
        if len(eq) > 0:
            eq_ret = float(eq.iloc[0].get("total_return", 0))
            eq_sharpe = float(eq.iloc[0].get("sharpe", 0))
        else:
            eq_ret = None
            eq_sharpe = None

        for _, row in group.iterrows():
            r = row.to_dict()
            if eq_ret is not None:
                r["alpha_vs_equal_weight"] = float(r.get("total_return", 0)) - eq_ret
                r["sharpe_diff_vs_equal_weight"] = float(r.get("sharpe", 0)) - eq_sharpe
                r["beats_equal_weight_return"] = r["alpha_vs_equal_weight"] > 0
                r["beats_equal_weight_sharpe"] = r["sharpe_diff_vs_equal_weight"] > 0
            enhanced.append(r)

    out_df = pd.DataFrame(enhanced)
    OUT.parent.mkdir(exist_ok=True)
    out_df.to_csv(OUT, index=False)

    print("Saved:", OUT)
    rl = out_df[out_df["name"].astype(str).str.contains("SAC|Hybrid", case=False, na=False)]
    cols = ["experiment", "name", "total_return", "sharpe", "max_drawdown", "avg_turnover",
            "avg_cash", "alpha_vs_equal_weight", "sharpe_diff_vs_equal_weight"]
    cols = [c for c in cols if c in rl.columns]
    print(rl.sort_values("total_return", ascending=False)[cols].head(30))

if __name__ == "__main__":
    main()
