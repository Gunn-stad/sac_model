"""Aggregate all summary_results*.json into results/experiment_summary.csv"""
from pathlib import Path
import json
import pandas as pd

rows = []
for p in Path("results").rglob("summary_results*.json"):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "name" in item:
                r = dict(item); r["experiment"] = p.parent.name; r["source_file"] = str(p); rows.append(r)
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "name" in item:
                        r = dict(item); r["experiment"] = p.parent.name; r["source_file"] = str(p); rows.append(r)

if not rows:
    print("No results found")
    raise SystemExit

df = pd.DataFrame(rows)
out_rows = []
for exp, g in df.groupby("experiment"):
    eq = g[g["name"].astype(str).str.contains("Baseline_EQ", case=False, na=False)]
    eq_return = float(eq.iloc[0]["total_return"]) if len(eq) else None
    eq_sharpe = float(eq.iloc[0]["sharpe"]) if len(eq) else None
    for _, row in g.iterrows():
        d = row.to_dict()
        if eq_return is not None:
            d["alpha_vs_equal_weight"] = float(d.get("total_return", 0)) - eq_return
            d["sharpe_diff_vs_equal_weight"] = float(d.get("sharpe", 0)) - eq_sharpe
            d["beats_equal_weight_return"] = d["alpha_vs_equal_weight"] > 0
            d["beats_equal_weight_sharpe"] = d["sharpe_diff_vs_equal_weight"] > 0
        out_rows.append(d)

out = pd.DataFrame(out_rows)
Path("results").mkdir(exist_ok=True)
out.to_csv("results/experiment_summary.csv", index=False)
print("Saved results/experiment_summary.csv")
print(out.sort_values("total_return", ascending=False).head(20))
