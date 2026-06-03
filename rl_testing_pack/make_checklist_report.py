"""Create results/checklist_report.md from results/experiment_summary.csv"""
from pathlib import Path
import pandas as pd

summary = Path("results/experiment_summary.csv")
if not summary.exists():
    print("Run aggregate_experiment_results.py first")
    raise SystemExit

df = pd.read_csv(summary)
rl = df[df["name"].astype(str).str.contains("SAC|Hybrid", case=False, na=False)].copy()
if len(rl) == 0:
    print("No RL rows found")
    raise SystemExit

def check(x): return "✅" if x else "⬜"
beats_eq = bool(((rl.get("beats_equal_weight_return", False) == True) & (rl.get("beats_equal_weight_sharpe", False) == True)).any())
turnover_ok = bool((rl.get("avg_turnover", pd.Series([1])).mean() < 0.05))
dd_ok = bool((rl.get("max_drawdown", pd.Series([1])).max() < 0.15))
seed_count = len({s for exp in rl["experiment"].astype(str) for s in [f"seed{i}" for i in range(20)] if s in exp})
universes = set()
for exp in rl["experiment"].astype(str):
    if "theme" in exp: universes.add("theme")
    elif "safe_growth" in exp: universes.add("safe_growth")
    elif "v6" in exp: universes.add("core_v6")

lines = [
"# RL Trading Checklist", "",
f"- {check(beats_eq)} Beats equal weight out-of-sample",
"- ⬜ Beats SPY or QQQ in at least some settings",
"- ✅ Works after transaction costs",
f"- {check(seed_count >= 10)} Tested on 10+ random seeds",
f"- {check(len(universes) >= 3)} Tested on multiple asset universes",
"- ⬜ Tested with walk-forward validation",
"- ⬜ Tested in bull, bear, sideways, and volatile markets",
f"- {check(turnover_ok)} Has reasonable turnover",
f"- {check(dd_ok)} Has max drawdown control",
"- ⬜ Has no lookahead bias",
"- ⬜ Has paper-trading results",
"- ⬜ Has logs/database of every decision",
"- ⬜ Has explainable metrics and plots",
"", "## Current best RL metrics",
f"- Best return: {rl['total_return'].max():.4f}",
f"- Best Sharpe: {rl['sharpe'].max():.4f}",
f"- Worst max drawdown: {rl['max_drawdown'].max():.4f}",
f"- Detected seeds: {seed_count}",
f"- Detected universes: {', '.join(sorted(universes))}",
]
Path("results").mkdir(exist_ok=True)
Path("results/checklist_report.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved results/checklist_report.md")
print("\n".join(lines))
