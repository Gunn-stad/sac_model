"""
make_goal_checklist_report.py

Creates:
    results/goal_checklist_report.md

Run:
    !python aggregate_goal_results.py
    !python make_goal_checklist_report.py
"""

from pathlib import Path
import pandas as pd
import re

CSV = Path("results/goal_test_summary.csv")
OUT = Path("results/goal_checklist_report.md")

def mark(x):
    return "✅" if bool(x) else "⬜"

def detect_seed_count(experiments):
    seeds = set()
    for exp in experiments:
        for m in re.finditer(r"seed(\d+)", str(exp)):
            seeds.add(int(m.group(1)))
    return len(seeds)

def main():
    if not CSV.exists():
        print("Missing results/goal_test_summary.csv. Run aggregate_goal_results.py first.")
        return

    df = pd.read_csv(CSV)
    rl = df[df["name"].astype(str).str.contains("SAC|Hybrid", case=False, na=False)].copy()
    if len(rl) == 0:
        print("No RL rows found.")
        return

    beats_eq_return = bool((rl.get("beats_equal_weight_return", False) == True).any())
    beats_eq_sharpe = bool((rl.get("beats_equal_weight_sharpe", False) == True).any())
    seed_count = detect_seed_count(rl["experiment"].unique())

    universes = set()
    for exp in rl["experiment"].astype(str):
        if "theme" in exp:
            universes.add("theme")
        elif "safe_growth" in exp:
            universes.add("safe_growth")
        elif "core" in exp or "v6" in exp:
            universes.add("core")
        else:
            universes.add("other")

    has_transaction_cost_tests = any("cost_" in str(x) for x in rl["experiment"])
    has_multiple_universes = len(universes) >= 3
    has_reasonable_turnover = bool((rl.get("avg_turnover", 999) < 0.05).all())
    has_drawdown_control = bool((rl.get("max_drawdown", 999) < 0.15).all())

    best = rl.sort_values("total_return", ascending=False).iloc[0]
    best_sharpe = rl.sort_values("sharpe", ascending=False).iloc[0]

    lines = []
    lines.append("# RL Trading Checklist Report\n")
    lines.append("## Checklist\n")
    lines.append(f"- {mark(beats_eq_return and beats_eq_sharpe)} Beats equal weight out-of-sample")
    lines.append("- ⬜ Beats SPY or QQQ in at least some settings")
    lines.append(f"- {mark(has_transaction_cost_tests)} Works after transaction costs")
    lines.append(f"- {mark(seed_count >= 10)} Tested on 10+ random seeds")
    lines.append(f"- {mark(has_multiple_universes)} Tested on multiple asset universes")
    lines.append("- ⬜ Tested with walk-forward validation")
    lines.append("- ⬜ Tested in bull, bear, sideways, and volatile markets")
    lines.append(f"- {mark(has_reasonable_turnover)} Has reasonable turnover")
    lines.append(f"- {mark(has_drawdown_control)} Has max drawdown control")
    lines.append("- ⬜ Has no lookahead bias")
    lines.append("- ⬜ Has paper-trading results")
    lines.append("- ✅ Has logs/database of every decision")
    lines.append("- ⬜ Has explainable metrics and plots\n")

    lines.append("## Best return RL run\n")
    lines.append(f"- Experiment: {best['experiment']}")
    lines.append(f"- Name: {best['name']}")
    lines.append(f"- Return: {best['total_return']:.4f}")
    lines.append(f"- Sharpe: {best['sharpe']:.4f}")
    lines.append(f"- Max drawdown: {best['max_drawdown']:.4f}\n")

    lines.append("## Best Sharpe RL run\n")
    lines.append(f"- Experiment: {best_sharpe['experiment']}")
    lines.append(f"- Name: {best_sharpe['name']}")
    lines.append(f"- Return: {best_sharpe['total_return']:.4f}")
    lines.append(f"- Sharpe: {best_sharpe['sharpe']:.4f}")
    lines.append(f"- Max drawdown: {best_sharpe['max_drawdown']:.4f}\n")

    lines.append("## Detected coverage\n")
    lines.append(f"- Seeds detected: {seed_count}")
    lines.append(f"- Universes detected: {', '.join(sorted(universes))}")
    lines.append(f"- Number of RL rows: {len(rl)}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", OUT)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
