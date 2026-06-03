"""
run_goal_tests_colab.py

Run a priority subset first.

Run in Colab:
    !python run_goal_tests_colab.py
"""

import subprocess
import sys
from pathlib import Path

PRIORITY_TESTS = [
    "goal_tests/core_reward_blended_turnover_30k_seed1.py",
    "goal_tests/core_reward_blended_30k_seed1.py",
    "goal_tests/core_reward_benchmark_relative_30k_seed1.py",
    "goal_tests/core_blended_turnover_30k_seed0.py",
    "goal_tests/core_blended_turnover_30k_seed1.py",
    "goal_tests/core_blended_turnover_30k_seed2.py",
    "goal_tests/theme_blended_turnover_30k_seed1.py",
    "goal_tests/safe_growth_blended_turnover_30k_seed1.py",
    "goal_tests/core_cost_0p001_30k_seed1.py",
    "goal_tests/core_cost_0p002_30k_seed1.py",
    "goal_tests/core_blended_turnover_20k_seed1.py",
    "goal_tests/core_blended_turnover_40k_seed1.py",
]

def main():
    for test in PRIORITY_TESTS:
        path = Path(test)
        if not path.exists():
            print("[missing]", test)
            continue
        print("\n" + "=" * 90)
        print("RUNNING:", test)
        print("=" * 90)
        proc = subprocess.run([sys.executable, test], check=False)
        print("[returncode]", proc.returncode)

if __name__ == "__main__":
    main()
