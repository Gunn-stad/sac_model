"""Run priority experiments in Colab: !python run_priority_tests_colab.py"""
import subprocess, sys
from pathlib import Path

TESTS = [
    "experiments/v6_blended_reward_30k_seed1.py",
    "experiments/v6_blended_turnover_30k_seed1.py",
    "experiments/v6_downside_alpha_30k_seed1.py",
    "experiments/safe_growth_fixed_entropy_30k_seed1.py",
    "experiments/theme_blended_reward_30k_seed1.py",
]

for test in TESTS:
    if not Path(test).exists():
        print("[skip missing]", test)
        continue
    print("\n" + "="*80)
    print("RUNNING", test)
    print("="*80)
    rc = subprocess.run([sys.executable, test]).returncode
    print("[returncode]", rc)
