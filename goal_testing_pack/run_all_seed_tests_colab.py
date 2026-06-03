"""
run_all_seed_tests_colab.py

Run all 10 seeds for the chosen setup.

Run in Colab:
    !python run_all_seed_tests_colab.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    for seed in range(10):
        test = Path(f"goal_tests/core_blended_turnover_30k_seed{seed}.py")
        if not test.exists():
            print("[missing]", test)
            continue
        print("\n" + "=" * 90)
        print("RUNNING SEED:", seed)
        print("=" * 90)
        subprocess.run([sys.executable, str(test)], check=False)

if __name__ == "__main__":
    main()
