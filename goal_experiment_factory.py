from pathlib import Path
import shutil

ROOT = Path(__file__).parent
OUT = ROOT / "goal_tests"

OUT.mkdir(exist_ok=True)

BASE = ROOT / "behavioral_sac_model_v6_hybrid_core.py"

for seed in range(10):
    dst = OUT / f"core_blended_turnover_30k_seed{seed}.py"

    shutil.copy(BASE, dst)

    print("created:", dst)

print("Done.")