"""
v6_exp_safer_exposure.py

Controlled V6 experiment.

How to use:
    Put this file in /content/sac_model/v6_experiments or C:\Gunnar\sac_model\v6_experiments
    Run in Colab:
        %cd /content/sac_model
        !python v6_experiments/v6_exp_safer_exposure.py

This script creates a temporary patched copy of a successful existing v6/universe script
and runs that copy. It avoids rewriting the whole model from scratch.
"""

import os
import re
from pathlib import Path

BASE_CANDIDATES = [
    "/content/sac_model/behavioral_sac_model_v6_hybrid_core.py",
    "/content/sac_model/behavioral_sac_model_v6_hybrid_stocastic_and_more.py",
    "/content/sac_model/diffrent_universes/behavioral_sac_universe_top20_plus_etfs.py",
]

def find_base_script():
    for p in BASE_CANDIDATES:
        path = Path(p)
        if path.exists():
            print("[base] using", path)
            return path
    raise FileNotFoundError("Could not find a successful v6/base universe script.")

def patch_results_name(text, name):
    # Patch name when variable exists.
    text = re.sub(r'UNIVERSE_NAME\s*=\s*["\'][^"\']+["\']', f'UNIVERSE_NAME = "{name}"', text, count=1)

    # Patch common RESULT_DIR forms if they exist.
    text = re.sub(
        r'RESULTS_DIR\s*=\s*f?["\'][^"\']*results/[^"\']+["\']',
        f'RESULTS_DIR = f"{{BASE_DIR}}/results/v6_experiments/{name}"',
        text,
        count=1,
    )
    return text

def run_patched(text, tmp_name):
    tmp = Path("/content/sac_model") / tmp_name
    tmp.write_text(text, encoding="utf-8")
    print("[run]", tmp)
    raise SystemExit(os.system(f"python {tmp}"))


def main():
    base = find_base_script()
    text = base.read_text(encoding="utf-8")
    text = patch_results_name(text, "v6_safer_exposure")

    text = re.sub(r"MIN_EXPOSURE\s*=\s*[0-9.eE_-]+", "MIN_EXPOSURE = 0.65", text)
    text = re.sub(r"LAMBDA_RISK\s*=\s*[0-9.eE_-]+", "LAMBDA_RISK = 0.05", text)
    text = re.sub(r"LAMBDA_CONC\s*=\s*[0-9.eE_-]+", "LAMBDA_CONC = 0.03", text)

    run_patched(text, "_tmp_v6_safer_exposure.py")

if __name__ == "__main__":
    main()
