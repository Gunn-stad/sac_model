"""
Early-stopping / training-length test.

Purpose:
    Test whether the SAC model performs better at fewer training steps.
    Finance RL often overfits if trained too long.

This script creates temporary copies of a base model file and changes TIMESTEPS
or TOTAL_TIMESTEPS if that variable exists.

Recommended:
    base_model = behavioral_sac_model_v6_hybrid_core.py
"""

import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"
OUT_DIR = Path(BASE_DIR) / "results" / "pre_live_tests" / "early_stopping"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "behavioral_sac_model_v6_hybrid_core.py"

TIMESTEPS_TO_TEST = [20_000, 40_000, 60_000, 80_000, 100_000, 120_000]

def patch_timesteps(source_text: str, timesteps: int) -> str:
    patterns = [
        r"TOTAL_TIMESTEPS\s*=\s*\d+",
        r"TIMESTEPS\s*=\s*\d+",
        r"total_timesteps\s*=\s*\d+",
    ]

    patched = source_text
    replaced = False

    for pat in patterns:
        if re.search(pat, patched):
            patched = re.sub(pat, lambda m: m.group(0).split("=")[0] + f"= {timesteps}", patched, count=1)
            replaced = True
            break

    if not replaced:
        # If the model uses model.learn(total_timesteps=...), patch that.
        patched2 = re.sub(
            r"model\.learn\(\s*total_timesteps\s*=\s*\d+",
            f"model.learn(total_timesteps={timesteps}",
            patched,
            count=1,
        )
        if patched2 != patched:
            patched = patched2
            replaced = True

    if not replaced:
        raise ValueError("Could not find a timesteps variable or model.learn(total_timesteps=...) to patch.")

    return patched

def run_variant(timesteps: int) -> dict:
    source = Path(BASE_MODEL)
    if not source.exists():
        return {"timesteps": timesteps, "status": "missing_base_model"}

    text = source.read_text(encoding="utf-8")
    patched = patch_timesteps(text, timesteps)

    variant = Path(f"_tmp_early_stop_{timesteps}.py")
    variant.write_text(patched, encoding="utf-8")

    print(f"\n=== Running {variant} ===")
    proc = subprocess.run(
        ["python", str(variant)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path = OUT_DIR / f"early_stop_{timesteps}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    try:
        variant.unlink()
    except Exception:
        pass

    return {
        "timesteps": timesteps,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "ended_utc": datetime.utcnow().isoformat(),
    }

def main():
    results = []
    for t in TIMESTEPS_TO_TEST:
        r = run_variant(t)
        results.append(r)
        (OUT_DIR / "early_stopping_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nDone. Logs saved to:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()
