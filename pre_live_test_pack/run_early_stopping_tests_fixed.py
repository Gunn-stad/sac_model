"""
Early stopping / training length experiments.

Purpose:
    Test whether the SAC model performs better when trained for fewer timesteps.

Run:
    %cd /content/sac_model
    !python pre_live_test_pack/run_early_stopping_tests.py
"""

import json
import re
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"

OUT_DIR = (
    Path(BASE_DIR)
    / "results"
    / "pre_live_tests"
    / "early_stopping"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CHANGE THIS TO THE MODEL YOU WANT TO TEST
BASE_MODEL = "diffrent_universes/behavioral_sac_universe_top20_plus_etfs.py"

TIMESTEPS_TO_TEST = [
    20_000,
    40_000,
    60_000,
    80_000,
    100_000,
    120_000,
]


def patch_timesteps(source_text: str, timesteps: int) -> str:
    """
    Try many common patterns used in your SAC scripts.
    """

    replacements = [
        (
            r"TOTAL_TIMESTEPS\s*=\s*[0-9_]+",
            f"TOTAL_TIMESTEPS = {timesteps}",
        ),
        (
            r"TIMESTEPS\s*=\s*[0-9_]+",
            f"TIMESTEPS = {timesteps}",
        ),
        (
            r"N_TIMESTEPS\s*=\s*[0-9_]+",
            f"N_TIMESTEPS = {timesteps}",
        ),
        (
            r"total_timesteps\s*=\s*[0-9_]+",
            f"total_timesteps={timesteps}",
        ),
        (
            r"model\.learn\(\s*total_timesteps\s*=\s*[0-9_]+",
            f"model.learn(total_timesteps={timesteps}",
        ),
        (
            r"model\.learn\(\s*[0-9_]+",
            f"model.learn({timesteps}",
        ),
    ]

    patched = source_text

    for pattern, replacement in replacements:
        patched2 = re.sub(
            pattern,
            replacement,
            patched,
            count=1,
        )

        if patched2 != patched:
            return patched2

    raise ValueError(
        "Could not patch timesteps. "
        "Search your model file for TOTAL_TIMESTEPS or model.learn(...)."
    )


def run_variant(timesteps: int) -> dict:
    source = Path(BASE_MODEL)

    if not source.exists():
        return {
            "timesteps": timesteps,
            "status": "missing_base_model",
            "base_model": BASE_MODEL,
        }

    text = source.read_text(encoding="utf-8")

    patched = patch_timesteps(text, timesteps)

    tmp_file = Path(
        f"_tmp_early_stop_{timesteps}.py"
    )

    tmp_file.write_text(
        patched,
        encoding="utf-8",
    )

    print(f"\n=== RUNNING {timesteps} timesteps ===")

    proc = subprocess.run(
        ["python", str(tmp_file)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path = (
        OUT_DIR
        / f"early_stop_{timesteps}.log"
    )

    log_path.write_text(
        proc.stdout,
        encoding="utf-8",
    )

    try:
        tmp_file.unlink()
    except Exception:
        pass

    return {
        "timesteps": timesteps,
        "status": (
            "ok"
            if proc.returncode == 0
            else "failed"
        ),
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "ended_utc": datetime.utcnow().isoformat(),
    }


def main():
    results = []

    for t in TIMESTEPS_TO_TEST:
        try:
            r = run_variant(t)

        except Exception as e:
            r = {
                "timesteps": t,
                "status": "exception",
                "error": str(e),
            }

        results.append(r)

        status_path = (
            OUT_DIR
            / "early_stopping_status.json"
        )

        status_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    print("\nDone.")
    print("Results saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
