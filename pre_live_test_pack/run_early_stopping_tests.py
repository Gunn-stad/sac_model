import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("/content/drive/MyDrive/lokaverkefni_bs")
REPO_DIR = Path("/content/sac_model")

OUT_DIR = BASE_DIR / "results" / "pre_live_tests" / "early_stopping"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = REPO_DIR / "behavioral_sac_model_v6_hybrid_core.py"

TIMESTEPS_TO_TEST = [20_000, 40_000, 60_000]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def patch_timesteps(source_text: str, timesteps: int) -> str:
    patterns = [
        r"TOTAL_TIMESTEPS\s*=\s*\d+",
        r"TIMESTEPS\s*=\s*\d+",
        r"total_timesteps\s*=\s*\d+",
    ]

    for pat in patterns:
        if re.search(pat, source_text):
            return re.sub(
                pat,
                lambda m: re.sub(r"\d+", str(timesteps), m.group(0), count=1),
                source_text,
                count=1,
            )

    raise ValueError("Could not patch timesteps.")

def main():
    print("Early stopping test helper ready.")

if __name__ == "__main__":
    main()
