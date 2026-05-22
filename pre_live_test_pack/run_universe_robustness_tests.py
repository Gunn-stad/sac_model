"""
Run asset-universe robustness tests before paper/live trading.

Put this file in your Colab repo folder:
    /content/sac_model

It runs the separate universe scripts one by one and saves a compact status log.
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"
OUT_DIR = Path(BASE_DIR) / "results" / "pre_live_tests" / "universe_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "behavioral_sac_universe_top20_plus_etfs.py",
    "behavioral_sac_universe_mixed_assets.py",
    "behavioral_sac_universe_index_etfs.py",
    "behavioral_sac_universe_sector_etfs.py",
    "behavioral_sac_universe_defensive_assets.py",
    "behavioral_sac_universe_war_index_spec_tech.py",
    "behavioral_sac_universe_top30_stocks.py",
]

def run_script(script: str) -> dict:
    started = datetime.utcnow().isoformat()
    print(f"\n=== RUNNING {script} ===")
    if not Path(script).exists():
        return {
            "script": script,
            "status": "missing",
            "started_utc": started,
            "ended_utc": datetime.utcnow().isoformat(),
            "returncode": None,
        }

    proc = subprocess.run(
        ["python", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path = OUT_DIR / f"{Path(script).stem}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    return {
        "script": script,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "started_utc": started,
        "ended_utc": datetime.utcnow().isoformat(),
        "log_path": str(log_path),
    }

def main():
    results = []
    for script in SCRIPTS:
        result = run_script(script)
        results.append(result)

        status_path = OUT_DIR / "universe_test_status.json"
        status_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        if result["status"] == "failed":
            print(f"[warning] {script} failed. Check log: {result.get('log_path')}")

    print("\nDone. Status saved to:")
    print(OUT_DIR / "universe_test_status.json")

if __name__ == "__main__":
    main()
