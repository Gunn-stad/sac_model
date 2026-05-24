import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/lokaverkefni_bs")
REPO_DIR = Path("/content/sac_model")
UNIVERSE_DIR = REPO_DIR / "diffrent_universes"

OUT_DIR = BASE_DIR / "results" / "pre_live_tests" / "universe_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    UNIVERSE_DIR / "behavioral_sac_universe_top20_plus_etfs.py",
    UNIVERSE_DIR / "behavioral_sac_universe_mixed_assets.py",
    UNIVERSE_DIR / "behavioral_sac_universe_index_etfs.py",
    UNIVER_DIR / "behavioral_sac_universe_sector_etfs.py",
    UNIVERSE_DIR / "behavioral_sac_universe_defensive_assets.py",
    UNIVERSE_DIR / "behavioral_sac_universe_war_index_spec_tech.py",
    UNIVERSE_DIR / "behavioral_sac_universe_top30_stocks.py",
]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def run_script(script: Path) -> dict:
    started = utc_now()
    print(f"\n=== RUNNING {script.name} ===")

    if not script.exists():
        return {
            "script": str(script),
            "status": "missing",
            "started_utc": started,
            "ended_utc": utc_now(),
            "returncode": None,
        }

    proc = subprocess.run(
        ["python", str(script)],
        cwd=str(REPO_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log_path = OUT_DIR / f"{script.stem}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    return {
        "script": str(script),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "started_utc": started,
        "ended_utc": utc_now(),
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
            print(f"[warning] failed: {script.name}")
            print(f"Check log: {result.get('log_path')}")

    print("\nDone. Status saved to:")
    print(OUT_DIR / "universe_test_status.json")

if __name__ == "__main__":
    main()
