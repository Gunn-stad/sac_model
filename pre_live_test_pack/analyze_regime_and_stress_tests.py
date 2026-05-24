import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/lokaverkefni_bs")
RESULTS_DIR = BASE_DIR / "results" / "pre_live_tests" / "regime_stress"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPLAY_FILE = (
    BASE_DIR
    / "results"
    / "universe_tests"
    / "top20_plus_etfs"
    / "episode_log_top20_plus_etfs.json"
)

def load_replay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        for key in ["steps", "episode", "log", "records", "replay"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Replay JSON must be a list or contain a list.")

    return pd.DataFrame(data)

def main():
    df = load_replay(REPLAY_FILE)
    print("Loaded replay.")
    print(df.head())

if __name__ == "__main__":
    main()
