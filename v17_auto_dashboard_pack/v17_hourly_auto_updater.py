import os
import time
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
UPDATE_SECONDS = 60 * 60

def market_is_open():
    now = datetime.now(MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

def run_update():
    cmd = ["python", "v14_paper_trading_dashboard.py", "--update"]
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.stderr:
        print("STDERR:", result.stderr, flush=True)
    return result.returncode

def main():
    print("Starting V17 hourly auto updater", flush=True)
    print("Working directory:", os.getcwd(), flush=True)
    while True:
        now = datetime.now(MARKET_TZ)
        print(f"\n[{now}] Checking market...", flush=True)
        if market_is_open():
            print("Market open. Updating paper dashboard.", flush=True)
            code = run_update()
            print("Return code:", code, flush=True)
        else:
            print("Market closed. Skipping.", flush=True)
        print(f"Sleeping {UPDATE_SECONDS} seconds.", flush=True)
        time.sleep(UPDATE_SECONDS)

if __name__ == "__main__":
    main()