"""
behavioral_sac_universe_safe_growth_hybrid.py

Safe-growth hybrid universe experiment.

Goal:
    Test whether combining boring/safe assets with growth assets improves
    the SAC portfolio model before paper/live trading.

Universe:
    Broad indexes + growth/tech + defensive sectors + bonds/cash-like ETFs + gold.

Run in Colab:
    !python /content/sac_model/diffrent_universes/behavioral_sac_universe_safe_growth_hybrid.py
"""

import os
import shutil
from pathlib import Path

# This wrapper is designed to reuse your existing v6 hybrid core script if present.
# It creates a temporary copy and patches the universe/tickers/output name.

BASE_SCRIPT_CANDIDATES = [
    "/content/sac_model/behavioral_sac_model_v6_hybrid_core.py",
    "/content/sac_model/diffrent_universes/behavioral_sac_universe_top20_plus_etfs.py",
]

SAFE_GROWTH_TICKERS = [
    # Broad index / core
    "SPY", "QQQ", "VTI", "DIA", "IWM",

    # Growth / tech
    "XLK", "SMH", "SOXX", "BOTZ", "ARKK",

    # Defensive sectors
    "XLP", "XLU", "XLV",

    # Dividend / low volatility
    "SCHD", "VIG", "USMV", "SPLV",

    # Bonds / cash-like
    "SHY", "BIL", "SGOV", "IEF", "TLT", "TIP",

    # Crisis hedge
    "GLD",
]


def find_base_script() -> Path:
    for p in BASE_SCRIPT_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a base script. Expected one of:\n"
        + "\n".join(BASE_SCRIPT_CANDIDATES)
    )


def patch_script(text: str) -> str:
    ticker_list_code = "[\n" + ",\n".join(f'    "{t}"' for t in SAFE_GROWTH_TICKERS) + "\n]"

    # Patch common variable names used in your scripts.
    replacements = {
        'UNIVERSE_NAME = "v6_hybrid_core_stochastic_etfs"': 'UNIVERSE_NAME = "safe_growth_hybrid"',
        'UNIVERSE_NAME = "top20_plus_etfs"': 'UNIVERSE_NAME = "safe_growth_hybrid"',
        'UNIVERSE_NAME = "top20_hybrid_assets_stochastic"': 'UNIVERSE_NAME = "safe_growth_hybrid"',
        'RESULTS_DIR = f"{BASE_DIR}/results/top20_trade_replay"': 'RESULTS_DIR = f"{BASE_DIR}/results/universe_tests/safe_growth_hybrid"',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Patch ticker lists. This handles simple TICKERS = [...] blocks.
    import re
    text = re.sub(
        r"TICKERS\s*=\s*\[[\s\S]*?\]\s*\n\s*BOTTOM_K",
        "TICKERS = " + ticker_list_code + "\n\nBOTTOM_K",
        text,
        count=1,
    )

    # Use bottom_k 5 for this 24-asset universe.
    text = re.sub(r"BOTTOM_K\s*=\s*\d+", "BOTTOM_K = 5", text, count=1)

    return text


def main():
    base = find_base_script()
    print("[safe-growth] using base script:", base)

    text = base.read_text(encoding="utf-8")
    patched = patch_script(text)

    tmp = Path("/content/sac_model/_tmp_safe_growth_hybrid_run.py")
    tmp.write_text(patched, encoding="utf-8")

    print("[safe-growth] running patched experiment")
    os.system(f"python {tmp}")


if __name__ == "__main__":
    main()
