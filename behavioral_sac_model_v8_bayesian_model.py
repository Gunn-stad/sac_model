from pathlib import Path
import re

ROOT = Path(__file__).parent
BASE = ROOT / "behavioral_sac_model_v6_hybrid_core.py"
OUT = ROOT / "behavioral_sac_model_v8_bayesian.py"

text = BASE.read_text(encoding="utf-8")

# Name/results
text = re.sub(
    r'RESULTS_DIR\s*=.*',
    'RESULTS_DIR = f"{BASE_DIR}/results/v8_bayesian"',
    text,
)

text = re.sub(
    r'UNIVERSE_NAME\s*=.*',
    'UNIVERSE_NAME = "v8_bayesian_features"',
    text,
)

# Fixed entropy and 30k
text = re.sub(r"seed\s*:\s*int\s*=\s*\d+", "seed: int = 1", text)
text = re.sub(r"total_timesteps\s*:\s*int\s*=\s*[0-9_]+", "total_timesteps: int = 30_000", text)

if "ent_coef:" not in text:
    text = re.sub(
        r"(learning_rate\s*:\s*float\s*=\s*[0-9.eE_-]+)",
        r"\1\n    ent_coef: float = 0.005",
        text,
        count=1,
    )

text = re.sub(r"ent_coef\s*=\s*['\"]auto['\"]", "ent_coef=cfg.ent_coef", text)

if "ent_coef=cfg.ent_coef" not in text:
    text = re.sub(
        r"(learning_rate\s*=\s*cfg\.learning_rate,\s*)",
        r"\1\n        ent_coef=cfg.ent_coef,",
        text,
        count=1,
    )

# Reward: blended alpha + turnover penalty
text = re.sub(
    r"reward\s*=\s*float\((.*?)\)",
    "reward = float(0.5 * port_ret + 0.5 * (port_ret - bench_ret) - 0.002 * turnover)",
    text,
    count=1,
    flags=re.DOTALL,
)

# Add Bayesian feature function after imports
insert_code = r'''

def add_bayesian_features(close, window=48):
    """
    Simple Bayesian-style features using normal approximation.

    Adds:
    - bayes_mu: shrinkage expected return
    - bayes_uncertainty: posterior uncertainty
    - prob_positive: P(return > 0)
    - prob_beat_eq: P(asset return > equal-weight return)
    - bayes_confidence: prob_beat_eq / uncertainty
    """
    import numpy as np
    import pandas as pd
    from math import erf, sqrt

    returns = close.pct_change().fillna(0.0)
    eq_ret = returns.mean(axis=1)

    rolling_mu = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std().replace(0, np.nan)

    # Prior: long-run mean close to 0
    prior_mu = 0.0
    prior_strength = 20.0

    n = window
    bayes_mu = (prior_strength * prior_mu + n * rolling_mu) / (prior_strength + n)
    bayes_uncertainty = rolling_std / np.sqrt(prior_strength + n)

    z_positive = bayes_mu / bayes_uncertainty.replace(0, np.nan)
    prob_positive = 0.5 * (1.0 + z_positive.applymap(lambda x: erf(x / sqrt(2)) if pd.notna(x) else 0.0))

    rel_mu = bayes_mu.sub(eq_ret, axis=0)
    z_beat = rel_mu / bayes_uncertainty.replace(0, np.nan)
    prob_beat_eq = 0.5 * (1.0 + z_beat.applymap(lambda x: erf(x / sqrt(2)) if pd.notna(x) else 0.0))

    bayes_confidence = prob_beat_eq / (bayes_uncertainty.abs() + 1e-8)

    features = {
        "bayes_mu": bayes_mu.fillna(0.0),
        "bayes_uncertainty": bayes_uncertainty.fillna(0.0),
        "prob_positive": prob_positive.fillna(0.5),
        "prob_beat_eq": prob_beat_eq.fillna(0.5),
        "bayes_confidence": bayes_confidence.replace([np.inf, -np.inf], 0).fillna(0.0),
    }

    return features
'''

text = text.replace("import os", "import os\n" + insert_code, 1)

OUT.write_text(text, encoding="utf-8")
print("created", OUT)