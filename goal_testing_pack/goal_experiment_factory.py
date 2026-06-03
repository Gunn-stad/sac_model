"""
goal_experiment_factory.py

Save in:
    C:\Gunnar\sac_model\goal_experiment_factory.py

Run locally:
    cd C:\Gunnar\sac_model
    python goal_experiment_factory.py

Creates permanent experiment files in:
    C:\Gunnar\sac_model\goal_tests\
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "goal_tests"
OUT.mkdir(exist_ok=True)

CORE_BASE = "behavioral_sac_model_v6_hybrid_core.py"
THEME_BASE = "behavioral_sac_model_theme_war_index_spec_tech.py"
SAFE_BASE = "behavioral_sac_universe_safe_growth_hybrid.py"


def read_base(base_name):
    path = ROOT / base_name
    if not path.exists():
        raise FileNotFoundError(f"Missing base file: {path}")
    return path.read_text(encoding="utf-8")


def patch_common(text, exp_name, seed, timesteps, ent_coef=0.005):
    text = re.sub(
        r"RESULTS_DIR\s*=.*",
        f'RESULTS_DIR = f"{{BASE_DIR}}/results/goal_tests/{exp_name}"',
        text,
        count=1,
    )

    text = re.sub(
        r"seed\s*:\s*int\s*=\s*\d+",
        f"seed: int = {seed}",
        text,
        count=1,
    )

    text = re.sub(
        r"total_timesteps\s*:\s*int\s*=\s*[0-9_]+",
        f"total_timesteps: int = {timesteps:_}",
        text,
        count=1,
    )

    if "ent_coef:" not in text:
        text = re.sub(
            r"(learning_rate\s*:\s*float\s*=\s*[0-9.eE_-]+)",
            rf"\1\n    ent_coef: float = {ent_coef}",
            text,
            count=1,
        )
    else:
        text = re.sub(
            r"ent_coef\s*:\s*[^=]+=\s*.*",
            f"ent_coef: float = {ent_coef}",
            text,
            count=1,
        )

    text = re.sub(
        r"ent_coef\s*=\s*['\"]auto['\"]",
        "ent_coef=cfg.ent_coef",
        text,
        count=1,
    )

    if "ent_coef=cfg.ent_coef" not in text:
        text = re.sub(
            r"(learning_rate\s*=\s*cfg\.learning_rate,\s*)",
            r"\1\n        ent_coef=cfg.ent_coef,",
            text,
            count=1,
        )

    return text


def patch_reward(text, mode):
    if mode == "default":
        return text

    if mode == "benchmark_relative":
        new = "reward = float(port_ret - bench_ret)"
    elif mode == "blended":
        new = "reward = float(0.5 * port_ret + 0.5 * (port_ret - bench_ret))"
    elif mode == "blended_turnover":
        new = "reward = float(0.5 * port_ret + 0.5 * (port_ret - bench_ret) - 0.002 * turnover)"
    elif mode == "downside_alpha":
        new = "reward = float((port_ret - bench_ret) - 0.5 * max(0.0, bench_ret - port_ret))"
    else:
        raise ValueError(mode)

    patched = re.sub(
        r"reward\s*=\s*float\((.*?)\)",
        new,
        text,
        count=1,
        flags=re.DOTALL,
    )

    if patched == text:
        patched = text.replace("reward = port_ret", new)

    return patched


def patch_transaction_cost(text, cost):
    pairs = [
        (r"transaction_cost\s*:\s*float\s*=\s*[0-9.eE_-]+", f"transaction_cost: float = {cost}"),
        (r"cost_rate\s*:\s*float\s*=\s*[0-9.eE_-]+", f"cost_rate: float = {cost}"),
        (r"kappa\s*:\s*float\s*=\s*[0-9.eE_-]+", f"kappa: float = {cost}"),
        (r"TRANSACTION_COST\s*=\s*[0-9.eE_-]+", f"TRANSACTION_COST = {cost}"),
    ]
    for pat, rep in pairs:
        text = re.sub(pat, rep, text, count=1)
    return text


def patch_turnover_limit(text):
    pairs = [
        (r"rebalance_every\s*:\s*int\s*=\s*\d+", "rebalance_every: int = 6"),
        (r"action_smooth\s*:\s*float\s*=\s*[0-9.eE_-]+", "action_smooth: float = 0.20"),
        (r"REBALANCE_EVERY\s*=\s*\d+", "REBALANCE_EVERY = 6"),
        (r"SMOOTHING\s*=\s*[0-9.eE_-]+", "SMOOTHING = 0.20"),
    ]
    for pat, rep in pairs:
        text = re.sub(pat, rep, text, count=1)
    return text


def make_file(base_name, exp_name, seed=1, timesteps=30000, reward_mode="default",
              transaction_cost=None, turnover_limit=False):
    text = read_base(base_name)
    text = patch_common(text, exp_name, seed, timesteps)
    text = patch_reward(text, reward_mode)

    if transaction_cost is not None:
        text = patch_transaction_cost(text, transaction_cost)

    if turnover_limit:
        text = patch_turnover_limit(text)

    header = f
Goal-test experiment:
    {exp_name}

Base:
    {base_name}

Settings:
    seed = {seed}
    timesteps = {timesteps}
    ent_coef = 0.005
    reward_mode = {reward_mode}
    transaction_cost = {transaction_cost}
    turnover_limit = {turnover_limit}

Run:
    !python goal_tests/{exp_name}.py
