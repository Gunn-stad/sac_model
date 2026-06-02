"""
make_fixed_entropy_experiments.py

Put this file in:
    C:\Gunnar\sac_model\make_fixed_entropy_experiments.py

Then run in VS Code terminal / PowerShell:
    cd C:\Gunnar\sac_model
    python make_fixed_entropy_experiments.py

It creates:
    experiments/v6_fixed_entropy_40k_seed0.py
    experiments/v6_fixed_entropy_40k_seed1.py
    experiments/v6_fixed_entropy_40k_seed2.py
    experiments/v6_fixed_entropy_60k_seed1.py
    experiments/theme_fixed_entropy_40k_seed1.py

Then push to GitHub and run the experiment files in Colab.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "experiments"
EXPERIMENT_DIR.mkdir(exist_ok=True)

CORE_BASE = ROOT / "behavioral_sac_model_v6_hybrid_core.py"
THEME_BASE = ROOT / "behavioral_sac_model_theme_war_index_spec_tech.py"


def patch_base_code(text: str, experiment_name: str, seed: int, timesteps: int) -> str:
    text = re.sub(
        r'RESULTS_DIR\s*=.*',
        f'RESULTS_DIR = f"{{BASE_DIR}}/results/experiments/{experiment_name}"',
        text,
    )

    text = re.sub(
        r"seed\s*:\s*int\s*=\s*\d+",
        f"seed: int = {seed}",
        text,
    )

    text = re.sub(
        r"total_timesteps\s*:\s*int\s*=\s*[0-9_]+",
        f"total_timesteps: int = {timesteps:_}",
        text,
    )

    if "ent_coef:" not in text:
        text = re.sub(
            r"(learning_rate\s*:\s*float\s*=\s*[0-9.eE_-]+)",
            r"\1\n    ent_coef: float = 0.005",
            text,
            count=1,
        )
    else:
        text = re.sub(
            r"ent_coef\s*:\s*[^=]+=\s*.*",
            "ent_coef: float = 0.005",
            text,
        )

    text = re.sub(
        r"ent_coef\s*=\s*['\"]auto['\"]",
        "ent_coef=cfg.ent_coef",
        text,
    )

    if "ent_coef=cfg.ent_coef" not in text:
        text = re.sub(
            r"(learning_rate\s*=\s*cfg\.learning_rate,\s*)",
            r"\1\n        ent_coef=cfg.ent_coef,",
            text,
            count=1,
        )

    header = f
Generated experiment:
    {experiment_name}

Settings:
    seed = {seed}
    total_timesteps = {timesteps}
    ent_coef = 0.005

Run in Colab:
    !python experiments/{experiment_name}.py
