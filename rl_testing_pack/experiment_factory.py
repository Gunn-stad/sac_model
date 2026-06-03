"""
experiment_factory.py

Save in: C:\\Gunnar\\sac_model\\experiment_factory.py
Run locally:
    cd C:\\Gunnar\\sac_model
    python experiment_factory.py

Creates experiment files in experiments/ that test:
- fixed entropy
- benchmark-relative reward
- blended reward
- blended reward + turnover penalty
- downside alpha reward
- theme universe
- safe-growth universe
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)


def patch_common(text: str, exp_name: str, seed: int, timesteps: int) -> str:
    text = re.sub(
        r"RESULTS_DIR\s*=.*",
        f'RESULTS_DIR = f"{{BASE_DIR}}/results/experiments/{exp_name}"',
        text,
        count=1,
    )
    text = re.sub(r"seed\s*:\s*int\s*=\s*\d+", f"seed: int = {seed}", text, count=1)
    text = re.sub(
        r"total_timesteps\s*:\s*int\s*=\s*[0-9_]+",
        f"total_timesteps: int = {timesteps:_}",
        text,
        count=1,
    )

    if "ent_coef:" not in text:
        text = re.sub(
            r"(learning_rate\s*:\s*float\s*=\s*[0-9.eE_-]+)",
            r"\1\n    ent_coef: float = 0.005",
            text,
            count=1,
        )
    else:
        text = re.sub(r"ent_coef\s*:\s*[^=]+=\s*.*", "ent_coef: float = 0.005", text, count=1)

    text = re.sub(r"ent_coef\s*=\s*['\"]auto['\"]", "ent_coef=cfg.ent_coef", text, count=1)
    if "ent_coef=cfg.ent_coef" not in text:
        text = re.sub(
            r"(learning_rate\s*=\s*cfg\.learning_rate,\s*)",
            r"\1\n        ent_coef=cfg.ent_coef,",
            text,
            count=1,
        )
    return text


def patch_reward(text: str, mode: str) -> str:
    if mode == "default":
        return text
    if mode == "benchmark_relative":
        new_reward = "reward = float(port_ret - bench_ret)"
    elif mode == "blended":
        new_reward = "reward = float(0.5 * port_ret + 0.5 * (port_ret - bench_ret))"
    elif mode == "blended_turnover":
        new_reward = "reward = float(0.5 * port_ret + 0.5 * (port_ret - bench_ret) - 0.002 * turnover)"
    elif mode == "downside_alpha":
        new_reward = "reward = float((port_ret - bench_ret) - 0.5 * max(0.0, bench_ret - port_ret))"
    else:
        raise ValueError(mode)

    patched = re.sub(r"reward\s*=\s*float\((.*?)\)", new_reward, text, count=1, flags=re.DOTALL)
    if patched == text:
        patched = text.replace("reward = port_ret", new_reward)
    return patched


def make_experiment(base_file: str, exp_name: str, seed: int, timesteps: int, reward_mode: str) -> None:
    base = ROOT / base_file
    if not base.exists():
        print(f"[skip missing] {base}")
        return
    text = base.read_text(encoding="utf-8")
    text = patch_common(text, exp_name, seed, timesteps)
    text = patch_reward(text, reward_mode)
    header = f'''"""
Generated experiment: {exp_name}
Base: {base_file}
seed={seed}, total_timesteps={timesteps}, ent_coef=0.005, reward_mode={reward_mode}
Run in Colab: !python experiments/{exp_name}.py
"""

'''
    out = OUT / f"{exp_name}.py"
    out.write_text(header + text, encoding="utf-8")
    print("[created]", out)


def main():
    core = "behavioral_sac_model_v6_hybrid_core.py"
    for seed in range(3):
        make_experiment(core, f"v6_fixed_entropy_40k_seed{seed}", seed, 40_000, "default")
        make_experiment(core, f"v6_benchmark_relative_30k_seed{seed}", seed, 30_000, "benchmark_relative")
        make_experiment(core, f"v6_blended_reward_30k_seed{seed}", seed, 30_000, "blended")
        make_experiment(core, f"v6_blended_turnover_30k_seed{seed}", seed, 30_000, "blended_turnover")
        make_experiment(core, f"v6_downside_alpha_30k_seed{seed}", seed, 30_000, "downside_alpha")

    make_experiment("behavioral_sac_model_theme_war_index_spec_tech.py", "theme_blended_reward_30k_seed1", 1, 30_000, "blended")
    make_experiment("behavioral_sac_universe_safe_growth_hybrid.py", "safe_growth_fixed_entropy_30k_seed1", 1, 30_000, "default")
    print("\nDone. Generated files are in:", OUT)

if __name__ == "__main__":
    main()
