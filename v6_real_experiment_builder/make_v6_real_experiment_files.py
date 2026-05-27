"""
make_v6_real_experiment_files.py

Creates real standalone v6 experiment files from your successful base scripts.
"""

from pathlib import Path
import re

CORE_BASE = Path("behavioral_sac_model_v6_hybrid_core.py")
THEME_BASE = Path("behavioral_sac_model_theme_war_index_spec_tech.py")

OUT_DIR = Path("v6_real_experiments")
OUT_DIR.mkdir(exist_ok=True)


def replace_assignment(text: str, var_name: str, value_code: str) -> str:
    pattern = rf"^{var_name}\s*=.*$"

    new_text = re.sub(
        pattern,
        f"{var_name} = {value_code}",
        text,
        flags=re.MULTILINE,
    )

    if new_text != text:
        return new_text

    m = re.search(
        r"^BASE_DIR\s*=.*$",
        text,
        flags=re.MULTILINE,
    )

    if m:
        pos = m.end()
        return text[:pos] + f"\n{var_name} = {value_code}" + text[pos:]

    return f"{var_name} = {value_code}\n" + text


def patch_results_dir(text: str, exp_name: str) -> str:
    pattern = r"^RESULTS_DIR\s*=.*$"

    replacement = (
        f'RESULTS_DIR = f"{{BASE_DIR}}/results/v6_real_experiments/{exp_name}"'
    )

    text2 = re.sub(
        pattern,
        replacement,
        text,
        flags=re.MULTILINE,
    )

    if text2 == text:
        text2 = replace_assignment(
            text,
            "RESULTS_DIR",
            f'f"{{BASE_DIR}}/results/v6_real_experiments/{exp_name}"'
        )

    return text2


def patch_universe_name(text: str, exp_name: str) -> str:
    return replace_assignment(
        text,
        "UNIVERSE_NAME",
        repr(exp_name),
    )


def patch_total_timesteps(text: str, timesteps: int) -> str:
    changed = False

    for var in [
        "TOTAL_TIMESTEPS",
        "TIMESTEPS",
        "N_TIMESTEPS",
        "TRAIN_TIMESTEPS",
    ]:
        pattern = rf"^{var}\s*=.*$"

        new_text = re.sub(
            pattern,
            f"{var} = {timesteps}",
            text,
            flags=re.MULTILINE,
        )

        if new_text != text:
            text = new_text
            changed = True

    pattern = r"total_timesteps\s*=\s*[^,\)\n]+"

    new_text = re.sub(
        pattern,
        f"total_timesteps={timesteps}",
        text,
    )

    if new_text != text:
        text = new_text
        changed = True

    pattern = r"model\.learn\(\s*[0-9_]+"

    new_text = re.sub(
        pattern,
        f"model.learn({timesteps}",
        text,
    )

    if new_text != text:
        text = new_text
        changed = True

    if not changed:
        text = replace_assignment(
            text,
            "TOTAL_TIMESTEPS",
            str(timesteps),
        )

        print(
            "[warning] Did not find existing timestep pattern; "
            "inserted TOTAL_TIMESTEPS only."
        )

    return text


def patch_entropy(text: str, ent_coef: str = "0.005") -> str:
    pattern = r"ent_coef\s*=\s*[^,\)\n]+"

    text2 = re.sub(
        pattern,
        f"ent_coef={ent_coef}",
        text,
    )

    if text2 != text:
        return text2

    pattern_lr = r"(learning_rate\s*=\s*[^,\)\n]+,)"

    text2 = re.sub(
        pattern_lr,
        rf"\1\n        ent_coef={ent_coef},",
        text,
        count=1,
    )

    if text2 != text:
        return text2

    return re.sub(
        r"SAC\(",
        f"SAC(\n        ent_coef={ent_coef},",
        text,
        count=1,
    )


def patch_low_turnover(text: str) -> str:
    text = replace_assignment(text, "LAMBDA_TURNOVER", "0.006")
    text = replace_assignment(text, "TRANSACTION_COST", "0.001")
    text = replace_assignment(text, "SMOOTHING", "0.25")
    text = replace_assignment(text, "REBALANCE_EVERY", "6")
    return text


def patch_higher_exposure(text: str) -> str:
    text = replace_assignment(text, "MIN_EXPOSURE", "0.85")
    text = replace_assignment(text, "CASH_PENALTY", "0.003")
    return text


def patch_safer_exposure(text: str) -> str:
    text = replace_assignment(text, "MIN_EXPOSURE", "0.65")
    text = replace_assignment(text, "LAMBDA_RISK", "0.05")
    text = replace_assignment(text, "LAMBDA_CONC", "0.03")
    return text


def write_experiment(
    base_path: Path,
    out_name: str,
    exp_name: str,
    patches,
):
    if not base_path.exists():
        print(f"[skip] missing base file: {base_path}")
        return

    text = base_path.read_text(encoding="utf-8")

    text = patch_universe_name(text, exp_name)
    text = patch_results_dir(text, exp_name)

    for fn in patches:
        text = fn(text)

    out_path = OUT_DIR / out_name

    out_path.write_text(
        text,
        encoding="utf-8",
    )

    print(f"[ok] wrote {out_path}")


def main():
    write_experiment(
        CORE_BASE,
        "v6_core_fixed_entropy_40k.py",
        "v6_core_fixed_entropy_40k",
        [
            lambda t: patch_total_timesteps(t, 40_000),
            lambda t: patch_entropy(t, "0.005"),
        ],
    )

    write_experiment(
        CORE_BASE,
        "v6_core_low_turnover.py",
        "v6_core_low_turnover",
        [
            patch_low_turnover,
            lambda t: patch_entropy(t, "0.005"),
        ],
    )

    write_experiment(
        CORE_BASE,
        "v6_core_higher_exposure.py",
        "v6_core_higher_exposure",
        [
            patch_higher_exposure,
            lambda t: patch_entropy(t, "0.005"),
        ],
    )

    write_experiment(
        CORE_BASE,
        "v6_core_safer_exposure.py",
        "v6_core_safer_exposure",
        [
            patch_safer_exposure,
            lambda t: patch_entropy(t, "0.005"),
        ],
    )

    write_experiment(
        THEME_BASE,
        "v6_theme_fixed_entropy_40k.py",
        "v6_theme_fixed_entropy_40k",
        [
            lambda t: patch_total_timesteps(t, 40_000),
            lambda t: patch_entropy(t, "0.005"),
        ],
    )

    print("\n[done] generated all v6 real experiment files")


if __name__ == "__main__":
    main()