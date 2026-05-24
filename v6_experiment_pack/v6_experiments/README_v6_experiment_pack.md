# V6 Experiment Pack

These scripts keep the successful v6/universe-test structure and test one idea at a time.

Put the `v6_experiments` folder into your repo root:

```text
C:\Gunnar\sac_model\v6_experiments
```

Push to GitHub, then in Colab:

```python
%cd /content/sac_model
!git pull
```

Recommended first runs:

```python
!python v6_experiments/v6_exp_fixed_entropy_short_training.py
!python v6_experiments/v6_exp_low_turnover.py
!python v6_experiments/v6_exp_higher_exposure.py
!python v6_experiments/v6_exp_theme_assets_fixed_entropy.py
```

All scripts are wrappers: they patch a temporary copy of your successful v6 code and run it.
This is safer than rewriting v7 from scratch.
