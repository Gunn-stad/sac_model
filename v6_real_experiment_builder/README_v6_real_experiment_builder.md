# V6 Real Experiment Builder

This replaces the earlier wrapper pack.

It creates real files in:

```text
v6_real_experiments/
```

## Use

Put `make_v6_real_experiment_files.py` in:

```text
C:\Gunnar\sac_model
```

Then push and run in Colab:

```python
%cd /content/sac_model
!git pull
!python make_v6_real_experiment_files.py
!ls v6_real_experiments
```

Run first:

```python
!python v6_real_experiments/v6_core_fixed_entropy_40k.py
!python v6_real_experiments/v6_core_low_turnover.py
```
