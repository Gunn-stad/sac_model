# RL Testing Pack

Save all files in:

```text
C:\Gunnar\sac_model
```

Then in VS Code / PowerShell:

```powershell
cd C:\Gunnar\sac_model
python experiment_factory.py

git add .
git commit -m "Add RL robustness testing experiments"
git push
```

In Colab:

```python
%cd /content
!rm -rf sac_model
!git clone https://github.com/Gunn-stad/sac_model.git
%cd /content/sac_model

!pip install stable-baselines3 gymnasium shimmy yfinance pyarrow textblob pytrends vaderSentiment

!python run_priority_tests_colab.py
!python aggregate_experiment_results.py
!python make_checklist_report.py
```

## Reward functions tested

Benchmark-relative:

```text
R_t = portfolio_return - equal_weight_return
```

Blended reward:

```text
R_t = 0.5 * portfolio_return + 0.5 * (portfolio_return - equal_weight_return)
```

Blended with turnover penalty:

```text
R_t = 0.5 * portfolio_return + 0.5 * alpha - 0.002 * turnover
```

Downside alpha:

```text
R_t = alpha - 0.5 * max(0, -alpha)
```

where:

```text
alpha = portfolio_return - equal_weight_return
```
