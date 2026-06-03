# Goal Testing Pack

Save all files in:

```text
C:\Gunnar\sac_model
```

Files:

```text
goal_experiment_factory.py
run_goal_tests_colab.py
run_all_seed_tests_colab.py
aggregate_goal_results.py
make_goal_checklist_report.py
README_GOAL_TESTING_PACK.md
```

## VS Code / PowerShell

```powershell
cd C:\Gunnar\sac_model

python goal_experiment_factory.py

git add .
git commit -m "Add goal testing experiments"
git push
```

## Colab

```python
%cd /content
!rm -rf sac_model
!git clone https://github.com/Gunn-stad/sac_model.git
%cd /content/sac_model

!pip install stable-baselines3 gymnasium shimmy yfinance pyarrow textblob pytrends vaderSentiment

!python run_goal_tests_colab.py
!python aggregate_goal_results.py
!python make_goal_checklist_report.py
```

## Run 10 seeds

```python
!python run_all_seed_tests_colab.py
!python aggregate_goal_results.py
!python make_goal_checklist_report.py
```
