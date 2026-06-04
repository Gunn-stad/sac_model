# V12 Champion Stress Test

Champion model: Momentum + Bayesian confidence, monthly top-N selection.

Tests:
A. Top 10 vs Top 20 vs Top 30
B. Equal weight vs risk parity vs minimum variance vs Bayesian weight
C. Transaction costs 0.05%, 0.10%, 0.20%
D. Year-by-year 2020-2026

## VS Code

Copy these files into:

C:\Gunnar\sac_model

Then run:

```powershell
cd C:\Gunnar\sac_model
git add v12_champion_stress_tests.py requirements_v12.txt README_v12.md
git commit -m "Add V12 champion stress tests"
git push
```

## Colab

```python
%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v12.txt
!python v12_champion_stress_tests.py
```

## Outputs

v12_results/v12_summary.csv
v12_results/v12_test_a_topn.csv
v12_results/v12_test_b_weights.csv
v12_results/v12_test_c_costs.csv
v12_results/v12_test_d_year_by_year.csv
v12_results/v12_all_backtests.csv
paper_trading/v12_today_portfolio.csv
