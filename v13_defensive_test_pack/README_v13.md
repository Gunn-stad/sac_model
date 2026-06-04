# V13 Defensive Momentum-Bayesian

Tests:
2. Sector diversification
3. Safe asset allocation
4. Crash tests with safe assets
5. Live portfolio construction

## VS Code

Put files in:

C:\Gunnar\sac_model

Then:

git add v13_defensive_momentum_bayesian.py requirements_v13.txt README_v13.md
git commit -m "Add V13 defensive momentum Bayesian tests"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v13.txt
!python v13_defensive_momentum_bayesian.py

## Outputs

v13_results/v13_summary.csv
v13_results/v13_sector_test.csv
v13_results/v13_safe_asset_test.csv
v13_results/v13_crash_test.csv
v13_results/v13_live_construction_test.csv
paper_trading/v13_today_portfolio.csv