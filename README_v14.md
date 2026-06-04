# V14 Paper Trading Dashboard

Tracks fake-money paper trading for 30-90 days.

## Before running

You should already have these files from V12 and V13:

paper_trading/v12_today_portfolio.csv
paper_trading/v13_today_portfolio.csv

If not, run:

python v12_champion_stress_tests.py
python v13_defensive_momentum_bayesian.py

## VS Code / GitHub

Put files in:

C:\Gunnar\sac_model

Then:

git add v14_paper_trading_dashboard.py requirements_v14.txt README_v14.md
git commit -m "Add V14 paper trading dashboard"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v14.txt

# First, create V12 and V13 portfolio files if needed:
!python v12_champion_stress_tests.py
!python v13_defensive_momentum_bayesian.py

# Initialize dashboard with $10,000 fake money per strategy:
!python v14_paper_trading_dashboard.py --init 10000

# Update daily or weekly:
!python v14_paper_trading_dashboard.py --update

# Monthly, after rerunning V12/V13:
!python v14_paper_trading_dashboard.py --rebalance

# See current positions:
!python v14_paper_trading_dashboard.py --positions

## Outputs

paper_dashboard/portfolio_state.csv
paper_dashboard/daily_log.csv
paper_dashboard/dashboard_summary.csv
paper_dashboard/equity_curve.png