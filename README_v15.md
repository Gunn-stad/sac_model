# V15 Visual Paper Trading Dashboard

This adds a visual Streamlit dashboard for the V14 paper trading system.

## Files

- v15_visual_paper_dashboard.py
- requirements_v15.txt

## VS Code

Put files in:

C:\Gunnar\sac_model

Then:

git add v15_visual_paper_dashboard.py requirements_v15.txt README_v15.md
git commit -m "Add V15 visual paper dashboard"
git push

## Run locally

In PowerShell:

cd C:\Gunnar\sac_model
pip install -r requirements_v15.txt
streamlit run v15_visual_paper_dashboard.py

## Run in Colab

%cd /content/sac_model
!pip install -r requirements_v15.txt
!streamlit run v15_visual_paper_dashboard.py --server.port 8501 &

## Before using

Make sure V14 has already created:

paper_dashboard/portfolio_state.csv
paper_dashboard/daily_log.csv
paper_dashboard/dashboard_summary.csv

If not:

python v14_paper_trading_dashboard.py --init 10000

or:

python v14_paper_trading_dashboard.py --update