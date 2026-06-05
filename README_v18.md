# V18 Iceland Market Test

Tests whether the Momentum + Bayesian model transfers to Icelandic stocks.

## Files

- v18_iceland_market_test.py
- requirements_v18.txt
- README_v18.md

## Push

cd C:\\Gunnar\\sac_model
git add v18_iceland_market_test.py requirements_v18.txt README_v18.md
git commit -m "Add V18 Iceland market test"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v18.txt
!python v18_iceland_market_test.py

## Outputs

v18_iceland_results/v18_iceland_summary.csv
v18_iceland_results/v18_iceland_backtests.csv
v18_iceland_results/v18_iceland_panel.csv
paper_trading/v18_iceland_today_portfolio.csv