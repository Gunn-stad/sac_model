# V11 Hybrid Alpha Allocator

## Put these files in VS Code

Save inside:

C:\Gunnar\sac_model

Files:
- v11_hybrid_alpha_allocator.py
- requirements_v11.txt

## Push to GitHub

cd C:\Gunnar\sac_model
git add v11_hybrid_alpha_allocator.py requirements_v11.txt
git commit -m "Add V11 hybrid alpha allocator"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v11.txt
!python v11_hybrid_alpha_allocator.py

## Outputs

v11_results/v11_summary.csv
v11_results/v11_backtests.csv
v11_results/v11_walk_forward.csv
v11_results/v11_crash_tests.csv
paper_trading/v11_today_portfolio.csv