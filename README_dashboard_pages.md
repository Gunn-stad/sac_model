# Dashboard Pages Update

Adds:
1. Portfolio / Holdings
2. Performance
3. Drawdowns
4. Trades
5. Benchmarks
6. Sector / Safe Assets

## Push

cd C:\Gunnar\sac_model

git add v18_paper_trading_dashboard.py requirements_dashboard_pages.txt README_dashboard_pages.md
git commit -m "Add performance drawdown trades benchmark dashboard pages"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_dashboard_pages.txt

!python v12_champion_stress_tests.py
!python v13_defensive_momentum_bayesian.py
!python v14_paper_trading_dashboard.py --init 10000

!nohup streamlit run v18_paper_trading_dashboard.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

from pyngrok import ngrok
ngrok.kill()
ngrok.set_auth_token("YOUR_REAL_TOKEN")
url = ngrok.connect(8501)
print(url)