# V17 Paper Trading App

Adds:
1. Live Performance
2. Current Holdings
3. Sector Exposure
4. Leaderboard
5. Monthly / Hourly History
6. Automatic hourly updater during US market hours

## Push

cd C:\Gunnar\sac_model
git add v17_paper_trading_app.py v17_hourly_auto_updater.py requirements_v17.txt README_v17.md
git commit -m "Add V17 paper trading app and hourly updater"
git push

## Colab

%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v17.txt

!python v12_champion_stress_tests.py
!python v13_defensive_momentum_bayesian.py
!python v14_paper_trading_dashboard.py --init 10000

!nohup python v17_hourly_auto_updater.py > hourly_updater.log 2>&1 &
!nohup streamlit run v17_paper_trading_app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

from pyngrok import ngrok
ngrok.kill()
ngrok.set_auth_token("YOUR_TOKEN_HERE")
url = ngrok.connect(8501)
print(url)

## Check logs

!tail -n 50 hourly_updater.log
!tail -n 50 streamlit.log

## Stop

!pkill -f streamlit
!pkill -f v17_hourly_auto_updater.py