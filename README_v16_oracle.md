# V16 Dashboard + Oracle VM Updater

Files:
- v16_visual_dashboard_plus.py
- v16_hourly_updater.py
- requirements_v16.txt

## VS Code

cd C:\Gunnar\sac_model
git add v16_visual_dashboard_plus.py v16_hourly_updater.py requirements_v16.txt README_v16_oracle.md
git commit -m "Add V16 visual dashboard and hourly updater"
git push

## Run locally

Terminal 1:
py -m streamlit run v16_visual_dashboard_plus.py

Terminal 2:
py v16_hourly_updater.py

## Oracle VM short setup

sudo apt update
sudo apt install -y git python3-pip python3-venv

git clone https://github.com/YOUR_USERNAME/sac_model.git
cd sac_model

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements_v16.txt
pip install -r requirements_v12.txt
pip install -r requirements_v13.txt
pip install -r requirements_v14.txt

python v12_champion_stress_tests.py
python v13_defensive_momentum_bayesian.py
python v14_paper_trading_dashboard.py --init 10000

nohup python v16_hourly_updater.py > hourly_updater.log 2>&1 &
nohup streamlit run v16_visual_dashboard_plus.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &