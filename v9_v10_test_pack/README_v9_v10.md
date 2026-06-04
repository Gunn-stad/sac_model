# V9 + V10 test pack

## What this tests

### V9
A non-RL alpha ranking model:

- Momentum
- Bayesian confidence
- Behavioral features
- Real Yahoo news sentiment with VADER
- Market regime detection
- Multi-agent voting

### V10
A real reinforcement learning model:

- SAC portfolio manager
- Uses V9 features as state
- Action = portfolio weights
- Reward = excess return minus turnover/risk penalties

## Files

- `v9_alpha_ranking_model.py`
- `v10_sac_portfolio_manager.py`
- `requirements_v9_v10.txt`

## VS Code / GitHub

Save these files inside:

```text
C:\Gunnar\sac_model
```

Then run:

```powershell
cd C:\Gunnar\sac_model
git add v9_alpha_ranking_model.py v10_sac_portfolio_manager.py requirements_v9_v10.txt
git commit -m "Add V9 alpha model and V10 SAC portfolio manager"
git push
```

## Colab

```python
%cd /content
!rm -rf sac_model
!git clone https://github.com/YOUR_USERNAME/sac_model.git
%cd /content/sac_model

!pip install -r requirements_v9_v10.txt

!python v9_alpha_ranking_model.py
!python v10_sac_portfolio_manager.py
```

Replace `YOUR_USERNAME` with your GitHub username.

## Outputs

V9 outputs:

```text
v9_results/v9_summary.csv
v9_results/v9_panel.csv
v9_results/v9_best_score_backtest.csv
```

V10 outputs:

```text
v10_results/v10_sac_summary.csv
v10_results/v10_sac_backtest.csv
v10_results/v10_selected_assets.csv
```