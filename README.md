Reinforcement Learning Portfolio Management with Soft Actor-Critic

Overview

This project explores the use of reinforcement learning for portfolio allocation and risk-aware investment management.

The system uses a Soft Actor-Critic (SAC) agent to dynamically allocate capital across a portfolio of stocks and defensive assets using historical market data and engineered financial features.

The goal is not only to maximize returns, but also to control portfolio risk, transaction costs, turnover, concentration, and drawdowns.

The project includes:

- Financial market data collection
- Feature engineering
- Reinforcement learning environment design
- Soft Actor-Critic training
- Portfolio constraints and transaction costs
- Risk-aware reward functions
- Historical backtesting
- Benchmark comparison
- Paper-trading infrastructure
- Market regime detection
- News and sentiment features
- Risk-managed portfolio variants

---

Motivation

Traditional portfolio strategies often use fixed allocation rules or optimization methods based on historical statistics.

This project investigates whether a reinforcement-learning agent can learn a dynamic portfolio allocation policy by interacting with financial market data.

The agent observes market conditions, chooses portfolio weights, receives a reward based on portfolio performance and risk, and gradually learns a policy for allocating capital.

---

System Architecture

Market Data
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Market / Risk Features
     ↓
Reinforcement Learning Environment
     ↓
Soft Actor-Critic Agent
     ↓
Portfolio Weights
     ↓
Transaction Costs + Risk Constraints
     ↓
Portfolio Return
     ↓
Backtesting / Paper Trading
     ↓
Performance Evaluation

---

Reinforcement Learning Formulation

The portfolio-management problem is modeled as a Markov Decision Process.

State

The state contains market and portfolio information such as:

- Short-term returns
- Medium-term returns
- Moving averages
- Volatility
- Z-scores
- Relative returns
- Beta
- Cross-sectional rankings
- Market-regime information
- Current portfolio allocation
- Risk indicators

Examples of engineered features include:

return_1
return_4
return_24
return_48
return_72

ma_8
ma_24
ma_72

vol_24
zscore_24
beta_24
relative_return_24

---

Actions

The SAC agent outputs continuous portfolio allocation weights.

For example:

AAPL    0.10
MSFT    0.12
NVDA    0.08
JPM     0.07
XLV     0.10
SHY     0.35
...

Portfolio constraints are applied to prevent unrealistic allocations.

Examples include:

- Maximum allocation per asset
- Portfolio-weight normalization
- Transaction-cost penalties
- Turnover penalties
- Concentration penalties

---

Reward Function

The reward function is designed to balance return and risk.

A simplified representation is:

reward =
portfolio_return
- transaction_cost
- risk_penalty
- concentration_penalty
- benchmark_penalty

This allows the agent to learn that maximizing raw return alone may not produce the best portfolio.

Risk-adjusted performance is therefore an important part of the system.

---

Soft Actor-Critic

Soft Actor-Critic is an off-policy reinforcement-learning algorithm designed for continuous action spaces.

It learns:

- An actor policy
- Two critic networks
- Target critic networks
- An entropy coefficient

The entropy term encourages exploration while the agent learns portfolio allocations.

Important training parameters used during experimentation include:

Network:       256 × 256
Gamma:         0.99
Tau:           0.005
Batch size:    256
Learning rate: ~1e-4
Training:      200k–300k steps

Different configurations were tested during development.

---

Financial Data

Historical market data is collected primarily using Python financial-data tools.

The project has experimented with universes containing:

- Large-cap US stocks
- Technology stocks
- Sector ETFs
- Defensive ETFs
- Treasury / short-duration bond ETFs
- Semiconductor ETFs
- Healthcare and financial-sector assets

Example assets include:

AAPL
MSFT
NVDA
GOOGL
AMZN
META
AMD
JPM
XOM
XLK
XLV
XLF
SMH
SOXX
SHY
SPY

---

Feature Engineering

The system transforms raw price data into information that the RL agent can use.

Features include:

Momentum

1-hour return
4-hour return
24-hour return
48-hour return
72-hour return

Trend

8-period moving average
24-period moving average
72-period moving average

Risk

rolling volatility
beta
z-score
market volatility

Relative Performance

Assets are compared with:

- SPY
- Other assets in the portfolio
- Sector benchmarks

Cross-sectional rankings are also used to identify relative strength.

---

Risk Management

Later versions of the project introduced additional risk controls.

These include:

- Defensive asset allocation
- Volatility estimation
- Maximum position sizes
- Portfolio concentration penalties
- Transaction costs
- Turnover control
- Market-regime detection
- Risk-on / risk-off allocation
- Defensive bond exposure

For example, during higher-risk conditions the portfolio may allocate more capital to defensive assets such as short-duration Treasury ETFs.

---

Market Regime Detection

The project also experiments with identifying different market environments.

Possible regimes include:

Risk On
Neutral
Risk Off

Signals may include:

- SPY momentum
- Realized volatility
- Technology-sector leadership
- Relative asset performance
- Market risk indicators

The regime information can influence portfolio construction and risk limits.

---

News and Sentiment Features

The project has also explored adding external information beyond price data.

This includes:

- Financial news
- News sentiment
- FinBERT sentiment analysis
- Macroeconomic information
- Sector-specific news
- AI / technology news

The goal is to test whether external information improves performance compared with price-based features alone.

---

Backtesting

Each strategy is evaluated on historical data.

Metrics include:

- Total return
- CAGR
- Sharpe ratio
- Maximum drawdown
- Annualized volatility
- Best day
- Worst day
- Portfolio turnover
- Transaction costs
- Number of positions
- Performance relative to SPY

Example output:

Strategy          Total Return   CAGR    Sharpe   Max Drawdown
--------------------------------------------------------------
SAC Strategy          ...         ...      ...         ...
SPY Benchmark         ...         ...      ...         ...
Equal Weight          ...         ...      ...         ...

Actual results depend on the test period and configuration.

---

Paper Trading

The project includes a paper-trading system that simulates portfolio performance using live market data without risking real capital.

The pipeline can:

1. Download new market data
2. Build the latest features
3. Generate target portfolio allocations
4. Update simulated holdings
5. Calculate portfolio value
6. Record performance
7. Compare the strategy against SPY

Portfolio history is saved for later analysis.

---

Project Evolution

The system has been developed incrementally.

Examples of different research versions include:

Initial MDP / Q-Learning model
        ↓
Portfolio reinforcement learning
        ↓
Soft Actor-Critic
        ↓
Portfolio constraints
        ↓
Risk management
        ↓
Defensive assets
        ↓
Market-regime detection
        ↓
News features
        ↓
Sentiment analysis
        ↓
Paper trading

Each major feature is tested independently before being added to the larger system.

---

Technologies

Programming

- Python

Machine Learning

- Reinforcement Learning
- Soft Actor-Critic
- Q-Learning
- Markov Decision Processes

Data

- Pandas
- NumPy
- yfinance

Machine Learning Libraries

- PyTorch
- Stable-Baselines3 / custom RL components where applicable

Analysis

- Statistical analysis
- Time-series analysis
- Portfolio optimization
- Risk modelling

Development

- Git
- GitHub
- VS Code
- Jupyter Notebook
- Google Colab
- PowerShell

---

Repository Structure

An example project structure:

sac-portfolio/
│
├── data/
│   └── market data
│
├── features/
│   └── feature engineering
│
├── environment/
│   └── portfolio RL environment
│
├── models/
│   └── trained SAC models
│
├── backtests/
│   └── backtesting scripts
│
├── paper_trading/
│   └── live paper-trading system
│
├── risk/
│   └── portfolio risk management
│
├── results/
│   └── performance results
│
├── requirements.txt
│
└── README.md

---

Running the Project

Install the required packages:

pip install -r requirements.txt

Example:

python backtest.py

or:

python train_sac.py

Exact commands depend on the project version being tested.

---

Research Questions

This project investigates questions such as:

- Can SAC outperform simple portfolio strategies?
- Does reinforcement learning improve risk-adjusted returns?
- Can defensive assets reduce drawdowns?
- Does market-regime detection improve portfolio allocation?
- Do financial news and sentiment provide useful predictive information?
- How sensitive is performance to transaction costs?
- How should risk penalties be incorporated into the RL reward function?
- Can the model generalize to unseen market conditions?

---

Limitations

Financial reinforcement learning has several important limitations.

Historical performance does not guarantee future performance.

Potential issues include:

- Overfitting
- Survivorship bias
- Look-ahead bias
- Transaction-cost assumptions
- Market-regime changes
- Limited historical data
- Model instability
- Differences between backtesting and real execution

The project is therefore primarily a research and portfolio-development project rather than a claim of guaranteed investment performance.

---

Future Work

Planned areas of research include:

- Improved sentiment modelling
- Macroeconomic features
- Better regime detection
- Dynamic risk constraints
- Drawdown-aware reward functions
- Automated hyperparameter optimization
- Walk-forward validation
- Out-of-sample testing
- Portfolio stress testing
- Improved execution modelling
- Comparison against classical portfolio optimization
- Combining return-maximization and risk-minimization models

---

