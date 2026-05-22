# Pre-live / Pre-paper-trading test pack

Run these before trusting the SAC model with paper trading or live trading.

## 1. Universe robustness

```python
!python run_universe_robustness_tests.py
```

Purpose: checks whether the model only works on one lucky asset universe.

## 2. Early stopping

```python
!python run_early_stopping_tests.py
```

Purpose: checks whether training too long hurts out-of-sample results.

## 3. Regime and stress analysis

First set `REPLAY_FILE` inside:

```text
analyze_regime_and_stress_tests.py
```

Then run:

```python
!python analyze_regime_and_stress_tests.py
```

Purpose: checks how the model behaves in high-volatility, negative-market, and shock regimes.

## 4. Aggregate all results

```python
!python aggregate_pre_live_results.py
```

Purpose: creates one CSV containing model comparisons.

## Minimum criteria before paper trading

Recommended thresholds:

```text
1. Same model works across more than one universe.
2. Sharpe is not dependent on one lucky seed.
3. Max drawdown is lower than or close to equal-weight benchmark.
4. Turnover is realistic after transaction costs.
5. Validation and test are not wildly different.
6. Model survives high-volatility/stress periods.
```

## Suggested workflow

```python
!python behavioral_sac_model_v6_hybrid_core.py
!python run_universe_robustness_tests.py
!python run_early_stopping_tests.py
!python aggregate_pre_live_results.py
```
