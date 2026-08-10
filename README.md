# Quant Analytics Toolkit

**Nathan Exelbirt** | Duke University, Statistical Science '29  
Built: Summer 2026 | Data: January 2020 – May 2026

A Python toolkit for quantitative financial analysis built from scratch
in 5 days as part of a structured quant trading self-study program.
No high-level quant libraries used — every tool is built from first
principles using numpy and pandas.

---

## What This Does

Five modules, each with a single clear responsibility:

| Module | Purpose |
|--------|---------|
| `ingest.py` | Downloads and caches adjusted price data for any list of tickers via yfinance |
| `portfolio.py` | Computes rolling volatility, Sharpe ratio, max drawdown, and correlation matrix |
| `backtest.py` | Simulates pairs trading strategies with transaction costs and lookahead bias prevention |
| `vol_strategy.py` | Volatility regime strategy — invested in SPY during calm markets, cash during stress |
| `main.py` | Single entry point that runs the full pipeline end to end |

---

## How to Run

```bash
git clone https://github.com/nate-ex/quant-analytics.git
cd quant-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install yfinance pandas numpy matplotlib
python3 main.py
```

---

## Key Findings

### 1. Portfolio Analytics — 18 Assets, 2020–2026

| Asset | Sharpe | Annual Return | Annual Vol | Max Drawdown |
|-------|--------|--------------|------------|--------------|
| NVDA | 1.088 | 56.6% | 52.1% | -66.3% |
| GLD | 0.921 | 16.7% | 18.1% | -22.0% |
| GOOGL | 0.847 | 27.4% | 32.3% | -44.3% |
| SPY | 0.705 | 14.4% | 20.4% | -33.7% |
| BAC | 0.242 | 8.2% | 33.8% | -48.9% |

**Key insight:** Raw returns are misleading without risk context. NVDA
returned 56.6% annually but drew down 66.3% — most investors would
have been stopped out before the recovery. GLD delivered a Sharpe of
0.921 with only 22% max drawdown, making it the best risk-adjusted
non-tech asset over this period. BAC took on nearly as much volatility
as NVDA but delivered a fraction of the return.

---

### 2. Correlation Structure

Three distinct clusters emerge from the pairwise correlation matrix:

- **Tech cluster** — AAPL, MSFT, NVDA, QQQ, XLK correlated 0.75–0.98
- **Financial cluster** — BAC, JPM, GS, XLF correlated 0.83–0.92
- **Defensive** — GLD and SLV largely uncorrelated with equities (GLD/SPY: 0.16)

**Key insight:** Owning multiple assets within the same cluster provides
almost no diversification. GLD's near-zero correlation with financials
(GLD/BAC: -0.01) is the mathematical proof of why institutions hold
gold as a hedge — not because it goes up when stocks fall, but because
it is genuinely independent.

---

### 3. Pairs Trading Backtest

Tested mean reversion on the log price spread across three pairs.
Entry at ±2 standard deviations, exit at 0, 5bps transaction cost.

| Pair | Window | Sharpe | Max Drawdown | Finding |
|------|--------|--------|-------------|---------|
| GLD/SLV | 21d | -0.447 | -108.9% | Structural divergence — silver underperformed gold permanently |
| SPY/QQQ | 63d | -0.082 | -17.6% | Too correlated — spread never moves enough to overcome costs |
| SPY/SPYG | 63d | -0.111 | -10.4% | Same issue — essentially identical assets |

**Key insight:** Mean reversion requires a pair that is correlated enough
to reliably revert but different enough that the spread moves meaningfully.
None of the tested pairs hit that balance. GLD/SLV failed due to
cointegration breakdown. SPY/QQQ and SPY/SPYG failed because high
correlation means near-zero spread movement after transaction costs.
Next step: formal cointegration testing (Engle-Granger) before pair selection.

---

### 4. Volatility Regime Strategy

A simple rule applied to SPY: invest when 21-day rolling annualized
volatility is below a threshold, hold cash when above.
Tested on full period and confirmed on post-COVID subset (2021–2026).

| Configuration | Sharpe | Annual Return | Max Drawdown | Time Invested |
|--------------|--------|--------------|-------------|---------------|
| 20% threshold | 0.881 | 10.7% | -17.0% | 76.2% |
| 15% threshold | 0.862 | 7.9% | -10.5% | 56.5% |
| Buy & Hold | 0.711 | 14.5% | -41.1% | 99.9% |

**Key insight:** Both configurations significantly improve Sharpe vs.
buy and hold by stepping aside during high-volatility periods — the
COVID crash and 2022 rate hike selloff account for most of buy and
hold's -41% drawdown. The tradeoff is clear: tighter thresholds reduce
drawdown but sacrifice raw return. The 20% threshold now outperforms
15% on Sharpe when 2025–2026 data is included, demonstrating that
overly conservative thresholds get whipsawed during volatile-but-rising
markets. Result holds post-COVID, confirming it is not a one-event artifact.

---

## Technical Concepts Implemented

- **Log returns** — additivity across time, symmetry, approximate normality
- **Rolling volatility** — 21-day and 63-day annualized vol windows
- **Sharpe and Sortino ratios** — risk-adjusted return measurement
- **Maximum drawdown** — peak-to-trough loss over full history
- **Correlation matrix** — pairwise asset relationships and cluster structure
- **Z-score signal generation** — entry and exit based on standard deviations from rolling mean
- **Lookahead bias prevention** — all signals shifted forward 1 day before execution
- **Transaction cost modeling** — fixed basis points applied on every position change
- **Volatility regime detection** — rolling vol as a market state classifier

---

## Honest Limitations

- Backtests cover one specific historical period and may not generalize
- Vol strategy threshold was chosen without formal optimization or walk-forward validation
- Pairs trading results suggest cointegration testing is required before live trading
- No out-of-sample testing performed — all results are in-sample

---

## What I Would Build Next

- Cointegration testing (Engle-Granger) to find genuinely stationary pairs
- Walk-forward backtesting to validate strategies on truly unseen data
- Multi-asset volatility regime strategy applied across sectors
- Limit order book simulator for execution microstructure analysis
- Transaction cost analysis (TCA) tool — pre-trade cost estimation

---

## About

Built as part of a structured quant trading preparation program
leading into a summer internship on the floor of the New York Stock Exchange.
Target: quantitative trading roles at systematic trading firms.
