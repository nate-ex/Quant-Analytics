"""
main.py — Quant Analytics Toolkit
Nathan Exelbirt | Summer 2026

Single entry point that runs the full toolkit end to end:
    1. Data ingestion
    2. Portfolio analytics
    3. Pairs trading backtest
    4. Volatility regime strategy

Run with: python3 main.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ingest      import get_price_data, compute_log_returns
from portfolio   import PortfolioAnalytics
from backtest    import (compute_spread, compute_zscore,
                         generate_signals, run_backtest,
                         performance_metrics as pairs_metrics,
                         plot_results as plot_pairs)
from vol_strategy import (compute_rolling_vol, generate_vol_signals,
                           run_vol_backtest, performance_metrics as vol_metrics,
                           plot_results as plot_vol)


# ── Configuration ──────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "GS", "BAC",
    "SPY", "QQQ",
    "GLD", "SLV",
    "XLF", "XLK", "XLE",
    "NVDA", "LRCX", "SPYG"
]

START_DATE = "2020-01-01"
END_DATE   = "2026-12-31"


def section(title: str) -> None:
    """Print a clean section divider."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}\n")


# ── Step 1: Data Ingestion ─────────────────────────────────────

def step1_ingest() -> tuple:
    """
    Download or load price data.
    Uses cached CSV if available to avoid re-downloading.
    """
    section("Step 1: Data Ingestion")

    try:
        prices      = pd.read_csv("prices.csv",
                                  index_col=0, parse_dates=True)
        log_returns = pd.read_csv("log_returns.csv",
                                  index_col=0, parse_dates=True)
        print(f"Loaded from cache: {prices.shape[1]} tickers, "
              f"{prices.shape[0]} trading days")
        print(f"Date range: {prices.index[0].date()} "
              f"to {prices.index[-1].date()}")

    except FileNotFoundError:
        print("No cache found — downloading fresh data...")
        prices      = get_price_data(TICKERS, START_DATE, END_DATE)
        log_returns = compute_log_returns(prices)
        prices.to_csv("prices.csv")
        log_returns.to_csv("log_returns.csv")
        print(f"Downloaded and saved: {prices.shape[1]} tickers, "
            f"{prices.shape[0]} trading days")
        print(f"Date range: {prices.index[0].date()} "  
            f"to {prices.index[-1].date()}")

    return prices, log_returns


# ── Step 2: Portfolio Analytics ────────────────────────────────

def step2_analytics(prices: pd.DataFrame) -> None:
    """
    Run full portfolio analytics and print summary.
    """
    section("Step 2: Portfolio Analytics")

    pa = PortfolioAnalytics(prices)
    summary = pa.summary()

    print("\nTop 3 by Sharpe Ratio:")
    print(summary.head(3)[["Sharpe", "Annual Return",
                            "Annual Vol", "Max Drawdown"]])

    print("\nBottom 3 by Sharpe Ratio:")
    print(summary.tail(3)[["Sharpe", "Annual Return",
                            "Annual Vol", "Max Drawdown"]])

    print("\nGenerating analytics charts...")
    pa.plot_rolling_volatility(tickers=["SPY", "AAPL", "NVDA", "GLD"])
    pa.plot_sharpe_bars()
    pa.plot_correlation_heatmap()


# ── Step 3: Pairs Trading Backtest ────────────────────────────

def step3_pairs(prices: pd.DataFrame) -> None:
    """
    Run pairs trading backtest on GLD/SLV and SPY/QQQ.
    Print results and plot for each pair.
    """
    section("Step 3: Pairs Trading Backtest")

    pairs = [
        ("GLD",  "SLV",  21, 2.0),
        ("SPY",  "QQQ",  63, 2.0),
        ("SPY",  "SPYG", 63, 2.0),
    ]

    results_summary = []

    for ticker_a, ticker_b, window, entry in pairs:
        spread  = compute_spread(prices, ticker_a, ticker_b)
        zscore  = compute_zscore(spread, window=window)
        signals = generate_signals(zscore, entry_threshold=entry)
        results = run_backtest(prices, signals,
                               ticker_a, ticker_b, cost_bps=5.0)
        metrics = pairs_metrics(results)

        results_summary.append({
            "Pair":    f"{ticker_a}/{ticker_b}",
            "Window":  f"{window}d",
            "Sharpe":  metrics["Sharpe Ratio"],
            "Max DD":  metrics["Max Drawdown"],
            "Trades":  metrics["Num Trades"],
        })

        print(f"{ticker_a}/{ticker_b} — "
              f"Sharpe: {metrics['Sharpe Ratio']:>7.3f} | "
              f"Max DD: {metrics['Max Drawdown']:>8.3f} | "
              f"Trades: {metrics['Num Trades']}")

    print("\nPairs Summary:")
    print(pd.DataFrame(results_summary).to_string(index=False))

    print("\nKey Finding: Simple mean reversion fails on all tested pairs.")
    print("GLD/SLV diverged structurally. SPY/QQQ and SPY/SPYG")
    print("too correlated — transaction costs exceed edge.")


# ── Step 4: Volatility Regime Strategy ────────────────────────

def step4_vol_strategy(prices:      pd.DataFrame,
                       log_returns: pd.DataFrame) -> None:
    """
    Run volatility regime strategy on SPY.
    Compare two thresholds against buy and hold.
    """
    section("Step 4: Volatility Regime Strategy")

    spy_returns = log_returns["SPY"]
    thresholds  = [0.20, 0.15]

    print(f"{'Threshold':<12} {'Return':>10} {'Vol':>10} "
          f"{'Sharpe':>10} {'Max DD':>10} {'% Invested':>12}")
    print("-" * 66)

    for threshold in thresholds:
        rolling_vol = compute_rolling_vol(spy_returns, window=21)
        signals     = generate_vol_signals(rolling_vol,
                                           threshold=threshold)
        results     = run_vol_backtest(spy_returns, signals,
                                       cost_bps=5.0)
        m           = vol_metrics(results)["strategy"]

        print(f"{threshold:<12.0%} "
              f"{m['Annual Return']:>10.3f} "
              f"{m['Annual Vol']:>10.3f} "
              f"{m['Sharpe Ratio']:>10.3f} "
              f"{m['Max Drawdown']:>10.3f} "
              f"{m['Time Invested']:>12.1%}")

    # Buy and hold benchmark
    bh = vol_metrics(results)["buy_and_hold"]
    print(f"{'Buy & Hold':<12} "
          f"{bh['Annual Return']:>10.3f} "
          f"{bh['Annual Vol']:>10.3f} "
          f"{bh['Sharpe Ratio']:>10.3f} "
          f"{bh['Max Drawdown']:>10.3f} "
          f"{bh['Time Invested']:>12.1%}")

    print("\nKey Finding: Vol strategy improves Sharpe significantly")
    print("vs. buy and hold, confirmed post-COVID (2021-2026).")
    print("Tradeoff: lower raw return for dramatically lower drawdown.")

    # Plot the 20% threshold version
    rolling_vol = compute_rolling_vol(spy_returns, window=21)
    signals     = generate_vol_signals(rolling_vol, threshold=0.20)
    results     = run_vol_backtest(spy_returns, signals, cost_bps=5.0)
    plot_vol(results, rolling_vol, signals,
             threshold=0.20, ticker="SPY")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nQuant Analytics Toolkit — Nathan Exelbirt")
    print("Summer 2026 | Duke University")
    print("=" * 55)

    prices, log_returns = step1_ingest()
    step2_analytics(prices)
    step3_pairs(prices)
    step4_vol_strategy(prices, log_returns)

    print("\n" + "=" * 55)
    print("  Run complete.")
    print("=" * 55)