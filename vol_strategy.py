import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── Volatility Regime Strategy ─────────────────────────────────

def compute_rolling_vol(log_returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Compute rolling annualized volatility for a single asset.

    Args:
        log_returns: Series of daily log returns
        window:      lookback window in trading days

    Returns:
        Series of rolling annualized volatility
    """
    return log_returns.rolling(window=window).std() * (252 ** 0.5)


def generate_vol_signals(rolling_vol: pd.Series,
                         threshold:   float = 0.20) -> pd.Series:
    """
    Generate signals based on volatility regime.

    Rules:
        +1 = long  (vol below threshold — calm market)
         0 = cash  (vol above threshold — stressed market)

    Signal is shifted forward 1 day to prevent lookahead bias.
    Signal on day T uses vol computed through day T, executes day T+1.

    Args:
        rolling_vol: Series of rolling annualized volatility
        threshold:   annualized vol level that triggers exit to cash

    Returns:
        Series of signals: +1 or 0
    """
    signal = (rolling_vol < threshold).astype(int)
    return signal.shift(1).fillna(0)


def run_vol_backtest(log_returns: pd.Series,
                     signals:     pd.Series,
                     cost_bps:    float = 5.0) -> pd.DataFrame:
    """
    Simulate the volatility regime strategy.

    When signal = 1: earn the market return that day
    When signal = 0: earn nothing (sitting in cash)

    Transaction costs applied on every position change.

    Args:
        log_returns: Series of daily log returns for the asset
        signals:     Series of signals (1 = invested, 0 = cash)
        cost_bps:    one-way transaction cost in basis points

    Returns:
        DataFrame with daily returns, strategy returns, cumulative PnL
    """
    # Strategy return = signal * market return
    strategy_return = signals * log_returns

    # Transaction costs on position changes
    position_changes = signals.diff().abs()
    cost_per_unit    = cost_bps / 10000
    costs            = position_changes * cost_per_unit

    # Net return
    net_return = strategy_return - costs

    # Cumulative returns — strategy vs. buy and hold
    cumulative_strategy = net_return.cumsum()
    cumulative_bh       = log_returns.cumsum()

    return pd.DataFrame({
        "market_ret":    log_returns,
        "signal":        signals,
        "strategy_ret":  strategy_return,
        "costs":         costs,
        "net_ret":       net_return,
        "cum_strategy":  cumulative_strategy,
        "cum_bh":        cumulative_bh
    }).dropna()


def performance_metrics(results: pd.DataFrame,
                        label:   str = "Strategy") -> dict:
    """
    Compute performance metrics for both strategy and buy-and-hold.

    Args:
        results: output DataFrame from run_vol_backtest()
        label:   name label for printing

    Returns:
        Dictionary of metrics for strategy and buy-and-hold
    """
    def metrics(returns: pd.Series, cum: pd.Series) -> dict:
        annual_return = returns.mean() * 252
        annual_vol    = returns.std()  * (252 ** 0.5)
        sharpe        = annual_return / annual_vol if annual_vol != 0 else 0
        rolling_max   = cum.cummax()
        max_dd        = (cum - rolling_max).min()
        time_invested = (returns != 0).mean()

        return {
            "Annual Return":  round(annual_return,  4),
            "Annual Vol":     round(annual_vol,      4),
            "Sharpe Ratio":   round(sharpe,          3),
            "Max Drawdown":   round(max_dd,          4),
            "Time Invested":  round(time_invested,   3),
        }

    strat = metrics(results["net_ret"],    results["cum_strategy"])
    bh    = metrics(results["market_ret"], results["cum_bh"])

    return {"strategy": strat, "buy_and_hold": bh}


def plot_results(results:      pd.DataFrame,
                 rolling_vol:  pd.Series,
                 signals:      pd.Series,
                 threshold:    float = 0.20,
                 ticker:       str   = "SPY") -> None:
    """
    Four-panel chart:
    1. Cumulative PnL — strategy vs. buy and hold
    2. Rolling volatility with threshold marked
    3. Daily signal (invested vs. cash)
    4. Underwater chart (drawdown over time)

    Args:
        results:     output from run_vol_backtest()
        rolling_vol: Series of rolling annualized volatility
        signals:     Series of signals
        threshold:   vol threshold used
        ticker:      asset name for chart title
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Volatility Regime Strategy — {ticker}",
                 fontsize=15, fontweight="bold")

    # Panel 1 — Cumulative PnL
    ax = axes[0]
    ax.plot(results.index, results["cum_strategy"],
            label="Vol Strategy", color="steelblue", linewidth=1.5)
    ax.plot(results.index, results["cum_bh"],
            label="Buy & Hold",   color="gray",      linewidth=1.5,
            linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Cumulative Log Return: Strategy vs. Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — Rolling volatility
    ax = axes[1]
    ax.plot(rolling_vol.index, rolling_vol,
            color="darkorange", linewidth=1.2, label="21d Rolling Vol")
    ax.axhline(threshold, color="red", linestyle="--",
               linewidth=1.0, label=f"Threshold ({threshold:.0%})")
    ax.fill_between(rolling_vol.index, rolling_vol, threshold,
                    where=(rolling_vol > threshold),
                    alpha=0.2, color="red", label="High vol (cash)")
    ax.fill_between(rolling_vol.index, rolling_vol, threshold,
                    where=(rolling_vol <= threshold),
                    alpha=0.2, color="green", label="Low vol (invested)")
    ax.set_title("Rolling 21-Day Annualized Volatility")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Signal
    ax = axes[2]
    ax.fill_between(signals.index, signals, 0,
                    where=(signals == 1),
                    alpha=0.6, color="steelblue", label="Invested")
    ax.fill_between(signals.index, signals, 0,
                    where=(signals == 0),
                    alpha=0.3, color="gray", label="Cash")
    ax.set_title("Position: Invested vs. Cash")
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4 — Drawdown
    ax = axes[3]
    cum_strat   = results["cum_strategy"]
    rolling_max = cum_strat.cummax()
    drawdown    = cum_strat - rolling_max
    ax.fill_between(drawdown.index, drawdown, 0,
                    alpha=0.4, color="red")
    ax.plot(drawdown.index, drawdown,
            color="red", linewidth=1.0)
    ax.set_title("Strategy Drawdown")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("vol_strategy.png", dpi=150)
    plt.show()
    print("Chart saved to vol_strategy.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    prices      = pd.read_csv("prices.csv",      index_col=0, parse_dates=True)
    log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)


    # Parameters
    TICKER    = "SPY"
    WINDOW    = 21
    THRESHOLD = 0.20
    COST_BPS  = 5.0

    print(f"\nRunning volatility regime strategy on {TICKER}")
    print(f"Window: {WINDOW} days | Threshold: {THRESHOLD:.0%} | Costs: {COST_BPS}bps\n")

    # Compute rolling vol and signals
    spy_returns  = log_returns[TICKER]
    rolling_vol  = compute_rolling_vol(spy_returns, window=WINDOW)
    signals      = generate_vol_signals(rolling_vol, threshold=THRESHOLD)

    # Run backtest
    results = run_vol_backtest(spy_returns, signals, cost_bps=COST_BPS)

    # Print metrics
    metrics = performance_metrics(results)

    print("── Strategy ───────────────────────────────────────")
    for k, v in metrics["strategy"].items():
        print(f"  {k:<20} {v}")

    print("\n── Buy & Hold ─────────────────────────────────────")
    for k, v in metrics["buy_and_hold"].items():
        print(f"  {k:<20} {v}")
    print("───────────────────────────────────────────────────")

    # Plot
    plot_results(results, rolling_vol, signals,
                 threshold=THRESHOLD, ticker=TICKER)