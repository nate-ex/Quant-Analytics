import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── Spread and Z-Score ─────────────────────────────────────────

def compute_spread(prices: pd.DataFrame,
                   ticker_a: str,
                   ticker_b: str) -> pd.Series:
    """
    Compute the spread between two assets as the difference
    of their log prices. Using log prices keeps the spread
    stationary and symmetric.

    Args:
        prices:   DataFrame of adjusted closing prices
        ticker_a: first ticker
        ticker_b: second ticker

    Returns:
        Series of daily spread values
    """
    log_a = np.log(prices[ticker_a])
    log_b = np.log(prices[ticker_b])
    return log_a - log_b


def compute_zscore(spread: pd.Series, window: int = 21) -> pd.Series:
    """
    Compute the rolling z-score of a spread series.

    Z-score = (spread - rolling mean) / rolling std

    Tells you how many standard deviations the current spread
    is away from its recent average. Values beyond +/- 2 are
    statistically unusual and likely to revert.

    Args:
        spread: Series of spread values
        window: lookback window in trading days

    Returns:
        Series of rolling z-scores
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std  = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std


# ── Signal Generation ──────────────────────────────────────────

def generate_signals(zscore: pd.Series,
                     entry_threshold: float = 2.0,
                     exit_threshold:  float = 0.0) -> pd.Series:
    """
    Generate trading signals from a z-score series.

    Rules:
        +1  = long the spread  (zscore < -entry_threshold)
        -1  = short the spread (zscore > +entry_threshold)
         0  = flat / no position

    Signals are shifted forward by 1 day to prevent lookahead
    bias — the signal generated on day T executes on day T+1.

    Args:
        zscore:           Series of rolling z-scores
        entry_threshold:  z-score level to enter a trade (default 2.0)
        exit_threshold:   z-score level to exit a trade  (default 0.0)

    Returns:
        Series of signals: +1, -1, or 0
    """
    signal = pd.Series(0, index=zscore.index)

    position = 0
    for i in range(1, len(zscore)):
        z = zscore.iloc[i]

        if pd.isna(z):
            signal.iloc[i] = 0
            continue

        # Entry logic
        if position == 0:
            if z > entry_threshold:
                position = -1   # spread too high — short it
            elif z < -entry_threshold:
                position = 1    # spread too low  — long it

        # Exit logic
        elif position == 1 and z >= exit_threshold:
            position = 0
        elif position == -1 and z <= exit_threshold:
            position = 0

        signal.iloc[i] = position

    # Shift forward 1 day — critical lookahead bias prevention
    return signal.shift(1).fillna(0)


# ── Backtesting Engine ─────────────────────────────────────────

def run_backtest(prices:     pd.DataFrame,
                 signals:    pd.Series,
                 ticker_a:   str,
                 ticker_b:   str,
                 cost_bps:   float = 5.0) -> pd.DataFrame:
    """
    Simulate trading the spread strategy on historical data.

    When signal = +1: long ticker_a, short ticker_b
    When signal = -1: short ticker_a, long ticker_b
    When signal =  0: flat, no position

    Transaction costs are applied every time the position changes.

    Args:
        prices:   DataFrame of adjusted closing prices
        signals:  Series of signals (+1, -1, 0)
        ticker_a: first ticker in the pair
        ticker_b: second ticker in the pair
        cost_bps: one-way transaction cost in basis points (default 5bps)

    Returns:
        DataFrame with daily strategy returns and cumulative PnL
    """
    # Daily log returns for each leg
    ret_a = np.log(prices[ticker_a] / prices[ticker_a].shift(1))
    ret_b = np.log(prices[ticker_b] / prices[ticker_b].shift(1))

    # Spread return: long A short B = ret_a - ret_b
    spread_return = ret_a - ret_b

    # Strategy return = signal * spread return
    strategy_return = signals * spread_return

    # Transaction costs — applied when position changes
    position_changes = signals.diff().abs()
    cost_per_unit    = cost_bps / 10000
    costs            = position_changes * cost_per_unit

    # Net return after costs
    net_return = strategy_return - costs

    # Cumulative PnL
    cumulative = net_return.cumsum()

    return pd.DataFrame({
        "signal":      signals,
        "spread_ret":  spread_return,
        "strat_ret":   strategy_return,
        "costs":       costs,
        "net_ret":     net_return,
        "cumulative":  cumulative
    }).dropna()


# ── Performance Metrics ────────────────────────────────────────

def performance_metrics(results: pd.DataFrame) -> dict:
    """
    Compute key performance metrics from backtest results.

    Args:
        results: output DataFrame from run_backtest()

    Returns:
        Dictionary of performance metrics
    """
    net = results["net_ret"]

    annual_return = net.mean() * 252
    annual_vol    = net.std()  * (252 ** 0.5)
    sharpe        = annual_return / annual_vol if annual_vol != 0 else 0

    cumulative    = results["cumulative"]
    rolling_max   = cumulative.cummax()
    drawdown      = cumulative - rolling_max
    max_dd        = drawdown.min()

    trades        = results["signal"].diff().abs()
    n_trades      = int((trades > 0).sum())
    win_days      = int((net > 0).sum())
    total_days    = int((net != 0).sum())
    win_rate      = win_days / total_days if total_days > 0 else 0

    total_costs   = results["costs"].sum()

    return {
        "Annual Return":  round(annual_return, 4),
        "Annual Vol":     round(annual_vol,    4),
        "Sharpe Ratio":   round(sharpe,        3),
        "Max Drawdown":   round(max_dd,        4),
        "Num Trades":     n_trades,
        "Win Rate":       round(win_rate,      3),
        "Total Costs":    round(total_costs,   4),
    }


# ── Plotting ───────────────────────────────────────────────────

def plot_results(prices:   pd.DataFrame,
                 spread:   pd.Series,
                 zscore:   pd.Series,
                 signals:  pd.Series,
                 results:  pd.DataFrame,
                 ticker_a: str,
                 ticker_b: str) -> None:
    """
    Four-panel chart showing the full backtest story:
    1. Asset prices normalized to 100
    2. Spread between the two assets
    3. Z-score with entry/exit thresholds marked
    4. Cumulative PnL of the strategy
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Pairs Trading Backtest: {ticker_a} / {ticker_b}",
                 fontsize=15, fontweight="bold")

    # Panel 1 — Normalized prices
    ax = axes[0]
    norm_a = prices[ticker_a] / prices[ticker_a].iloc[0] * 100
    norm_b = prices[ticker_b] / prices[ticker_b].iloc[0] * 100
    ax.plot(norm_a.index, norm_a, label=ticker_a, linewidth=1.5)
    ax.plot(norm_b.index, norm_b, label=ticker_b, linewidth=1.5)
    ax.set_title("Normalized Prices (base = 100)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — Spread
    ax = axes[1]
    ax.plot(spread.index, spread, color="purple", linewidth=1.2)
    ax.axhline(spread.mean(), color="black", linestyle="--",
               linewidth=0.8, label="Mean")
    ax.set_title("Log Price Spread")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3 — Z-score with signals
    ax = axes[2]
    ax.plot(zscore.index, zscore, color="steelblue", linewidth=1.2)
    ax.axhline( 2.0, color="red",   linestyle="--", linewidth=0.8, label="Entry (+2)")
    ax.axhline(-2.0, color="green", linestyle="--", linewidth=0.8, label="Entry (-2)")
    ax.axhline( 0.0, color="black", linestyle="-",  linewidth=0.6, label="Exit (0)")

    # Shade long and short regions
    ax.fill_between(zscore.index, zscore, 0,
                    where=(signals == 1),
                    alpha=0.2, color="green", label="Long spread")
    ax.fill_between(zscore.index, zscore, 0,
                    where=(signals == -1),
                    alpha=0.2, color="red", label="Short spread")
    ax.set_title("Z-Score with Entry/Exit Signals")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4 — Cumulative PnL
    ax = axes[3]
    ax.plot(results.index, results["cumulative"],
            color="darkgreen", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(results.index, results["cumulative"], 0,
                    where=(results["cumulative"] >= 0),
                    alpha=0.2, color="green")
    ax.fill_between(results.index, results["cumulative"], 0,
                    where=(results["cumulative"] < 0),
                    alpha=0.2, color="red")
    ax.set_title("Cumulative PnL (log return units)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150)
    plt.show()
    print("Chart saved to backtest_results.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load price data
    print("Loading data...")
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

    # Parameters
    TICKER_A = "SPY"
    TICKER_B = "SPYG"
    WINDOW   = 63

    print(f"\nRunning pairs backtest: {TICKER_A} / {TICKER_B}")
    print(f"Z-score window: {WINDOW} days")
    print(f"Entry threshold: ±2 standard deviations")
    print(f"Exit threshold:  0.0 (mean reversion)")

    # Build spread and signals
    spread  = compute_spread(prices, TICKER_A, TICKER_B)
    zscore  = compute_zscore(spread, window=WINDOW)
    signals = generate_signals(zscore, entry_threshold=2)

    # Run backtest
    results = run_backtest(prices, signals, TICKER_A, TICKER_B, cost_bps=5.0)

    # Print metrics
    metrics = performance_metrics(results)
    print("\n── Backtest Results ───────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print("───────────────────────────────────────────────────")

    # Plot
    plot_results(prices, spread, zscore, signals, results, TICKER_A, TICKER_B)