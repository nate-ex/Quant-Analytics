import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ── Rolling Volatility ─────────────────────────────────────────

def rolling_volatility(log_returns: pd.DataFrame,
                       window: int = 21) -> pd.DataFrame:
    """
    Compute rolling annualized volatility for each ticker.

    Args:
        log_returns: DataFrame of daily log returns
        window:      number of trading days to look back (default 21 = ~1 month)

    Returns:
        DataFrame of rolling annualized volatility
    """
    daily_vol   = log_returns.rolling(window=window).std()
    annual_vol  = daily_vol * (252 ** 0.5)
    return annual_vol.dropna()


# ── Sharpe Ratio ───────────────────────────────────────────────

def sharpe_ratio(log_returns: pd.DataFrame,
                 risk_free_rate: float = 0.0) -> pd.Series:
    """
    Compute the annualized Sharpe ratio for each ticker.

    Sharpe ratio = (mean return - risk free rate) / std of returns
    Annualized by multiplying mean by 252 and std by sqrt(252).

    Args:
        log_returns:     DataFrame of daily log returns
        risk_free_rate:  annual risk-free rate as a decimal (default 0.0)

    Returns:
        Series of Sharpe ratios, one per ticker
    """
    daily_rf      = risk_free_rate / 252
    excess_return = log_returns - daily_rf
    mean_excess   = excess_return.mean() * 252
    annual_std    = excess_return.std()   * (252 ** 0.5)
    return (mean_excess / annual_std).round(3)


# ── Max Drawdown ───────────────────────────────────────────────

def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Compute the maximum drawdown for each ticker.

    Max drawdown = the largest peak-to-trough decline in the price series.
    Expressed as a negative percentage.

    Args:
        prices: DataFrame of asset prices

    Returns:
        Series of max drawdowns, one per ticker
    """
    rolling_max  = prices.cummax()
    drawdown     = (prices - rolling_max) / rolling_max
    return drawdown.min().round(3)


# ── Plotting ───────────────────────────────────────────────────

def plot_rolling_volatility(rolling_vol: pd.DataFrame,
                            tickers: list,
                            window: int = 21) -> None:
    """
    Plot rolling volatility over time for a selection of tickers.

    Args:
        rolling_vol: DataFrame of rolling annualized volatility
        tickers:     list of tickers to include in the plot
        window:      the window used (for the chart title)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for ticker in tickers:
        ax.plot(rolling_vol.index, rolling_vol[ticker], label=ticker, linewidth=1.5)

    ax.set_title(f"Rolling {window}-Day Annualized Volatility", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rolling_volatility.png", dpi=150)
    plt.show()
    print("Chart saved to rolling_volatility.png")


def plot_sharpe_bars(sharpe: pd.Series) -> None:
    """
    Bar chart of Sharpe ratios across all tickers.

    Args:
        sharpe: Series of Sharpe ratios
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["steelblue" if v >= 0 else "tomato" for v in sharpe.values]
    ax.bar(sharpe.index, sharpe.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="green", linewidth=0.8, linestyle="--", label="Sharpe = 1 (good)")
    ax.set_title("Annualized Sharpe Ratio by Ticker (2020–2024)", fontsize=14)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("sharpe_ratios.png", dpi=150)
    plt.show()
    print("Chart saved to sharpe_ratios.png")


def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pairwise correlation matrix of log returns.

    A value close to +1 means two assets move together strongly.
    A value close to -1 means they move inversely.
    A value near 0 means they are largely independent.

    Args:
        log_returns: DataFrame of daily log returns

    Returns:
        Symmetric correlation matrix DataFrame
    """
    return log_returns.corr().round(2)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    """
    Visualize the correlation matrix as a color-coded heatmap.

    Args:
        corr_matrix: output of correlation_matrix()
    """
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 10))

    # Build a clean red-white-blue colormap
    cmap = plt.cm.RdBu_r

    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Tick labels
    tickers = corr_matrix.columns.tolist()
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(tickers, fontsize=10)

    # Annotate each cell with the correlation value
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pairwise Correlation Matrix of Log Returns (2020–2024)", fontsize=13)

    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Chart saved to correlation_heatmap.png")

# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load saved data
    print("Loading data...")
    prices      = pd.read_csv("prices.csv",      index_col=0, parse_dates=True)
    log_returns = pd.read_csv("log_returns.csv", index_col=0, parse_dates=True)
    print(f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} trading days\n")

    # Rolling volatility
    roll_vol_21 = rolling_volatility(log_returns, window=21)
    roll_vol_63 = rolling_volatility(log_returns, window=63)

    # Sharpe ratios
    sharpe = sharpe_ratio(log_returns)
    print("Sharpe Ratios (2020-2024):")
    print(sharpe.sort_values(ascending=False))

    # Max drawdown
    mdd = max_drawdown(prices)
    print("\nMax Drawdowns:")
    print(mdd.sort_values())

    # Correlation matrix
    print("\nCorrelation Matrix:")
    corr = correlation_matrix(log_returns)
    print(corr)

    # Charts
    print("\nGenerating charts...")
    plot_rolling_volatility(roll_vol_21,
                            tickers=["SPY", "AAPL", "GS", "GLD"],
                            window=21)
    plot_sharpe_bars(sharpe)
    plot_correlation_heatmap(corr)

    print("\nDay 2 complete.")