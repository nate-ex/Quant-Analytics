import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PortfolioAnalytics:
    """
    A self-contained analytics object for a portfolio of assets.

    Usage:
        pa = PortfolioAnalytics(prices)
        pa.summary()
        pa.plot_rolling_volatility(['SPY', 'AAPL'])
        pa.plot_correlation_heatmap()
    """

    def __init__(self, prices: pd.DataFrame):
        """
        Initialize with a DataFrame of adjusted closing prices.
        Automatically computes log returns on creation.

        Args:
            prices: DataFrame with dates as index, tickers as columns
        """
        self.prices      = prices
        self.log_returns = self._compute_log_returns()
        self.tickers     = prices.columns.tolist()
        print(f"PortfolioAnalytics initialized: {len(self.tickers)} tickers, "
              f"{len(self.prices)} trading days")

    # ── Private methods (internal helpers) ────────────────────

    def _compute_log_returns(self) -> pd.DataFrame:
        """Compute log returns from prices. Called automatically on init."""
        return np.log(self.prices / self.prices.shift(1)).dropna()

    # ── Core analytics ─────────────────────────────────────────

    def rolling_volatility(self, window: int = 21) -> pd.DataFrame:
        """
        Compute rolling annualized volatility.

        Args:
            window: lookback window in trading days (default 21 = 1 month)

        Returns:
            DataFrame of rolling annualized volatility
        """
        return (self.log_returns
                    .rolling(window=window)
                    .std()
                    .dropna()
                    * (252 ** 0.5))

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> pd.Series:
        """
        Compute annualized Sharpe ratio for each ticker.

        Args:
            risk_free_rate: annual risk-free rate as decimal (default 0.0)

        Returns:
            Series of Sharpe ratios sorted descending
        """
        daily_rf      = risk_free_rate / 252
        excess        = self.log_returns - daily_rf
        mean_excess   = excess.mean() * 252
        annual_std    = excess.std()   * (252 ** 0.5)
        return (mean_excess / annual_std).round(3).sort_values(ascending=False)

    def max_drawdown(self) -> pd.Series:
        """
        Compute maximum peak-to-trough drawdown for each ticker.

        Returns:
            Series of max drawdowns sorted ascending (worst first)
        """
        rolling_max = self.prices.cummax()
        drawdown    = (self.prices - rolling_max) / rolling_max
        return drawdown.min().round(3).sort_values()

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix of log returns.

        Returns:
            Symmetric correlation matrix
        """
        return self.log_returns.corr().round(2)

    def summary(self) -> pd.DataFrame:
        """
        Print a combined summary table: Sharpe, max drawdown,
        annualized return, and annualized volatility for every ticker.

        Returns:
            Summary DataFrame
        """
        annual_return = (self.log_returns.mean() * 252).round(3)
        annual_vol    = (self.log_returns.std() * (252 ** 0.5)).round(3)
        sharpe        = self.sharpe_ratio()
        mdd           = self.max_drawdown()

        summary_df = pd.DataFrame({
            "Annual Return": annual_return,
            "Annual Vol":    annual_vol,
            "Sharpe":        sharpe,
            "Max Drawdown":  mdd
        }).sort_values("Sharpe", ascending=False)

        print("\n── Portfolio Summary ──────────────────────────────")
        print(summary_df.to_string())
        print("───────────────────────────────────────────────────\n")
        return summary_df

    # ── Plotting ───────────────────────────────────────────────

    def plot_rolling_volatility(self, tickers: list = None,
                                window: int = 21) -> None:
        """
        Plot rolling annualized volatility for selected tickers.

        Args:
            tickers: list of tickers to plot (default: all)
            window:  lookback window in trading days
        """
        if tickers is None:
            tickers = self.tickers

        roll_vol = self.rolling_volatility(window=window)

        fig, ax = plt.subplots(figsize=(14, 6))
        for ticker in tickers:
            ax.plot(roll_vol.index, roll_vol[ticker],
                    label=ticker, linewidth=1.5)

        ax.set_title(f"Rolling {window}-Day Annualized Volatility", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("rolling_volatility.png", dpi=150)
        plt.show()

    def plot_correlation_heatmap(self) -> None:
        """Plot a color-coded heatmap of the correlation matrix."""
        corr = self.correlation_matrix()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr.values, cmap=plt.cm.RdBu_r,
                       vmin=-1, vmax=1, aspect="auto")

        tickers = corr.columns.tolist()
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(tickers, fontsize=10)

        for i in range(len(tickers)):
            for j in range(len(tickers)):
                val   = corr.values[i, j]
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Pairwise Correlation Matrix (Log Returns)", fontsize=13)
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150)
        plt.show()

    def plot_sharpe_bars(self) -> None:
        """Bar chart of Sharpe ratios across all tickers."""
        sharpe = self.sharpe_ratio()

        fig, ax = plt.subplots(figsize=(12, 5))
        colors  = ["steelblue" if v >= 0 else "tomato" for v in sharpe.values]
        ax.bar(sharpe.index, sharpe.values, color=colors,
               edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="black",  linewidth=0.8)
        ax.axhline(1, color="green",  linewidth=0.8,
                   linestyle="--", label="Sharpe = 1")
        ax.set_title("Annualized Sharpe Ratio by Ticker", fontsize=14)
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("sharpe_ratios.png", dpi=150)
        plt.show()


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load saved data — no re-downloading needed
    print("Loading data...")
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

    # Create the analytics object — one line, everything ready
    pa = PortfolioAnalytics(prices)

    # Full summary table
    pa.summary()

    # Charts
    pa.plot_rolling_volatility(tickers=["SPY", "AAPL", "NVDA", "GLD"])
    pa.plot_sharpe_bars()
    pa.plot_correlation_heatmap()