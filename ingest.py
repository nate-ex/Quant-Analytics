import yfinance as yf
import pandas as pd

def get_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.
    
    Args:
        tickers: list of ticker strings e.g. ['AAPL', 'MSFT']
        start:   start date string e.g. '2020-01-01'
        end:     end date string e.g. '2024-01-01'
    
    Returns:
        DataFrame with dates as index, tickers as columns
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    prices.columns = [col if isinstance(col, str) else col for col in prices.columns]
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.
    Log returns are preferred in finance because they are:
    - Additive across time (daily returns sum to monthly)
    - More normally distributed than simple returns
    - Symmetric (a 50% drop and 100% gain are equal magnitude)
    """
    import numpy as np
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "JPM", "GS", "BAC",
        "SPY", "QQQ",
        "GLD", "SLV",
        "XLF", "XLK", "NVDA"
    ]

    start_date = "2020-01-01"
    end_date   = "2024-12-31"

    # Pull prices
    print("Downloading price data...")
    prices = get_price_data(tickers, start_date, end_date)

    # Compute log returns
    log_returns = compute_log_returns(prices)

    print(f"\nPrice data shape: {prices.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"\nSample log returns (first 5 rows):")
    print(log_returns.head().round(4))

    print(f"\nAnnualized volatility (std of daily returns * sqrt(252)):")
    annual_vol = log_returns.std() * (252 ** 0.5)
    print(annual_vol.round(4))

    # Save both to CSV
    prices.to_csv("prices.csv")
    log_returns.to_csv("log_returns.csv")
    print("\nSaved prices.csv and log_returns.csv")