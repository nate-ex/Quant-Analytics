# Backtest Analysis — GLD/SLV and SPY/QQQ Pairs Trading

## Strategy
Mean reversion on log price spread. Entry at ±2 z-score, exit at 0.

## Results Summary
| Pair | Window | Entry | Sharpe | Max DD |
|------|--------|-------|--------|--------|
| GLD/SLV | 21d | 2.0 | -0.521 | -69.8% |
| SPY/QQQ | 63d | 2.0 | -0.073 | -17.6% |
| SPY/QQQ | 63d | 1.5 | -0.256 | -23.4% |

## Why GLD/SLV Failed
Silver strongly outperformed gold, they did not converge.

## Why SPY/QQQ Failed
Too closely correlated. Rarely met the entry threshold and when it did, transaction costs ate up the profits. Essentially a coin flip with transaction costs.

## Key Lessons
Mean reversion only works on a pair that is truly correlated, but not too closely correlated.

## What I Would Test Next
Find new pairs. Also try trading ETFs against all the individual underlying stocks.