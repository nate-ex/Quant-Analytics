"""
execution.py — Execution Algorithm Simulator
Nathan Exelbirt | Summer 2026

Simulates three execution strategies for a large order:
    1. Naive  — single market order, maximum impact
    2. TWAP   — equal slices over time
    3. VWAP   — volume-proportional slices over time

Compares total cost, market impact, and slippage for each.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lob import LimitOrderBook


# ── Market Simulator ───────────────────────────────────────────

def build_simulated_book(lob: LimitOrderBook,
                         mid_price:   float = 150.0,
                         spread:      float = 0.05,
                         depth:       int   = 10,
                         qty_per_level: int = 500) -> None:
    """
    Populate a LOB with realistic-looking limit orders.

    Creates depth levels on both sides around a mid price,
    with quantity at each level representing available liquidity.

    Args:
        lob:           LimitOrderBook to populate
        mid_price:     center price around which to build the book
        spread:        bid-ask spread in dollars
        depth:         number of price levels per side
        qty_per_level: shares available at each price level
    """
    half_spread = spread / 2
    tick        = 0.01  # minimum price increment

    # Build ask side — prices above mid
    for i in range(depth):
        price = round(mid_price + half_spread + i * tick, 4)
        # Liquidity thins out further from mid — realistic shape
        qty   = qty_per_level - i * 30
        if qty > 0:
            lob.add_limit_order("sell", price, qty)

    # Build bid side — prices below mid
    for i in range(depth):
        price = round(mid_price - half_spread - i * tick, 4)
        qty   = qty_per_level - i * 30
        if qty > 0:
            lob.add_limit_order("buy", price, qty)


def replenish_book(lob: LimitOrderBook,
                   mid_price:   float = 150.0,
                   spread:      float = 0.05,
                   qty_per_level: int = 200) -> None:
    """
    Simulate market makers replenishing liquidity after fills.

    In real markets, market makers continuously post new orders
    as their existing orders get filled. This function adds a
    small amount of fresh liquidity at the top of each side.

    Args:
        lob:           LimitOrderBook to replenish
        mid_price:     current mid price
        spread:        bid-ask spread
        qty_per_level: shares to add at each level
    """
    half_spread = spread / 2
    lob.add_limit_order("sell", round(mid_price + half_spread, 4),
                         qty_per_level)
    lob.add_limit_order("buy",  round(mid_price - half_spread, 4),
                         qty_per_level)


# ── Execution Algorithms ───────────────────────────────────────

def execute_naive(total_qty:  int,
                  mid_price:  float = 150.0,
                  spread:     float = 0.05,
                  cost_bps:   float = 5.0) -> dict:
    """
    Naive execution — submit the entire order as one market order.

    This is the baseline. It maximizes market impact because it
    consumes all available liquidity immediately, walking through
    multiple price levels.

    Args:
        total_qty: total shares to buy
        mid_price: starting mid price
        spread:    bid-ask spread
        cost_bps:  additional transaction cost in basis points

    Returns:
        Dictionary of execution results
    """
    lob = LimitOrderBook("AAPL_NAIVE")
    build_simulated_book(lob, mid_price, spread, depth=15,
                         qty_per_level=500)

    # Single market order for the full quantity
    fills    = lob.submit_market_order("buy", total_qty)
    avg_fill = (sum(f.price * f.quantity for f in fills)
                / sum(f.quantity for f in fills))
    slippage = avg_fill - (mid_price + spread / 2)
    cost     = cost_bps / 10000 * avg_fill

    return {
        "algorithm":    "Naive",
        "total_qty":    total_qty,
        "avg_fill":     round(avg_fill, 4),
        "arrival":      mid_price + spread / 2,
        "slippage":     round(slippage, 4),
        "slippage_bps": round(slippage / mid_price * 10000, 2),
        "cost_bps":     cost_bps,
        "num_orders":   1,
        "fills":        fills,
    }


def execute_twap(total_qty:   int,
                 num_slices:  int   = 10,
                 mid_price:   float = 150.0,
                 spread:      float = 0.05,
                 cost_bps:    float = 5.0) -> dict:
    """
    TWAP execution — equal slices at regular time intervals.

    Divides the total order into equal chunks and executes
    one chunk per interval. Simple, predictable, and reduces
    market impact by giving the book time to replenish.

    Args:
        total_qty:  total shares to buy
        num_slices: number of equal time slices
        mid_price:  starting mid price
        spread:     bid-ask spread
        cost_bps:   transaction cost in basis points

    Returns:
        Dictionary of execution results with per-slice detail
    """
    lob         = LimitOrderBook("AAPL_TWAP")
    slice_qty   = total_qty // num_slices
    remainder   = total_qty % num_slices

    build_simulated_book(lob, mid_price, spread,
                         depth=15, qty_per_level=500)

    all_fills   = []
    slice_avgs  = []

    for i in range(num_slices):
        # Last slice absorbs any remainder
        qty = slice_qty + (remainder if i == num_slices - 1 else 0)

        fills = lob.submit_market_order("buy", qty)
        all_fills.extend(fills)

        if fills:
            avg = (sum(f.price * f.quantity for f in fills)
                   / sum(f.quantity for f in fills))
            slice_avgs.append(avg)

        # Replenish liquidity between slices
        replenish_book(lob, mid_price, spread, qty_per_level=200)

    filled_qty = sum(f.quantity for f in all_fills)
    avg_fill   = (sum(f.price * f.quantity for f in all_fills)
                  / filled_qty) if filled_qty > 0 else 0
    slippage   = avg_fill - (mid_price + spread / 2)

    return {
        "algorithm":    "TWAP",
        "total_qty":    total_qty,
        "num_slices":   num_slices,
        "avg_fill":     round(avg_fill, 4),
        "arrival":      mid_price + spread / 2,
        "slippage":     round(slippage, 4),
        "slippage_bps": round(slippage / mid_price * 10000, 2),
        "cost_bps":     cost_bps,
        "num_orders":   num_slices,
        "slice_avgs":   slice_avgs,
        "fills":        all_fills,
    }


def execute_vwap(total_qty:      int,
                 volume_profile: list = None,
                 mid_price:      float = 150.0,
                 spread:         float = 0.05,
                 cost_bps:       float = 5.0) -> dict:
    """
    VWAP execution — slices proportional to expected volume.

    Trades more during high-volume periods (open, close) and
    less during quiet periods (midday). This minimizes footprint
    relative to natural market volume.

    Args:
        total_qty:      total shares to buy
        volume_profile: list of volume weights per interval
                        (must sum to 1.0). If None, uses a
                        realistic U-shaped intraday profile.
        mid_price:      starting mid price
        spread:         bid-ask spread
        cost_bps:       transaction cost in basis points

    Returns:
        Dictionary of execution results with per-slice detail
    """
    # Default: U-shaped volume profile (high at open/close)
    # This reflects real intraday volume patterns on US exchanges
    if volume_profile is None:
        raw     = [8, 6, 4, 3, 3, 3, 3, 3, 4, 5, 6, 8]
        total_v = sum(raw)
        volume_profile = [v / total_v for v in raw]

    lob        = LimitOrderBook("AAPL_VWAP")
    num_slices = len(volume_profile)

    build_simulated_book(lob, mid_price, spread,
                         depth=15, qty_per_level=500)

    all_fills  = []
    slice_qtys = []
    slice_avgs = []

    for i, weight in enumerate(volume_profile):
        qty = round(total_qty * weight)
        slice_qtys.append(qty)

        if qty > 0:
            fills = lob.submit_market_order("buy", qty)
            all_fills.extend(fills)

            if fills:
                avg = (sum(f.price * f.quantity for f in fills)
                       / sum(f.quantity for f in fills))
                slice_avgs.append(avg)

        # Replenish proportional to volume weight
        replenish_qty = max(100, int(300 * weight))
        replenish_book(lob, mid_price, spread,
                       qty_per_level=replenish_qty)

    filled_qty = sum(f.quantity for f in all_fills)
    avg_fill   = (sum(f.price * f.quantity for f in all_fills)
                  / filled_qty) if filled_qty > 0 else 0
    slippage   = avg_fill - (mid_price + spread / 2)

    return {
        "algorithm":    "VWAP",
        "total_qty":    total_qty,
        "num_slices":   num_slices,
        "avg_fill":     round(avg_fill, 4),
        "arrival":      mid_price + spread / 2,
        "slippage":     round(slippage, 4),
        "slippage_bps": round(slippage / mid_price * 10000, 2),
        "cost_bps":     cost_bps,
        "num_orders":   num_slices,
        "slice_qtys":   slice_qtys,
        "slice_avgs":   slice_avgs,
        "fills":        all_fills,
    }


# ── Comparison and Visualization ───────────────────────────────

def print_comparison(results: list) -> None:
    """
    Print a side-by-side comparison of execution results.

    Args:
        results: list of result dicts from execute_* functions
    """
    print(f"\n{'='*60}")
    print(f"  Execution Algorithm Comparison")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<12} {'Avg Fill':>10} {'Slippage':>10} "
          f"{'Slip (bps)':>12} {'# Orders':>10}")
    print(f"  {'-'*56}")
    for r in results:
        print(f"  {r['algorithm']:<12} "
              f"{r['avg_fill']:>10.4f} "
              f"{r['slippage']:>10.4f} "
              f"{r['slippage_bps']:>12.2f} "
              f"{r['num_orders']:>10}")
    print(f"{'='*60}\n")


def plot_comparison(results: list) -> None:
    """
    Three-panel visualization comparing execution algorithms.

    Panel 1: Average fill price vs. arrival price
    Panel 2: Slippage in basis points
    Panel 3: VWAP volume profile and slice sizes

    Args:
        results: list of result dicts from execute_* functions
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Execution Algorithm Comparison — 5,000 Share Order",
                 fontsize=14, fontweight="bold")

    algorithms = [r["algorithm"] for r in results]
    avg_fills  = [r["avg_fill"]     for r in results]
    slippages  = [r["slippage_bps"] for r in results]
    arrival    = results[0]["arrival"]
    colors     = ["tomato", "steelblue", "seagreen"]

    # Panel 1 — Average fill price
    ax = axes[0]
    bars = ax.bar(algorithms, avg_fills, color=colors, edgecolor="white")
    ax.axhline(arrival, color="black", linestyle="--",
               linewidth=1.0, label=f"Arrival price ({arrival})")
    ax.set_title("Average Fill Price")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.set_ylim(min(avg_fills) - 0.05, max(avg_fills) + 0.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2 — Slippage in bps
    ax = axes[1]
    ax.bar(algorithms, slippages, color=colors, edgecolor="white")
    ax.set_title("Slippage vs. Arrival Price")
    ax.set_ylabel("Basis Points")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (algo, slip) in enumerate(zip(algorithms, slippages)):
        ax.text(i, slip + 0.1, f"{slip:.1f} bps",
                ha="center", va="bottom", fontsize=9)

    # Panel 3 — VWAP volume profile
    ax = axes[2]
    vwap_result = next((r for r in results
                        if r["algorithm"] == "VWAP"), None)
    if vwap_result and "slice_qtys" in vwap_result:
        intervals = list(range(len(vwap_result["slice_qtys"])))
        ax.bar(intervals, vwap_result["slice_qtys"],
               color="seagreen", alpha=0.7, edgecolor="white")
        ax.set_title("VWAP — Shares Per Interval")
        ax.set_xlabel("Time Interval")
        ax.set_ylabel("Shares")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("execution_comparison.png", dpi=150)
    plt.show()
    print("Chart saved to execution_comparison.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Execution Algorithm Simulator")
    print("  Simulating a 5,000 share buy order in AAPL")
    print("="*60)

    TOTAL_QTY = 5000
    MID_PRICE = 150.0
    SPREAD    = 0.05
    COST_BPS  = 5.0

    print(f"\n  Order size:  {TOTAL_QTY:,} shares")
    print(f"  Mid price:   ${MID_PRICE}")
    print(f"  Spread:      ${SPREAD} ({SPREAD/MID_PRICE*10000:.1f} bps)")
    print(f"  Notional:    ${TOTAL_QTY * MID_PRICE:,.0f}")

    # Run all three algorithms
    print("\nRunning Naive execution...")
    naive = execute_naive(TOTAL_QTY, MID_PRICE, SPREAD, COST_BPS)

    print("Running TWAP execution (10 slices)...")
    twap  = execute_twap(TOTAL_QTY, num_slices=10,
                         mid_price=MID_PRICE,
                         spread=SPREAD, cost_bps=COST_BPS)

    print("Running VWAP execution (12 intervals)...")
    vwap  = execute_vwap(TOTAL_QTY, mid_price=MID_PRICE,
                         spread=SPREAD, cost_bps=COST_BPS)

    # Compare
    results = [naive, twap, vwap]
    print_comparison(results)

    # Dollar cost of slippage for each
    print("  Dollar cost of slippage on $750,000 notional order:")
    for r in results:
        dollar_cost = r["slippage"] * TOTAL_QTY
        print(f"    {r['algorithm']:<8} "
              f"${dollar_cost:>10,.2f}  "
              f"({r['slippage_bps']:.1f} bps)")

    # Plot
    plot_comparison(results)