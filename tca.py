"""
tca.py — Transaction Cost Analysis Model
Nathan Exelbirt | Summer 2026

Decomposes the total cost of executing a trade into its
component parts: spread cost, market impact, timing cost,
and opportunity cost.

This is the core analytical product that execution firms
provide to their institutional clients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ── Trade Representation ───────────────────────────────────────

@dataclass
class Trade:
    """
    A single trade to be analyzed.

    Attributes:
        ticker:         instrument symbol
        side:           'buy' or 'sell'
        target_qty:     shares intended to trade
        filled_qty:     shares actually filled
        arrival_price:  mid price when the decision to trade was made
        avg_fill_price: volume-weighted average price actually achieved
        final_price:    mid price at the end of the execution window
        bid_ask_spread: average bid-ask spread during execution
        adv:            average daily volume of the instrument
    """
    ticker:         str
    side:           str
    target_qty:     int
    filled_qty:     int
    arrival_price:  float
    avg_fill_price: float
    final_price:    float
    bid_ask_spread: float
    adv:            int


# ── Market Impact Model ────────────────────────────────────────

def estimate_market_impact(order_qty: int,
                           adv:       int,
                           volatility: float = 0.20,
                           coefficient: float = 0.1) -> float:
    """
    Estimate market impact using the square-root model.

    The square-root model is the industry standard for estimating
    how much a trade moves the price. Impact grows with the square
    root of order size relative to ADV — meaning impact is concave:
    doubling your order size does NOT double your impact.

    Impact (bps) = coefficient * volatility * sqrt(order_qty / adv) * 10000

    Args:
        order_qty:   shares being traded
        adv:         average daily volume
        volatility:  annualized volatility of the instrument
        coefficient: model calibration constant (typically 0.1 to 1.0)

    Returns:
        Estimated market impact in basis points
    """
    participation_rate = order_qty / adv
    impact_fraction    = coefficient * volatility * np.sqrt(participation_rate)
    return impact_fraction * 10000  # convert to basis points


# ── TCA Decomposition ──────────────────────────────────────────

def analyze_trade(trade: Trade) -> dict:
    """
    Decompose the total cost of a trade into its components.

    All costs are expressed in basis points relative to the
    arrival price, and signed so that positive = cost (bad)
    and negative = price improvement (good).

    Args:
        trade: a Trade object to analyze

    Returns:
        Dictionary of cost components and totals
    """
    direction = 1 if trade.side == "buy" else -1
    arrival   = trade.arrival_price

    # ── 1. Implementation Shortfall (total cost) ──
    # The all-in difference between what you paid and the arrival price
    is_per_share = direction * (trade.avg_fill_price - arrival)
    is_bps       = is_per_share / arrival * 10000

    # ── 2. Spread cost ──
    # Half the spread is the minimum cost of crossing to trade
    spread_cost_per_share = trade.bid_ask_spread / 2
    spread_bps            = spread_cost_per_share / arrival * 10000

    # ── 3. Market impact ──
    # How much the fill price exceeded arrival beyond the spread
    impact_per_share = max(0, is_per_share - spread_cost_per_share)
    impact_bps       = impact_per_share / arrival * 10000

    # ── 4. Timing cost ──
    # Price drift over the execution window (on filled portion)
    timing_per_share = direction * (trade.final_price - arrival)
    timing_bps       = timing_per_share / arrival * 10000

    # ── 5. Opportunity cost (on unfilled shares) ──
    # What the unfilled portion cost as price moved away
    unfilled_qty       = trade.target_qty - trade.filled_qty
    fill_rate          = trade.filled_qty / trade.target_qty
    opp_per_share      = direction * (trade.final_price - arrival)
    opp_cost_bps       = (opp_per_share / arrival * 10000
                          * (unfilled_qty / trade.target_qty))

    # ── Dollar costs ──
    is_dollars     = is_per_share * trade.filled_qty
    spread_dollars = spread_cost_per_share * trade.filled_qty
    impact_dollars = impact_per_share * trade.filled_qty

    return {
        "ticker":            trade.ticker,
        "side":              trade.side,
        "target_qty":        trade.target_qty,
        "filled_qty":        trade.filled_qty,
        "fill_rate":         round(fill_rate, 3),
        "arrival_price":     arrival,
        "avg_fill_price":    trade.avg_fill_price,
        "implementation_shortfall_bps": round(is_bps, 2),
        "spread_cost_bps":   round(spread_bps, 2),
        "market_impact_bps": round(impact_bps, 2),
        "timing_cost_bps":   round(timing_bps, 2),
        "opportunity_cost_bps": round(opp_cost_bps, 2),
        "is_dollars":        round(is_dollars, 2),
        "spread_dollars":    round(spread_dollars, 2),
        "impact_dollars":    round(impact_dollars, 2),
    }


def print_tca_report(analysis: dict) -> None:
    """
    Print a clean, professional TCA report for a single trade.

    Args:
        analysis: output from analyze_trade()
    """
    print(f"\n{'='*55}")
    print(f"  Transaction Cost Analysis — {analysis['ticker']}")
    print(f"{'='*55}")
    print(f"  Side:            {analysis['side'].upper()}")
    print(f"  Target Qty:      {analysis['target_qty']:,}")
    print(f"  Filled Qty:      {analysis['filled_qty']:,} "
          f"({analysis['fill_rate']:.1%})")
    print(f"  Arrival Price:   ${analysis['arrival_price']:.4f}")
    print(f"  Avg Fill Price:  ${analysis['avg_fill_price']:.4f}")
    print(f"\n  {'Cost Component':<28} {'bps':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Spread Cost':<28} "
          f"{analysis['spread_cost_bps']:>8.2f}")
    print(f"  {'Market Impact':<28} "
          f"{analysis['market_impact_bps']:>8.2f}")
    print(f"  {'Timing Cost':<28} "
          f"{analysis['timing_cost_bps']:>8.2f}")
    print(f"  {'Opportunity Cost':<28} "
          f"{analysis['opportunity_cost_bps']:>8.2f}")
    print(f"  {'-'*38}")
    print(f"  {'Implementation Shortfall':<28} "
          f"{analysis['implementation_shortfall_bps']:>8.2f}")
    print(f"\n  Dollar Cost (filled portion): "
          f"${analysis['is_dollars']:,.2f}")
    print(f"{'='*55}\n")


# ── Pre-Trade Cost Estimation ──────────────────────────────────

def estimate_pretrade_cost(order_qty:  int,
                           adv:        int,
                           price:      float,
                           spread:     float,
                           volatility: float = 0.20) -> dict:
    """
    Estimate the expected cost of a trade BEFORE executing it.

    This is what a trader uses to decide how aggressively to trade.
    A large order in an illiquid stock has high expected impact —
    the trader might split it over several days.

    Args:
        order_qty:  shares to trade
        adv:        average daily volume
        price:      current price
        spread:     bid-ask spread
        volatility: annualized volatility

    Returns:
        Dictionary of estimated costs
    """
    participation = order_qty / adv
    spread_bps    = (spread / 2) / price * 10000
    impact_bps    = estimate_market_impact(order_qty, adv, volatility)
    total_bps     = spread_bps + impact_bps

    notional      = order_qty * price
    total_dollars = total_bps / 10000 * notional

    return {
        "order_qty":      order_qty,
        "participation":  round(participation, 4),
        "spread_bps":     round(spread_bps, 2),
        "impact_bps":     round(impact_bps, 2),
        "total_bps":      round(total_bps, 2),
        "notional":       round(notional, 2),
        "total_dollars":  round(total_dollars, 2),
    }


def plot_cost_curve(adv:        int,
                    price:      float,
                    spread:     float,
                    volatility: float = 0.20,
                    max_participation: float = 0.20) -> None:
    """
    Plot expected cost as a function of order size.

    Shows the concave shape of market impact — the key insight
    that bigger orders cost proportionally more, but with
    diminishing marginal impact due to the square root model.

    Args:
        adv:               average daily volume
        price:             current price
        spread:            bid-ask spread
        volatility:        annualized volatility
        max_participation: max order size as fraction of ADV to plot
    """
    participations = np.linspace(0.001, max_participation, 100)
    order_sizes    = participations * adv

    spread_bps  = (spread / 2) / price * 10000
    impact_bps  = [estimate_market_impact(q, adv, volatility)
                   for q in order_sizes]
    total_bps   = [spread_bps + imp for imp in impact_bps]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(participations * 100, impact_bps,
            label="Market Impact", color="tomato", linewidth=2)
    ax.axhline(spread_bps, color="steelblue", linewidth=2,
               linestyle="--", label=f"Spread Cost ({spread_bps:.1f} bps)")
    ax.plot(participations * 100, total_bps,
            label="Total Cost", color="darkgreen", linewidth=2)

    ax.set_title("Expected Execution Cost vs. Order Size\n"
                 "(Square-Root Market Impact Model)", fontsize=13)
    ax.set_xlabel("Participation Rate (% of ADV)")
    ax.set_ylabel("Cost (basis points)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tca_cost_curve.png", dpi=150)
    plt.show()
    print("Chart saved to tca_cost_curve.png")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Transaction Cost Analysis Model")
    print("="*55)

    # ── Example 1: Analyze a completed trade ──
    print("\n[1] Post-trade analysis of a completed execution:\n")

    trade = Trade(
        ticker         = "AAPL",
        side           = "buy",
        target_qty     = 50000,
        filled_qty     = 48000,
        arrival_price  = 150.00,
        avg_fill_price = 150.18,
        final_price    = 150.25,
        bid_ask_spread = 0.05,
        adv            = 2_000_000
    )

    analysis = analyze_trade(trade)
    print_tca_report(analysis)

    # ── Example 2: Pre-trade cost estimation ──
    print("[2] Pre-trade cost estimates for different order sizes:\n")

    print(f"  {'Order Size':>12} {'% ADV':>8} {'Spread':>8} "
          f"{'Impact':>8} {'Total':>8} {'$ Cost':>12}")
    print(f"  {'-'*60}")

    for qty in [10_000, 50_000, 100_000, 250_000, 500_000]:
        est = estimate_pretrade_cost(
            order_qty  = qty,
            adv        = 2_000_000,
            price      = 150.00,
            spread     = 0.05,
            volatility = 0.25
        )
        print(f"  {qty:>12,} "
              f"{est['participation']*100:>7.1f}% "
              f"{est['spread_bps']:>8.2f} "
              f"{est['impact_bps']:>8.2f} "
              f"{est['total_bps']:>8.2f} "
              f"${est['total_dollars']:>11,.0f}")

    print()

    # ── Example 3: Cost curve visualization ──
    print("[3] Generating cost curve...")
    plot_cost_curve(
        adv        = 2_000_000,
        price      = 150.00,
        spread     = 0.05,
        volatility = 0.25,
        max_participation = 0.25
    )

    print("\nTCA analysis complete.")