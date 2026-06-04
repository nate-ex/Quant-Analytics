"""
lob.py — Limit Order Book Simulator
Nathan Exelbirt | Summer 2026

A simplified but structurally accurate limit order book.
Supports: add limit order, cancel order, submit market order,
match fills, and print the current book state.

This is the engine underneath every trade on every exchange.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import time


# ── Data Structures ────────────────────────────────────────────

@dataclass
class Order:
    """
    A single order in the book.

    Attributes:
        order_id:  unique identifier
        side:      'buy' or 'sell'
        price:     limit price
        quantity:  number of shares
        timestamp: time of submission (for price-time priority)
    """
    order_id:  int
    side:      str
    price:     float
    quantity:  int
    timestamp: float = field(default_factory=time.time)

    def __repr__(self):
        return (f"Order(id={self.order_id}, side={self.side}, "
                f"price={self.price}, qty={self.quantity})")


@dataclass
class Fill:
    """
    A record of a completed trade.

    Attributes:
        aggressor_id: order that triggered the fill (market or crossing limit)
        passive_id:   order that was resting in the book
        price:        price at which the fill occurred
        quantity:     number of shares filled
    """
    aggressor_id: int
    passive_id:   int
    price:        float
    quantity:     int

    def __repr__(self):
        return (f"Fill(aggressor={self.aggressor_id}, "
                f"passive={self.passive_id}, "
                f"price={self.price}, qty={self.quantity})")


# ── Limit Order Book ───────────────────────────────────────────

class LimitOrderBook:
    """
    A price-time priority limit order book for a single instrument.

    Price-time priority means:
        1. Best price gets filled first (highest bid, lowest ask)
        2. Among orders at the same price, earlier orders get filled first

    This is the standard matching rule on NYSE, Nasdaq, and most
    electronic exchanges worldwide.
    """

    def __init__(self, ticker: str):
        """
        Initialize an empty order book.

        Args:
            ticker: instrument symbol (e.g. 'AAPL')
        """
        self.ticker    = ticker
        self._bids     = {}   # price -> list of Orders, buy side
        self._asks     = {}   # price -> list of Orders, sell side
        self._orders   = {}   # order_id -> Order (for fast lookup/cancel)
        self._next_id  = 1    # auto-incrementing order ID counter
        self._fills    = []   # complete fill history

    # ── Internal helpers ───────────────────────────────────────

    def _next_order_id(self) -> int:
        """Generate the next unique order ID."""
        oid = self._next_id
        self._next_id += 1
        return oid

    def _best_bid(self) -> Optional[float]:
        """Return the highest bid price, or None if book is empty."""
        return max(self._bids.keys()) if self._bids else None

    def _best_ask(self) -> Optional[float]:
        """Return the lowest ask price, or None if book is empty."""
        return min(self._asks.keys()) if self._asks else None

    def _spread(self) -> Optional[float]:
        """Return the bid-ask spread, or None if either side is empty."""
        bb = self._best_bid()
        ba = self._best_ask()
        if bb is not None and ba is not None:
            return round(ba - bb, 4)
        return None

    def _remove_empty_levels(self, side: str, price: float) -> None:
        """Clean up price levels that have no remaining orders."""
        book = self._bids if side == "buy" else self._asks
        if price in book and len(book[price]) == 0:
            del book[price]

    # ── Public interface ───────────────────────────────────────

    def add_limit_order(self, side: str, price: float,
                        quantity: int) -> int:
        """
        Add a limit order to the book.

        The order rests at the specified price level until it is
        filled by an incoming market order or cancelled.

        Args:
            side:     'buy' or 'sell'
            price:    limit price
            quantity: number of shares

        Returns:
            order_id of the newly created order
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}.")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}.")

        order = Order(
            order_id  = self._next_order_id(),
            side      = side,
            price     = price,
            quantity  = quantity
        )

        # Add to the correct side of the book
        book = self._bids if side == "buy" else self._asks
        if price not in book:
            book[price] = []
        book[price].append(order)

        # Register for fast lookup
        self._orders[order.order_id] = order

        return order.order_id

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a resting limit order and remove it from the book.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancelled successfully, False if not found
        """
        if order_id not in self._orders:
            print(f"  Order {order_id} not found — may already be filled.")
            return False

        order = self._orders[order_id]
        book  = self._bids if order.side == "buy" else self._asks

        # Remove from price level
        if order.price in book:
            book[order.price] = [o for o in book[order.price]
                                  if o.order_id != order_id]
            self._remove_empty_levels(order.side, order.price)

        # Remove from lookup
        del self._orders[order_id]
        print(f"  Cancelled order {order_id} "
              f"({order.side} {order.quantity} @ {order.price})")
        return True

    def submit_market_order(self, side: str,
                            quantity: int) -> list[Fill]:
        """
        Submit a market order — fill immediately at best available prices.

        A market order walks through the opposite side of the book,
        consuming liquidity level by level until the full quantity
        is filled or the book is exhausted.

        Args:
            side:     'buy' (takes from ask side) or 'sell' (takes from bid side)
            quantity: number of shares to fill

        Returns:
            List of Fill objects representing completed trades
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side}.")

        fills         = []
        remaining     = quantity
        aggressor_id  = self._next_order_id()

        # Market buy consumes the ask side (lowest ask first)
        # Market sell consumes the bid side (highest bid first)
        opposite_book = self._asks if side == "buy" else self._bids
        sort_prices   = sorted(opposite_book.keys(),
                               reverse=(side == "sell"))

        for price in sort_prices:
            if remaining <= 0:
                break

            # Work through orders at this price level (time priority)
            orders_at_level = opposite_book[price]
            i = 0
            while i < len(orders_at_level) and remaining > 0:
                passive_order = orders_at_level[i]
                fill_qty      = min(remaining, passive_order.quantity)

                # Record the fill
                fill = Fill(
                    aggressor_id = aggressor_id,
                    passive_id   = passive_order.order_id,
                    price        = price,
                    quantity     = fill_qty
                )
                fills.append(fill)
                self._fills.append(fill)

                # Update quantities
                passive_order.quantity -= fill_qty
                remaining              -= fill_qty

                if passive_order.quantity == 0:
                    # Order fully filled — remove it
                    del self._orders[passive_order.order_id]
                    orders_at_level.pop(i)
                else:
                    i += 1

            self._remove_empty_levels(
                "sell" if side == "buy" else "buy", price
            )

        if remaining > 0:
            print(f"  Warning: market order partially filled. "
                  f"{remaining} shares unfilled — book exhausted.")

        return fills

    def get_book_state(self) -> dict:
        """
        Return a snapshot of the current book state.

        Returns:
            Dictionary with bid levels, ask levels, best bid,
            best ask, and spread.
        """
        bid_levels = {
            price: sum(o.quantity for o in orders)
            for price, orders in sorted(
                self._bids.items(), reverse=True)
        }
        ask_levels = {
            price: sum(o.quantity for o in orders)
            for price, orders in sorted(self._asks.items())
        }

        return {
            "ticker":    self.ticker,
            "best_bid":  self._best_bid(),
            "best_ask":  self._best_ask(),
            "spread":    self._spread(),
            "bid_levels": bid_levels,
            "ask_levels": ask_levels,
        }

    def print_book(self, depth: int = 5) -> None:
        """
        Print a formatted view of the order book.

        Shows the top N price levels on each side, with the
        spread highlighted in the middle. This is what a trader
        sees on their screen in real time.

        Args:
            depth: number of price levels to show per side
        """
        state = self.get_book_state()

        print(f"\n{'─'*40}")
        print(f"  Order Book — {self.ticker}")
        print(f"{'─'*40}")
        print(f"  {'PRICE':>10}  {'QTY':>8}  {'SIDE':<6}")
        print(f"{'─'*40}")

        # Ask side — show lowest asks at bottom (closest to mid)
        ask_levels = sorted(state["ask_levels"].items())[:depth]
        for price, qty in reversed(ask_levels):
            print(f"  {price:>10.2f}  {qty:>8}  {'ASK':<6}")

        # Spread
        spread = state["spread"]
        bb     = state["best_bid"]
        ba     = state["best_ask"]
        if spread is not None:
            mid = round((bb + ba) / 2, 4)
            print(f"{'─'*40}")
            print(f"  {'spread':>10}: {spread:.4f}   mid: {mid:.4f}")
            print(f"{'─'*40}")
        else:
            print(f"{'─'*40}  (no spread — one side empty)")

        # Bid side — show highest bids at top (closest to mid)
        bid_levels = sorted(state["bid_levels"].items(), reverse=True)[:depth]
        for price, qty in bid_levels:
            print(f"  {price:>10.2f}  {qty:>8}  {'BID':<6}")

        print(f"{'─'*40}\n")

    def print_fills(self, last_n: int = 10) -> None:
        """
        Print the most recent fills (completed trades).

        Args:
            last_n: number of recent fills to show
        """
        recent = self._fills[-last_n:]
        if not recent:
            print("  No fills yet.")
            return

        print(f"\n{'─'*40}")
        print(f"  Recent Fills — {self.ticker}")
        print(f"{'─'*40}")
        for fill in recent:
            print(f"  {fill.quantity:>6} shares @ "
                  f"{fill.price:>8.2f}  "
                  f"(aggressor: {fill.aggressor_id}, "
                  f"passive: {fill.passive_id})")
        print(f"{'─'*40}\n")


# ── Simulation ─────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*50)
    print("  Limit Order Book Simulation — AAPL")
    print("="*50)

    lob = LimitOrderBook("AAPL")

    # ── Step 1: Build an initial book ─────────────────────────
    print("\n[1] Building initial order book...")

    # Ask side — sellers
    lob.add_limit_order("sell", 150.05, 200)
    lob.add_limit_order("sell", 150.05, 150)
    lob.add_limit_order("sell", 150.10, 300)
    lob.add_limit_order("sell", 150.15, 100)
    lob.add_limit_order("sell", 150.20, 400)

    # Bid side — buyers
    lob.add_limit_order("buy",  150.00, 250)
    lob.add_limit_order("buy",  150.00, 100)
    lob.add_limit_order("buy",  149.95, 300)
    lob.add_limit_order("buy",  149.90, 200)
    lob.add_limit_order("buy",  149.85, 500)

    lob.print_book()

    # ── Step 2: Cancel an order ────────────────────────────────
    print("[2] Cancelling order ID 1 (sell 200 @ 150.05)...")
    lob.cancel_order(1)
    lob.print_book()

    # ── Step 3: Market buy order ───────────────────────────────
    print("[3] Submitting market BUY for 400 shares...")
    fills = lob.submit_market_order("buy", 400)
    print(f"  Filled {len(fills)} order(s):")
    for f in fills:
        print(f"    {f.quantity} shares @ {f.price}")
    lob.print_book()
    lob.print_fills()

    # ── Step 4: Market sell order ──────────────────────────────
    print("[4] Submitting market SELL for 200 shares...")
    fills = lob.submit_market_order("sell", 200)
    print(f"  Filled {len(fills)} order(s):")
    for f in fills:
        print(f"    {f.quantity} shares @ {f.price}")
    lob.print_book()
    lob.print_fills()

    # ── Step 5: Add more orders and observe spread change ──────
    print("[5] Adding aggressive limit orders to tighten spread...")
    lob.add_limit_order("buy",  150.03, 100)
    lob.add_limit_order("sell", 150.04, 100)
    lob.print_book()

    print("Simulation complete.")