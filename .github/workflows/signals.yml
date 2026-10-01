"""gold_signals.py — XAUUSD M15 SMC signal bot for Telegram (single file).

Runs on GitHub Actions (free). Every 15 minutes it checks gold, runs the same
SMC strategy as the full trading bot (trendline liquidity sweep + break &
retest), and messages you on Telegram:

    🟡 ENTER BUY/SELL  entry, SL, TP, reason
    🔒 MOVE SL TO ENTRY price reached +1R (breakeven)
    ✅/❌ TP HIT / SL HIT result in pips

You execute manually in your broker's app. No broker credentials involved.

Setup (full guide: TELEGRAM.md):
  1. GitHub repo with this file + .github/workflows/signals.yml
  2. Repo secrets: TELEGRAM_BOT_TOKEN (@BotFather), TELEGRAM_CHAT_ID,
     TWELVEDATA_API_KEY (free key from twelvedata.com)
  3. Actions tab -> enable + Run workflow

Local test:  DRY_RUN=1 python gold_signals.py
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

"""Currency pair helpers and session times (UTC)."""


import datetime as _dt

# Trading sessions as UTC hour ranges [start, end).
SESSIONS_UTC: dict[str, tuple[int, int]] = {
    "sydney": (21, 24),
    "asia": (0, 7),        # Tokyo
    "london": (7, 16),
    "newyork": (12, 21),
}


def normalize_pair(pair: str) -> str:
    """'eur_usd', 'EUR/USD', 'EURUSD=X', 'EURUSD' -> 'EURUSD'."""
    p = pair.strip().upper().replace("=X", "").replace("/", "").replace("_", "").replace("-", "")
    if len(p) != 6 or not p.isalpha():
        raise ValueError(f"cannot parse currency pair from {pair!r} (expected e.g. EUR_USD / EURUSD)")
    return p


def oanda_instrument(pair: str) -> str:
    """'EURUSD' -> 'EUR_USD' (OANDA v20 instrument naming)."""
    p = normalize_pair(pair)
    return f"{p[:3]}_{p[3:]}"


def yahoo_symbol(pair: str) -> str:
    """OANDA-style pair -> Yahoo Finance symbol.

    Yahoo has no spot gold symbol (XAUUSD=X is delisted), so XAU maps to
    GC=F (COMEX front-month futures) as a proxy: same structure/returns,
    slightly different absolute price (futures basis + contract rolls).
    """
    p = normalize_pair(pair)
    return _YAHOO_OVERRIDES.get(p, p + "=X")


_YAHOO_OVERRIDES = {
    "XAUUSD": "GC=F",   # spot gold -> COMEX front-month futures proxy
}


# Metals are quoted per troy ounce: on XAU_USD one "pip" is 0.10 USD
# (OANDA pipLocation -1, displayPrecision 3).
_METAL_PIPS = {"XAUUSD": 0.1}


def pip_size(pair: str) -> float:
    """Pip size: 0.1 for gold, 0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    p = normalize_pair(pair)
    if p in _METAL_PIPS:
        return _METAL_PIPS[p]
    return 0.01 if p[3:] == "JPY" else 0.0001


def typical_spread_pips(pair: str) -> float:
    """Round-trip retail spread assumption (pips) for cost modelling.

    Gold on a decent retail broker runs ~$0.20-0.40 round trip => 3 pips
    with pip=0.1. Majors ~1 pip. Used when config/CLI don't override.
    """
    p = normalize_pair(pair)
    if p == "XAUUSD":
        return 3.0
    return 1.0


def in_session(ts_epoch: int, sessions: list[str] | tuple[str, ...]) -> bool:
    """True if the bar time falls inside one of the named sessions (UTC).

    Weekend bars are always rejected. An empty session list = trade 24h.
    """
    if not sessions:
        return True
    dt = _dt.datetime.fromtimestamp(ts_epoch, tz=_dt.timezone.utc)
    if dt.weekday() >= 5:  # Sat/Sun
        return False
    hour = dt.hour
    for name in sessions:
        name = name.lower()
        if name not in SESSIONS_UTC:
            raise ValueError(f"unknown session {name!r}; options: {sorted(SESSIONS_UTC)}")
        lo, hi = SESSIONS_UTC[name]
        if lo <= hour < hi:
            return True
    return False


def fmt_time(ts_epoch: int) -> str:
    return _dt.datetime.fromtimestamp(ts_epoch, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

"""Candle primitives shared by every module.

The same Candle/Series objects flow through backtesting, paper trading and
live trading, so the strategy code path is identical in all three modes.
"""


from dataclasses import dataclass


@dataclass(slots=True)
class Candle:
    """One OHLC bar. `time` is the bar OPEN time as epoch seconds (UTC)."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    complete: bool = True

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Candle(t={self.time}, o={self.open:.5f}, h={self.high:.5f}, l={self.low:.5f}, c={self.close:.5f})"


class Series:
    """A rolling window of candles addressable by ABSOLUTE bar index.

    Bars are numbered 0, 1, 2, ... from the first candle the strategy ever
    sees. When old bars are trimmed, `base` advances so absolute indices of
    surviving bars never change. Every SMC component (swings, structure,
    trendlines, zones) stores absolute indices, so trimmed history never
    corrupts state.
    """

    def __init__(self, maxlen: int = 600):
        self.maxlen = maxlen
        self.data: list[Candle] = []
        self.base = 0  # absolute index of data[0]

    def append(self, c: Candle) -> None:
        self.data.append(c)
        overflow = len(self.data) - self.maxlen
        if overflow > 0:
            self.data = self.data[overflow:]
            self.base += overflow

    def __getitem__(self, abs_i: int) -> Candle:
        rel = abs_i - self.base
        if rel < 0:
            raise IndexError(f"bar {abs_i} was trimmed from the window (base={self.base})")
        if rel >= len(self.data):
            raise IndexError(f"bar {abs_i} not appended yet (last={self.last_index})")
        return self.data[rel]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def last_index(self) -> int:
        """Absolute index of the most recent bar (-1 if empty)."""
        return self.base + len(self.data) - 1

    def has(self, abs_i: int) -> bool:
        return self.base <= abs_i < self.base + len(self.data)

    def iter_range(self, i_from: int, i_to: int):
        """Yield (abs_i, candle) for i_from..i_to inclusive, clipped to window."""
        lo = max(i_from, self.base)
        hi = min(i_to, self.last_index)
        for abs_i in range(lo, hi + 1):
            yield abs_i, self.data[abs_i - self.base]


def true_range(prev_close: float | None, c: Candle) -> float:
    if prev_close is None:
        return c.high - c.low
    return max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))


class ATRTracker:
    """Incremental Wilder ATR (average true range)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.value: float | None = None
        self._seed: list[float] = []
        self._prev_close: float | None = None

    def update(self, c: Candle) -> float | None:
        tr = true_range(self._prev_close, c)
        self._prev_close = c.close
        if self.value is None:
            self._seed.append(tr)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
            return self.value
        self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value

"""Swing (pivot) detection with proper confirmation lag.

A pivot high at bar i is only *confirmed* once `right` further bars have
closed. This module is incremental: each new bar can confirm at most one
pivot, so there is no lookahead bias and no O(n^2) rescanning.
"""


from dataclasses import dataclass




@dataclass(frozen=True, slots=True)
class Swing:
    kind: str            # 'high' | 'low'
    index: int           # absolute bar index of the pivot candle
    price: float         # pivot price (high for 'high', low for 'low')
    confirm_index: int   # absolute bar index at which the pivot became knowable


class SwingDetector:
    """Incremental pivot detector over a Series.

    Strict on the left, tolerant of equal extremes on the right, so the last
    of a row of equal highs/lows becomes the pivot (equal extremes are where
    liquidity rests - we want to know about them).
    """

    def __init__(self, series: Series, left: int = 3, right: int = 3):
        self.series = series
        self.left = left
        self.right = right
        self.swings: list[Swing] = []

    def on_bar(self) -> None:
        """Call after Series.append(). Confirms at most one new pivot."""
        s = self.series
        if len(s) < self.left + self.right + 1:
            return
        cand_abs = s.last_index - self.right
        if cand_abs - self.left < s.base:
            return
        cand = s[cand_abs]
        confirm_at = s.last_index

        ok_high = True
        ok_low = True
        for k in range(1, self.left + 1):
            left = s[cand_abs - k]
            if left.high >= cand.high:
                ok_high = False
            if left.low <= cand.low:
                ok_low = False
        for k in range(1, self.right + 1):
            right = s[cand_abs + k]
            if right.high > cand.high:   # equal highs allowed (liquidity)
                ok_high = False
            if right.low < cand.low:     # equal lows allowed
                ok_low = False

        if ok_high:
            self.swings.append(Swing("high", cand_abs, cand.high, confirm_at))
        if ok_low:
            self.swings.append(Swing("low", cand_abs, cand.low, confirm_at))

    def confirmed(self, kind: str, as_of_bar: int) -> list[Swing]:
        """Swings of `kind` knowable as of `as_of_bar` (no lookahead)."""
        return [sw for sw in self.swings if sw.kind == kind and sw.confirm_index <= as_of_bar]

    def last(self, kind: str, as_of_bar: int) -> Swing | None:
        sws = self.confirmed(kind, as_of_bar)
        return sws[-1] if sws else None

"""Market structure: break of structure (BOS) and change of character (CHoCH).

Tracks the most recent confirmed swing high / swing low. When a candle
CLOSES beyond the most recent swing, that's a structural break:

- same direction as current bias  -> BOS (continuation)
- against current bias            -> CHoCH (potential reversal)

The strategy uses these events as the "break" leg of the setup, and `bias`
to decide which side of the market to hunt liquidity on.
"""


from dataclasses import dataclass





@dataclass(frozen=True, slots=True)
class StructureEvent:
    kind: str          # 'BOS' | 'CHoCH'
    direction: str     # 'bull' | 'bear'
    level: float       # price of the broken swing (the retest level)
    bar_index: int     # absolute index of the candle that closed through it
    swing_index: int   # absolute index of the broken swing


class StructureAnalyzer:
    """Incremental structure tracker. Call update() after each new bar."""

    def __init__(self, series: Series, detector: SwingDetector):
        self.series = series
        self.det = detector
        self.bias: str | None = None   # 'bull' | 'bear' | None (undetermined)
        self.events: list[StructureEvent] = []
        self._consumed_high_idx = -1    # absolute index of last consumed swing high
        self._consumed_low_idx = -1
        self._processed = -1            # last processed absolute bar index

    def _active(self, kind: str, as_of: int, after_idx: int) -> Swing | None:
        cands = [sw for sw in self.det.swings
                 if sw.kind == kind and sw.confirm_index <= as_of and sw.index > after_idx]
        return cands[-1] if cands else None

    def update(self) -> list[StructureEvent]:
        """Process any bars not yet handled; return events from the newest bar(s)."""
        fired: list[StructureEvent] = []
        for abs_i in range(self._processed + 1, self.series.last_index + 1):
            close = self.series[abs_i].close

            # --- bullish break: close above the most recent confirmed swing high
            sh = self._active("high", abs_i, self._consumed_high_idx)
            if sh is not None and close > sh.price:
                kind = "CHoCH" if self.bias == "bear" else "BOS"
                ev = StructureEvent(kind, "bull", sh.price, abs_i, sh.index)
                self.events.append(ev)
                fired.append(ev)
                self.bias = "bull"
                self._consumed_high_idx = sh.index

            # --- bearish break: close below the most recent confirmed swing low
            sl = self._active("low", abs_i, self._consumed_low_idx)
            if sl is not None and close < sl.price:
                kind = "CHoCH" if self.bias == "bull" else "BOS"
                ev = StructureEvent(kind, "bear", sl.price, abs_i, sl.index)
                self.events.append(ev)
                fired.append(ev)
                self.bias = "bear"
                self._consumed_low_idx = sl.index

            self._processed = abs_i
        return fired

"""Fair value gaps (FVG / imbalance).

A bullish FVG exists when candle i's LOW is above candle i-2's HIGH - price
moved so fast one candle that it left a gap. Price often returns to fill
part of it, which makes FVGs natural retest zones after a break.
"""


from dataclasses import dataclass




@dataclass(frozen=True, slots=True)
class FVG:
    direction: str   # 'bull' | 'bear'
    top: float
    bottom: float
    index: int       # absolute index of the third candle of the pattern


def find_fvgs(series: Series, i_from: int, i_to: int) -> list[FVG]:
    """All FVGs created by candles in [i_from, i_to] (inclusive)."""
    out: list[FVG] = []
    for i in range(max(i_from, series.base + 2), i_to + 1):
        if not series.has(i):
            continue
        c = series[i]
        c2 = series[i - 2]
        if c.low > c2.high:
            out.append(FVG("bull", top=c.low, bottom=c2.high, index=i))
        elif c.high < c2.low:
            out.append(FVG("bear", top=c2.low, bottom=c.high, index=i))
    return out

"""Order blocks (simplified).

The bullish order block for an up-move is the last DOWN candle before the
impulse that broke structure - the zone where institutions filled longs.
Its range is a valid retest zone for break & retest entries.
"""


from dataclasses import dataclass




@dataclass(frozen=True, slots=True)
class OrderBlock:
    direction: str   # 'bull' | 'bear' (direction of the impulse it precedes)
    top: float
    bottom: float
    index: int


def order_block_before(series: Series, i_from: int, i_break: int,
                       direction: str) -> OrderBlock | None:
    """Last opposite-colour candle in (i_from, i_break] before the breaking leg."""
    for i in range(i_break, max(i_from, series.base), -1):
        c = series[i]
        if direction == "bull" and c.close < c.open:
            return OrderBlock("bull", top=c.high, bottom=c.low, index=i)
        if direction == "bear" and c.close > c.open:
            return OrderBlock("bear", top=c.high, bottom=c.low, index=i)
    return None

"""Trendlines and trendline liquidity sweeps.

In an uptrend the bot connects the two most recent ascending swing lows
(vice versa in a downtrend) and extends the line to the right. Stops from
traders who bought the trendline rest just below it -> that is *trendline
liquidity*.

A sweep is a candle that WICKS through the line but CLOSES back on the
right side of it: liquidity taken, trendline still intact. That is the
manipulation leg of the setup.
"""


from dataclasses import dataclass





@dataclass(frozen=True, slots=True)
class Trendline:
    kind: str          # 'support' (rising, under lows) | 'resistance' (falling, over highs)
    i1: int
    p1: float
    i2: int
    p2: float

    def value_at(self, i: int) -> float:
        if self.i2 == self.i1:
            return self.p2
        return self.p1 + (self.p2 - self.p1) * (i - self.i1) / (self.i2 - self.i1)


def find_trendline(swings: list[Swing], kind: str, series: Series, upto_bar: int,
                   max_pairs: int = 4) -> Trendline | None:
    """Best trendline built from the most recent swing points.

    `kind='support'`: last two swing lows forming an ASCENDING line, with no
    candle CLOSE below it since the second anchor. 'resistance' mirrors.
    """
    pts = [sw for sw in swings if sw.kind == ("low" if kind == "support" else "high")
           and sw.index < upto_bar]
    tried = 0
    for j in range(len(pts) - 1, 0, -1):
        a, b = pts[j - 1], pts[j]
        if tried >= max_pairs:
            break
        tried += 1
        if kind == "support":
            if b.price <= a.price:
                continue                      # must ascend
            tl = Trendline("support", a.index, a.price, b.index, b.price)
            violated = any(c.close < tl.value_at(i)
                           for i, c in series.iter_range(b.index, upto_bar))
        else:
            if b.price >= a.price:
                continue                      # must descend
            tl = Trendline("resistance", a.index, a.price, b.index, b.price)
            violated = any(c.close > tl.value_at(i)
                           for i, c in series.iter_range(b.index, upto_bar))
        if not violated:
            return tl
    return None


@dataclass(frozen=True, slots=True)
class Sweep:
    direction: str     # 'bull' (sell-side liquidity under support grabbed) | 'bear'
    bar_index: int
    extreme: float     # the sweep wick extreme (low for bull, high for bear)
    line_value: float  # trendline value at the sweep bar


def sweep_at(series: Series, tl: Trendline, i: int) -> Sweep | None:
    """Sweep check for candle i against trendline `tl`."""
    c = series[i]
    v = tl.value_at(i)
    if tl.kind == "support" and c.low < v and c.close > v:
        return Sweep("bull", i, c.low, v)
    if tl.kind == "resistance" and c.high > v and c.close < v:
        return Sweep("bear", i, c.high, v)
    return None

"""Strategy: trendline liquidity sweep -> break of structure -> break & retest.

State machine (per instrument):

  HUNT        bias is known; watch the active trendline for a liquidity sweep
  WAIT_BOS    sweep happened; wait for a CLOSE through the last swing
              (break of structure) within `sweep_bos_window` bars
  WAIT_RETEST break happened; wait for price to return to the broken level /
              order block / FVG zone and reclaim it -> enter with the break
  COOLDOWN    position open; ignore new setups until it closes

Stop loss sits beyond the sweep wick (the liquidity extreme), take profit is
a fixed R multiple or the next opposing swing (configurable). This exact
class runs in the backtester, the paper broker and live trading.
"""



from dataclasses import dataclass, field
from enum import Enum









log = logging.getLogger("smc.strategy")


# --------------------------------------------------------------------------- params

@dataclass(slots=True)
class StrategyParams:
    # swing detection
    pivot_left: int = 3
    pivot_right: int = 3
    # ATR
    atr_period: int = 14
    # sweep -> break window (bars)
    sweep_bos_window: int = 15
    # break -> retest window (bars)
    retest_window: int = 24
    # how close to the level counts as a touch (in ATRs above the level)
    touch_tol_atr: float = 0.20
    # stop buffer beyond the sweep extreme, in ATRs
    sl_buffer_atr: float = 0.15
    # skip setups whose stop distance is outside [min, max] ATRs
    min_sl_atr: float = 0.35
    max_sl_atr: float = 4.00
    # take profit
    rr: float = 2.0                    # take-profit R multiple (tp_mode='rr')
    tp_mode: str = "rr"                # 'rr' | 'structure'
    # move stop to entry once price travels this many R (0 = off)
    breakeven_at_r: float = 1.0
    # confirmation candle must close in the trade direction
    require_reclaim_candle: bool = True
    # session filter for ENTRIES: ['london', 'newyork'] etc. [] = 24h
    sessions: list[str] = field(default_factory=lambda: ["london", "newyork"])
    # how many candidate swing pairs to try when building a trendline
    trendline_max_pairs: int = 4
    # rolling candle window size
    window: int = 600


# --------------------------------------------------------------------------- signal

@dataclass(slots=True)
class Signal:
    direction: str        # 'long' | 'short'
    entry: float          # reference entry (close of confirmation candle)
    sl: float
    tp: float
    time: int             # confirmation candle time (epoch, UTC)
    bar_index: int
    reason: str
    meta: dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def r_multiple(self) -> float:
        return abs(self.tp - self.entry) / self.risk if self.risk else 0.0


class StrategyState(str, Enum):
    HUNT = "hunt"
    WAIT_BOS = "wait_bos"
    WAIT_RETEST = "wait_retest"
    COOLDOWN = "cooldown"


# --------------------------------------------------------------------------- strategy

class TrendlineBreakRetest:
    """SMC trendline-liquidity + break & retest strategy (long and short)."""

    def __init__(self, params: StrategyParams | None = None):
        self.p = params or StrategyParams()
        self.series = Series(maxlen=self.p.window)
        self.swings = SwingDetector(self.series, self.p.pivot_left, self.p.pivot_right)
        self.structure = StructureAnalyzer(self.series, self.swings)
        self.atr = ATRTracker(self.p.atr_period)
        self.state = StrategyState.HUNT
        self.setup: dict = {}
        self._next_index = 0   # absolute index of the next candle to arrive
        self.last_signal: Signal | None = None

    # ------------------------------------------------------------- public API

    def on_candle(self, c: Candle) -> Signal | None:
        """Feed one COMPLETED candle; returns a Signal if this bar triggers entry."""
        i = self._next_index
        self._next_index += 1
        self.series.append(c)
        atr = self.atr.update(c) or 0.0
        self.swings.on_bar()
        events = self.structure.update()

        if self.state is StrategyState.HUNT:
            self._hunt(i, c, atr)
        elif self.state is StrategyState.WAIT_BOS:
            self._wait_bos(i, c, atr, events)
        elif self.state is StrategyState.WAIT_RETEST:
            return self._wait_retest(i, c, atr)
        elif self.state is StrategyState.COOLDOWN:
            pass
        return None

    def on_position_closed(self) -> None:
        """Called by the engine/backtester when our position is flat again."""
        if self.state is StrategyState.COOLDOWN:
            self.state = StrategyState.HUNT
        self.setup = {}

    # ------------------------------------------------------------- state logic

    def _hunt(self, i: int, c: Candle, atr: float) -> None:
        bias = self.structure.bias
        if bias not in ("bull", "bear") or atr <= 0:
            return
        kind = "support" if bias == "bull" else "resistance"
        tl = find_trendline(self.swings.confirmed("low" if bias == "bull" else "high", i),
                            kind, self.series, i, self.p.trendline_max_pairs)
        if tl is None:
            return
        sw = sweep_at(self.series, tl, i)
        if sw is None:
            return
        if sw.direction != bias:      # safety; sweep direction always matches line kind
            return
        self.setup = {
            "dir": sw.direction,
            "sweep_bar": sw.bar_index,
            "sweep_extreme": sw.extreme,
            "sweep_line_value": sw.line_value,
            "trendline": tl,
            "deadline": i + self.p.sweep_bos_window,
            "atr_at_sweep": atr,
        }
        self.state = StrategyState.WAIT_BOS
        log.info("[%s] liquidity sweep @ %s (wick %.5f through %s trendline %.5f, closed back inside)",
                 sw.direction, self._t(c), sw.extreme, kind, sw.line_value)

    def _wait_bos(self, i: int, c: Candle, atr: float, events: list[StructureEvent]) -> None:
        d = self.setup["dir"]

        # hard invalidation FIRST (against the original sweep extreme): a close
        # through it means the trendline genuinely broke - that is not a sweep
        if (d == "bull" and c.close < self.setup["sweep_extreme"]) or \
           (d == "bear" and c.close > self.setup["sweep_extreme"]):
            log.info("[%s] setup invalidated: close %.5f through sweep extreme %.5f",
                     d, c.close, self.setup["sweep_extreme"])
            self._reset()
            return

        # a further wick beyond the sweep extreme while still closing back
        # inside = more liquidity grabbed; extend the sweep extreme
        if d == "bull" and c.low < self.setup["sweep_extreme"]:
            self.setup["sweep_extreme"] = c.low
        if d == "bear" and c.high > self.setup["sweep_extreme"]:
            self.setup["sweep_extreme"] = c.high

        if i > self.setup["deadline"]:
            self._reset()
            return

        ev = next((e for e in events if e.direction == d), None)
        if ev is None:
            return

        level = ev.level
        buf = self.p.sl_buffer_atr * atr
        sl = (self.setup["sweep_extreme"] - buf) if d == "bull" else (self.setup["sweep_extreme"] + buf)
        ob = order_block_before(self.series, self.setup["sweep_bar"], ev.bar_index, d)
        fvgs = [f for f in find_fvgs(self.series, self.setup["sweep_bar"], ev.bar_index)
                if f.direction == d]

        extremes = [self.setup["sweep_extreme"]]
        if ob:
            extremes.append(ob.bottom if d == "bull" else ob.top)
        extremes += [(f.bottom if d == "bull" else f.top) for f in fvgs]
        # bull: deepest valid pullback (a close below kills the setup)
        # bear: highest valid rally    (a close above kills the setup)
        invalid_price = min(extremes) if d == "bull" else max(extremes)
        # how far price must return toward the broken level to count as a touch
        touch_price = (level + self.p.touch_tol_atr * atr) if d == "bull" \
            else (level - self.p.touch_tol_atr * atr)

        self.setup.update({
            "level": level,
            "sl": sl,
            "ob": ob,
            "fvgs": fvgs,
            "invalid_price": invalid_price,
            "touch_price": touch_price,
            "bos_bar": ev.bar_index,
            "bos_kind": ev.kind,
            "deadline": i + self.p.retest_window,
            "touched": False,
        })
        self.state = StrategyState.WAIT_RETEST
        log.info("[%s] %s confirmed @ %.5f close %.5f - waiting for retest "
                 "(touch <= %.5f | invalid beyond %.5f | SL %.5f)",
                 d, ev.kind, level, c.close, touch_price, invalid_price, sl)

    def _wait_retest(self, i: int, c: Candle, atr: float) -> Signal | None:
        d = self.setup["dir"]
        level = self.setup["level"]

        # invalidation: liquidity beyond the sweep extreme taken, or a close
        # through the far side of the retest zone = failed break
        if (d == "bull" and (c.low < self.setup["sweep_extreme"]
                             or c.close < self.setup["invalid_price"])) or \
           (d == "bear" and (c.high > self.setup["sweep_extreme"]
                             or c.close > self.setup["invalid_price"])):
            log.info("[%s] retest failed: liquidity beyond sweep taken / close %.5f "
                     "through invalidation level %.5f",
                     d, c.close, self.setup["invalid_price"])
            self._reset()
            return None
        if i > self.setup["deadline"]:
            log.info("[%s] retest window expired without entry", d)
            self._reset()
            return None

        # touch: price returns to the broken level
        if d == "bull" and c.low <= self.setup["touch_price"]:
            self.setup["touched"] = True
        if d == "bear" and c.high >= self.setup["touch_price"]:
            self.setup["touched"] = True

        if not self.setup["touched"]:
            return None

        # confirmation: candle reclaims the broken level
        if d == "bull":
            ok = c.close >= level and (c.bullish or not self.p.require_reclaim_candle)
        else:
            ok = c.close <= level and ((not c.bullish) or not self.p.require_reclaim_candle)
        if not ok:
            return None

        if not in_session(c.time, self.p.sessions):
            log.info("[%s] signal outside session filter (%s) - not entering",
                     d, ",".join(self.p.sessions) or "24h")
            return None

        entry = c.close
        sl = self.setup["sl"]
        risk = abs(entry - sl)
        if not (self.p.min_sl_atr * atr <= risk <= self.p.max_sl_atr * atr):
            log.info("[%s] entry skipped: SL distance %.5f outside %.2f..%.2f ATR band",
                     d, risk, self.p.min_sl_atr, self.p.max_sl_atr)
            self._reset()
            return None

        tp = self._take_profit(entry, sl, d, i, risk)
        sig = Signal(
            direction="long" if d == "bull" else "short",
            entry=entry, sl=sl, tp=tp,
            time=c.time, bar_index=i,
            reason=(f"{self.setup['bos_kind']} @ {level:.5f} after trendline sweep "
                    f"@ {self.setup['sweep_extreme']:.5f}; break & retest"),
            meta={
                "level": level,
                "sweep_extreme": self.setup["sweep_extreme"],
                "atr": atr,
                "rr": self.p.rr,
                "breakeven_at_r": self.p.breakeven_at_r,
            },
        )
        self.last_signal = sig
        self.state = StrategyState.COOLDOWN
        log.info("[SIGNAL %s] entry %.5f | SL %.5f | TP %.5f | %.1fR (%s)",
                 sig.direction, entry, sl, tp, sig.r_multiple, sig.reason)
        return sig

    # ------------------------------------------------------------- helpers

    def _take_profit(self, entry: float, sl: float, d: str, i: int, risk: float) -> float:
        if self.p.tp_mode == "structure":
            kind = "high" if d == "bull" else "low"
            cands = [sw.price for sw in self.swings.confirmed(kind, i)
                     if (sw.price > entry + risk if d == "bull" else sw.price < entry - risk)]
            if cands:
                return min(cands) if d == "bull" else max(cands)
        if d == "bull":
            return entry + self.p.rr * risk
        return entry - self.p.rr * risk

    def _reset(self) -> None:
        self.setup = {}
        self.state = StrategyState.HUNT

    @staticmethod
    def _t(c: Candle) -> str:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(c.time, tz=_dt.timezone.utc).strftime("%m-%d %H:%M")


PAIR = "XAUUSD"
TF_MINUTES = 15
API = "https://api.telegram.org/bot{token}/{method}"
STATE_FILE = Path(__file__).resolve().parent / "state.json"

# ------------------------------------------------------------------ telegram
def tg(method: str, token: str, **params) -> dict:
    r = requests.post(API.format(token=token, method=method), json=params,
                      timeout=20)
    r.raise_for_status()
    return r.json()


def send(text: str, token: str, chat_id: str, dry: bool) -> None:
    if dry:
        print(f"[dry-run] TO TELEGRAM:\n{text}\n", flush=True)
        return
    tg("sendMessage", token, chat_id=chat_id, text=text,
       parse_mode="HTML", disable_web_page_preview=True)
    print("sent:", text.splitlines()[0], flush=True)


def resolve_chat_id(token: str) -> str | None:
    """If TELEGRAM_CHAT_ID is unset, read the latest message sent to the bot."""
    data = tg("getUpdates", token, limit=1, allowed_updates=["message"])
    for upd in data.get("result", []):
        chat = upd.get("message", {}).get("chat", {})
        if chat.get("id"):
            return str(chat["id"])
    return None


# -------------------------------------------------------------------- feeds

def fetch_twelvedata(key: str, bars: int) -> list[Candle]:
    import datetime as _dt
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": "XAU/USD", "interval": "15min", "outputsize": str(bars),
        "apikey": key, "format": "JSON", "timezone": "UTC",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    values = data.get("values") or []
    if data.get("status") == "error" or not values:
        raise RuntimeError(f"twelvedata error: {data.get('message') or data}")
    out = []
    for v in values:                      # newest first -> reverse
        t = _dt.datetime.fromisoformat(v["datetime"]).replace(tzinfo=_dt.timezone.utc)
        out.append(Candle(
            time=int(t.timestamp()),
            open=float(v["open"]), high=float(v["high"]),
            low=float(v["low"]), close=float(v["close"]),
            volume=float(v.get("volume") or 0), complete=True))
    out.reverse()
    return out


def fetch_yahoo(bars: int) -> list[Candle]:
    raise RuntimeError("yahoo fallback needs the full repo - set "
                       "TWELVEDATA_API_KEY (free at twelvedata.com)")


def fetch_candles(bars: int) -> list[Candle]:
    key = os.environ.get("TWELVEDATA_API_KEY", "")
    feed = os.environ.get("FEED", "").lower()
    if feed == "yahoo" or not key:
        return fetch_yahoo(bars)
    return fetch_twelvedata(key, bars)


# -------------------------------------------------------------------- state

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_signal_bar": 0, "active": None}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1) + "\n")


# ------------------------------------------------------------------- alerts

def enter_message(sig) -> str:
    sl_pips = abs(sig.entry - sig.sl) * 10
    return (f"🟡 <b>{PAIR} M15 — {sig.direction.upper()} signal</b>\n"
            f"Entry: <b>{sig.entry:.2f}</b>\n"
            f"SL: {sig.sl:.2f}  ({sl_pips:.0f} pips)\n"
            f"TP: {sig.tp:.2f}  ({sig.r_multiple:.1f}R)\n"
            f"{sig.reason}\n\n"
            f"Size the trade so a SL hit ≈ your daily risk budget.")


def check(candles: list[Candle], state: dict, token: str, chat_id: str,
          dry: bool) -> None:
    strat = TrendlineBreakRetest(StrategyParams())
    last = candles[-1]
    sig = None
    for c in candles:
        s = strat.on_candle(c)
        if s is None:
            continue
        if c.time == last.time:
            sig = s                    # actionable - newest completed bar
        elif strat.state is StrategyState.COOLDOWN:
            strat.on_position_closed()  # historical signal: release and keep hunting
    if sig is not None and last.time != state.get("last_signal_bar"):
        if state.get("active"):
            send(f"🟡 <b>new {sig.direction.upper()} signal on {PAIR}</b> skipped — "
                 f"previous trade still open (one at a time). "
                 f"It was entry {sig.entry:.2f}, SL {sig.sl:.2f}, TP {sig.tp:.2f}.",
                 token, chat_id, dry)
        else:
            send(enter_message(sig), token, chat_id, dry)
            state["active"] = {
                "dir": sig.direction, "entry": sig.entry, "sl": sig.sl,
                "tp": sig.tp, "time": last.time, "be_alerted": False,
            }
        state["last_signal_bar"] = last.time
        save_state(state)
        return

    # manage the active position: breakeven at +1R, exit on TP/SL touch
    a = state.get("active")
    if not a:
        save_state(state)
        return
    signed = 1 if a["dir"] == "long" else -1
    moved = signed * (last.close - a["entry"])
    risk = abs(a["entry"] - a["sl"])
    hit_tp = (last.high >= a["tp"]) if signed > 0 else (last.low <= a["tp"])
    hit_sl = (last.low <= a["sl"]) if signed > 0 else (last.high >= a["sl"])
    if hit_tp or hit_sl:
        which, emoji = ("TP", "✅") if hit_tp else ("SL", "❌")
        pips = moved * 10
        send(f"{emoji} <b>{PAIR} {which} hit</b> "
             f"({a['dir'].upper()} from {a['entry']:.2f})\n"
             f"Result ≈ {pips:+.0f} pips", token, chat_id, dry)
        state["active"] = None
    elif not a.get("be_alerted") and risk and moved >= risk:
        send(f"🔒 <b>{PAIR} at +1R</b> — move your SL to entry "
             f"({a['entry']:.2f}) for a free trade", token, chat_id, dry)
        a["be_alerted"] = True
    save_state(state)


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    dry = os.environ.get("DRY_RUN") == "1"
    if not token and not dry:
        print("TELEGRAM_BOT_TOKEN not set")
        return 1
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not chat_id and not dry:
        chat_id = resolve_chat_id(token or "dry")
        if chat_id:
            send(f"Your chat id is <code>{chat_id}</code> — save it as the "
                 f"TELEGRAM_CHAT_ID secret and re-run.", token, chat_id, dry)
            return 0
        print("No chat id found — send /start to your bot first, then re-run.")
        return 1

    candles = fetch_candles(600)
    # drop the still-forming candle: keep only bars that closed
    now = time.time()
    closed = [c for c in candles if c.time + TF_MINUTES * 60 <= now]
    if len(closed) < 60:
        print(f"only {len(closed)} closed candles — not enough history yet")
        return 1
    check(closed, load_state(), token or "dry", chat_id or "dry", dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
