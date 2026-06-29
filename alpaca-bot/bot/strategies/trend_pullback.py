"""
bot/strategies/trend_pullback.py
================================
Long-only trend strategy with phased (pyramided) entries.

This encodes the five discretionary principles you described. Translating
discretionary rules into code always involves judgment calls, so each principle
is mapped explicitly below and every threshold is a tunable parameter.

  Principle 1 — "healthy uptrend only"
      trend_ok(): only act long while close > 50MA, close > 200MA, and
      50MA > 200MA (golden-cross alignment). This is a hard gate on every entry.

  Principle 2 — "buy the low-volume pullback to the 50MA, on the rebound"
      pullback_entry(): price has dipped to within `touch_band` of the 50MA in
      the last few bars, the pullback came on contracting volume (< vol_contraction
      x the prior baseline), and the current bar rebounds (closes up and back
      above the 50MA).

  Principle 3 — "high-level consolidation breakout on rising volume"
      breakout_entry(): the prior `consolidation_bars` formed a tight range
      (<= consolidation_range of price), and the current bar closes above that
      range high with volume >= breakout_vol_mult x the range's average volume.

  Principle 4 — "phased 30% / 30% / 40% entries"
      The position is built in three tranches. Tranche 1 opens on the first valid
      entry (pullback or breakout). Tranches 2 and 3 are added as the trend extends
      (each time price makes a new high `add_step` above the last add), while the
      trend gate still holds. This "tests the water, then adds into strength."

  Principle 5 — "hold the trend; exit on a break of the MA/structure"
      trend_exit(): exit the whole position on a daily close below the 50MA or the
      recent structural (swing) low. The protective stop is volatility-adaptive,
      stop_distance = max(5%, 2 x ATR(14) / price) below the average entry, and is
      monitored intraday by the executor (see bot/backtest_pullback.py).

  Multi-timeframe execution
      The daily chart decides whether a trade is valid (all the methods above run on
      daily bars). The intraday chart (1h/4h) only improves the entry price: an armed
      entry tries to fill on an intraday dip below the daily close, but still fills by
      the end of the session if no dip comes, so intraday never vetoes a valid daily
      setup. Intraday is also where the protective stop is checked.

The strategy methods are index-based (df, i) so the pyramiding backtester can
drive them bar by bar. They never look beyond bar `i` (no lookahead).
"""

from dataclasses import dataclass

from .. import indicators as ind


@dataclass
class PullbackParams:
    ma_fast: int = 50
    ma_slow: int = 200
    use_ema: bool = False             # "MA" -> SMA by default; flip for EMA
    # Pullback entry
    touch_band: float = 0.02          # within 2% of the 50MA counts as "at" it
    pullback_lookback: int = 5
    vol_baseline: int = 20
    vol_contraction: float = 0.90     # pullback volume < 0.90x baseline
    # Breakout entry
    consolidation_bars: int = 10
    consolidation_range: float = 0.06  # range <= 6% of price = consolidation
    breakout_vol_mult: float = 1.5
    # Phased adds
    tranches: tuple = (0.30, 0.30, 0.40)
    add_step: float = 0.03            # add next tranche every +3% new high
    # Exit / volatility stop
    structural_low_lookback: int = 10
    atr_period: int = 14              # ATR(14) for the volatility stop
    atr_mult: float = 2.0             # stop_distance = max(min_stop, atr_mult * ATR / price)
    min_stop: float = 0.05            # 5% floor
    # Multi-timeframe execution
    improve_pct: float = 0.004        # try to fill 0.4% better intraday than the daily close


class TrendPullbackStrategy:
    name = "trend_pullback"

    def __init__(self, instrument=None, params: PullbackParams = None):
        self.instrument = instrument
        self.p = params or PullbackParams()

    # -- indicators ----------------------------------------------------- #
    def _ma(self, series, n):
        return ind.ema(series, n) if self.p.use_ema else ind.sma(series, n)

    def moving_averages(self, df):
        return self._ma(df["close"], self.p.ma_fast), self._ma(df["close"], self.p.ma_slow)

    def warmup(self) -> int:
        return self.p.ma_slow + 5

    # -- Principle 1 ---------------------------------------------------- #
    def trend_ok(self, df, ma_f, ma_s, i) -> bool:
        c = df["close"].iloc[i]
        return c > ma_f.iloc[i] and c > ma_s.iloc[i] and ma_f.iloc[i] > ma_s.iloc[i]

    # -- Principle 2 ---------------------------------------------------- #
    def _pullback_entry(self, df, ma_f, i):
        p = self.p
        lb = p.pullback_lookback
        recent_low = df["low"].iloc[i - lb + 1: i + 1].min()
        near_ma = recent_low <= ma_f.iloc[i] * (1 + p.touch_band)

        pull_vol = df["volume"].iloc[i - 2: i + 1].mean()
        base_vol = df["volume"].iloc[i - p.vol_baseline: i - 2].mean()
        vol_contract = base_vol > 0 and pull_vol < base_vol * p.vol_contraction

        rebound = (df["close"].iloc[i] > df["close"].iloc[i - 1]
                   and df["close"].iloc[i] > ma_f.iloc[i]
                   and df["low"].iloc[i] >= ma_f.iloc[i] * (1 - p.touch_band))

        if near_ma and vol_contract and rebound:
            return True, "pullback to 50MA on lighter volume, rebound"
        return False, ""

    # -- Principle 3 ---------------------------------------------------- #
    def _breakout_entry(self, df, i):
        p = self.p
        c = p.consolidation_bars
        win_high = df["high"].iloc[i - c: i].max()      # prior c bars, excl. current
        win_low = df["low"].iloc[i - c: i].min()
        rng = (win_high - win_low) / df["close"].iloc[i]
        consolidated = rng <= p.consolidation_range
        broke_out = df["close"].iloc[i] > win_high
        avg_vol = df["volume"].iloc[i - c: i].mean()
        vol_ok = avg_vol > 0 and df["volume"].iloc[i] >= p.breakout_vol_mult * avg_vol

        if consolidated and broke_out and vol_ok:
            return True, "consolidation breakout on rising volume"
        return False, ""

    def entry_signal(self, df, ma_f, i):
        """Tranche-1 trigger: pullback first, else breakout."""
        ok, reason = self._pullback_entry(df, ma_f, i)
        if ok:
            return True, reason
        return self._breakout_entry(df, i)

    # -- Principle 4 ---------------------------------------------------- #
    def should_add(self, df, ma_f, i, last_add_price):
        if df["close"].iloc[i] <= ma_f.iloc[i]:
            return False, ""
        if df["close"].iloc[i] >= last_add_price * (1 + self.p.add_step):
            return True, f"trend extending (+{self.p.add_step*100:.0f}% since last add)"
        return False, ""

    # -- Volatility stop ------------------------------------------------ #
    def stop_distance(self, atr, price) -> float:
        """stop_distance = max(min_stop, atr_mult * ATR(14) / price).

        Volatility-adaptive: wider in turbulent markets, but never tighter than
        the 5% floor. Computed on the signal (daily) timeframe.
        """
        if price is None or price <= 0 or atr is None or atr != atr or atr <= 0:
            return self.p.min_stop
        return max(self.p.min_stop, self.p.atr_mult * atr / price)

    # -- Principle 5: daily trend-break exit ---------------------------- #
    def trend_exit(self, df, ma_f, i):
        """Daily-close trend break: below the 50MA or the recent structural low.

        This is a *daily* decision (the hard volatility stop is monitored
        intraday by the executor). Returns (exit_level, reason) or None.
        """
        c = df["close"].iloc[i]
        if c < ma_f.iloc[i]:
            return (c, "closed below 50MA")
        swing_low = df["low"].iloc[i - self.p.structural_low_lookback: i].min()
        if c < swing_low:
            return (c, "closed below structural low")
        return None
