"""
bot/strategies/trend_pullback.py
================================
Long-only trend strategy with phased entries.

Signals are evaluated only from bars at or before the supplied index. The stop
distance is volatility-derived when a position is initiated; executors keep that
actual position stop distance fixed while adding tranches so sizing and protection
refer to the same risk distance.
"""

from dataclasses import dataclass

from .. import indicators as ind


@dataclass
class PullbackParams:
    ma_fast: int = 50
    ma_slow: int = 200
    use_ema: bool = False
    touch_band: float = 0.02
    pullback_lookback: int = 5
    vol_baseline: int = 20
    vol_contraction: float = 0.90
    consolidation_bars: int = 10
    consolidation_range: float = 0.06
    breakout_vol_mult: float = 1.5
    tranches: tuple = (0.30, 0.30, 0.40)
    add_step: float = 0.03
    structural_low_lookback: int = 10
    atr_period: int = 14
    atr_mult: float = 2.0
    min_stop: float = 0.05
    improve_pct: float = 0.004

    def __post_init__(self):
        ints = {
            "ma_fast": self.ma_fast, "ma_slow": self.ma_slow,
            "pullback_lookback": self.pullback_lookback,
            "vol_baseline": self.vol_baseline,
            "consolidation_bars": self.consolidation_bars,
            "structural_low_lookback": self.structural_low_lookback,
            "atr_period": self.atr_period,
        }
        bad_ints = [k for k, v in ints.items() if not isinstance(v, int) or v <= 0]
        if bad_ints:
            raise ValueError("Positive integer required for: " + ", ".join(bad_ints))
        if self.ma_fast >= self.ma_slow:
            raise ValueError("ma_fast must be smaller than ma_slow")
        for name, value in {
            "touch_band": self.touch_band, "consolidation_range": self.consolidation_range,
            "add_step": self.add_step, "atr_mult": self.atr_mult,
            "min_stop": self.min_stop, "improve_pct": self.improve_pct,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 < self.vol_contraction:
            raise ValueError("vol_contraction must be positive")
        if self.breakout_vol_mult <= 0:
            raise ValueError("breakout_vol_mult must be positive")
        if not self.tranches or any(x <= 0 for x in self.tranches):
            raise ValueError("tranches must contain positive fractions")
        if abs(sum(self.tranches) - 1.0) > 1e-9:
            raise ValueError("tranches must sum to 1.0")


class TrendPullbackStrategy:
    name = "trend_pullback"

    def __init__(self, instrument=None, params=None):
        self.instrument = instrument
        self.p = params or PullbackParams()

    def _ma(self, series, n):
        return ind.ema(series, n) if self.p.use_ema else ind.sma(series, n)

    def moving_averages(self, df):
        return self._ma(df["close"], self.p.ma_fast), self._ma(df["close"], self.p.ma_slow)

    def warmup(self) -> int:
        p = self.p
        return max(p.ma_slow, p.vol_baseline + 3, p.consolidation_bars + 1,
                   p.structural_low_lookback + 1, p.atr_period + 1) + 5

    def trend_ok(self, df, ma_f, ma_s, i) -> bool:
        c = df["close"].iloc[i]
        return c > ma_f.iloc[i] and c > ma_s.iloc[i] and ma_f.iloc[i] > ma_s.iloc[i]

    def _pullback_entry(self, df, ma_f, i):
        p = self.p
        lb = p.pullback_lookback
        recent_low = df["low"].iloc[i - lb + 1:i + 1].min()
        ma_now = ma_f.iloc[i]
        # "Within touch_band" is a two-sided band. The old one-sided comparison
        # also accepted crashes far below the MA as valid touches.
        near_ma = ma_now * (1 - p.touch_band) <= recent_low <= ma_now * (1 + p.touch_band)

        pull_vol = df["volume"].iloc[i - 2:i + 1].mean()
        base_end = i - 2
        base_start = base_end - p.vol_baseline
        base_vol = df["volume"].iloc[base_start:base_end].mean()
        vol_contract = base_vol > 0 and pull_vol < base_vol * p.vol_contraction

        rebound = (df["close"].iloc[i] > df["close"].iloc[i - 1]
                   and df["close"].iloc[i] > ma_now
                   and df["low"].iloc[i] >= ma_now * (1 - p.touch_band))
        if near_ma and vol_contract and rebound:
            return True, "pullback to fast MA on lighter volume, rebound"
        return False, ""

    def _breakout_entry(self, df, i):
        p = self.p
        c = p.consolidation_bars
        win_high = df["high"].iloc[i - c:i].max()
        win_low = df["low"].iloc[i - c:i].min()
        close = df["close"].iloc[i]
        rng = (win_high - win_low) / close if close > 0 else float("inf")
        consolidated = rng <= p.consolidation_range
        broke_out = close > win_high
        avg_vol = df["volume"].iloc[i - c:i].mean()
        vol_ok = avg_vol > 0 and df["volume"].iloc[i] >= p.breakout_vol_mult * avg_vol
        if consolidated and broke_out and vol_ok:
            return True, "consolidation breakout on rising volume"
        return False, ""

    def entry_signal(self, df, ma_f, i):
        ok, reason = self._pullback_entry(df, ma_f, i)
        if ok:
            return True, reason
        return self._breakout_entry(df, i)

    def should_add(self, df, ma_f, i, last_add_price):
        if last_add_price is None or last_add_price <= 0:
            return False, ""
        if df["close"].iloc[i] <= ma_f.iloc[i]:
            return False, ""
        if df["close"].iloc[i] >= last_add_price * (1 + self.p.add_step):
            return True, f"trend extending (+{self.p.add_step*100:.0f}% since last add)"
        return False, ""

    def stop_distance(self, atr, price) -> float:
        if price is None or price <= 0 or atr is None or atr != atr or atr <= 0:
            return self.p.min_stop
        return max(self.p.min_stop, self.p.atr_mult * atr / price)

    def trend_exit(self, df, ma_f, i):
        c = df["close"].iloc[i]
        if c < ma_f.iloc[i]:
            return (c, "closed below fast MA")
        swing_low = df["low"].iloc[i - self.p.structural_low_lookback:i].min()
        if c < swing_low:
            return (c, "closed below structural low")
        return None
