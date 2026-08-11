import pandas as pd

import bot.taxes as tx


def test_binary_tax_uses_prior_close_as_acquisition_and_disposal_date(monkeypatch):
    idx = pd.DatetimeIndex([
        pd.Timestamp("2020-01-03", tz="UTC"),
        pd.Timestamp("2020-01-06", tz="UTC"),
        pd.Timestamp("2025-01-03", tz="UTC"),
        pd.Timestamp("2025-01-06", tz="UTC"),
    ])
    returns = pd.Series([0.0, 0.01, 0.10, -0.001], index=idx)
    held = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
    seen = []

    def capture(acquired, disposed):
        seen.append((pd.Timestamp(acquired), pd.Timestamp(disposed)))
        return 0.20

    monkeypatch.setattr(tx, "slovenia_rate_for_dates", capture)
    tx.after_tax_binary_strategy(returns, held)
    assert seen == [(idx[0], idx[2])]
