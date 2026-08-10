# Reproducibility protocol

Historical strategy output is only meaningful when the code, environment, data and
assumptions that produced it can be identified. Do not treat a number copied into a
README, spreadsheet or chat as a durable benchmark by itself.

## Minimum record for a research run

Save all of the following together with the result:

```bash
git rev-parse HEAD
python --version
python -m pip freeze > pip-freeze.txt
```

Also record:

- command line used;
- exact requested symbol universe;
- start/end dates;
- data source and download/as-of date;
- any suspect/quarantined data timestamps reported by `bot.data`;
- slippage/transaction-cost assumption;
- borrow, hedge, cash/risk-free assumptions when applicable;
- currency scenario;
- tax scenario and the date its legal assumptions were checked;
- whether the result is full-sample, chronological consistency, or true parameter
  fit/test walk-forward;
- generated CSV/PNG outputs that materially support the conclusion.

Yahoo history can change because vendors correct corporate actions and historical
records. A later download is not guaranteed to be byte-for-byte identical to an old
one. For a result that matters, archive the permitted source data or a checksum of a
lawfully retained data extract in addition to the download date.

## Dependency policy

Runtime requirement files use lower and major-version upper bounds so an install
cannot silently cross a completely untested major release. They are not a full
transitive lock. For an exact historical reproduction, use the saved `pip-freeze`
from that run in a fresh virtual environment.

CI currently exercises `alpaca-bot` on Python 3.11 and 3.12 and `gex-lab` on Python
3.12. A deployment on a different interpreter should be treated as unverified until
CI explicitly covers it.

## Live deployment record

For a live or paper deployment, additionally record:

- broker/account mode (paper/live); never store credentials in the record;
- configured symbol list;
- strategy state filename;
- whether `--adopt-existing` was used and which broker positions were deliberately
  handed to the strategy;
- service-unit version and environment-file path;
- Git commit deployed;
- test/CI status for that commit.

Update by stopping the service, fast-forwarding the real Git clone, reinstalling any
changed dependencies, running tests, and then restarting. Do not pull new source
underneath a running trading process.

## GEX snapshots

GEX snapshots are point-in-time research data. Keep the original timestamped files.
Do not rename a later snapshot to an earlier date. The backtester deliberately makes
a snapshot available only to later sessions; changing its timestamp changes the
causal information set.

The snapshot CSV records the risk-free rate, dividend yield and dealer-sign model
used by the free-data approximation. Preserve those fields with any exported result.

## Tax research

Tax code in this repository is scenario modeling, not filing software. Save the date
on which legal assumptions were independently checked. Laws and administrative
guidance can change while old backtest output remains on disk.
