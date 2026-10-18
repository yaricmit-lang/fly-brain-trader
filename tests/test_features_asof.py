"""
Feature store: external data must not be known before it was published.
The daily index for the 15th is published on the 16th at 00:00, so at 14:00
on the 15th only the value for the 14th is available.
"""

import pandas as pd

from flybrain.features.external import asof_join


def _table():
    days = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame({
        "ts": days,
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "available_at": days + pd.Timedelta(days=1),
    })


def test_asof_uses_only_published_values():
    base = pd.DatetimeIndex([
        "2024-01-01 14:00", "2024-01-02 00:00", "2024-01-02 23:59", "2024-01-03 12:00",
    ], tz="UTC")
    out = asof_join(base, _table(), ["value"])
    got = out["value"].tolist()
    # 01-01 14:00: nothing published yet -> NaN
    assert pd.isna(got[0])
    # 01-02 00:00: the value for 01-01 (10) is published
    assert got[1] == 10.0
    # 01-02 23:59: still only the 01-01 value
    assert got[2] == 10.0
    # 01-03 12:00: 01-02 (20) is available, but not 01-03 (30)
    assert got[3] == 20.0


def test_asof_handles_mixed_time_resolution():
    """pandas 3: ms and us in different tables must not break the join."""
    base = pd.to_datetime([1704110400000, 1704196800000], unit="ms", utc=True)   # ms
    tbl = _table()
    tbl["available_at"] = tbl["available_at"].astype("datetime64[us, UTC]")      # us
    out = asof_join(pd.DatetimeIndex(base), tbl, ["value"])
    assert len(out) == 2
