"""
Codex provides coordinates as a single string such as "[352484 175164  55880]":
numbers in square brackets separated by variable-width spaces (alignment
padding). There are no plain x, y, z columns. The format is unusual and had to
be parsed by hand once already, so we pin it with a test to keep the next
dataset update from silently breaking the parsing.
"""

import polars as pl


def parse_position_column(coords: pl.DataFrame, root_col: str) -> pl.DataFrame:
    """Same logic as in connectome/loader.py, replicated here to test it in isolation."""
    parsed = coords.select([
        pl.col(root_col).cast(pl.UInt64).alias("root_id"),
        pl.col("position").str.strip_chars("[]").alias("_pos"),
    ]).with_columns(
        pl.col("_pos").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "")
        ).alias("_parts")
    )
    return parsed.select([
        pl.col("root_id"),
        pl.col("_parts").list.get(0).cast(pl.Float32, strict=False).alias("x"),
        pl.col("_parts").list.get(1).cast(pl.Float32, strict=False).alias("y"),
        pl.col("_parts").list.get(2).cast(pl.Float32, strict=False).alias("z"),
    ])


def test_single_space_format():
    df = pl.DataFrame({"root_id": [1], "position": ["[352484 175164 229040]"]})
    out = parse_position_column(df, "root_id")
    row = out.row(0, named=True)
    assert row["x"] == 352484.0
    assert row["y"] == 175164.0
    assert row["z"] == 229040.0


def test_double_space_alignment():
    """Codex aligns the numbers: shorter ones are padded with an extra space."""
    df = pl.DataFrame({"root_id": [1], "position": ["[631424 216180  55880]"]})
    out = parse_position_column(df, "root_id")
    row = out.row(0, named=True)
    assert row["x"] == 631424.0
    assert row["y"] == 216180.0
    assert row["z"] == 55880.0, "double space before short numbers was not parsed"


def test_negative_coordinates():
    """Defensive case: negative coordinates (possible when relative to a center)."""
    df = pl.DataFrame({"root_id": [1], "position": ["[-1000 2000 -300]"]})
    out = parse_position_column(df, "root_id")
    row = out.row(0, named=True)
    assert row["x"] == -1000.0
    assert row["z"] == -300.0
