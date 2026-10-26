"""Shared OVGK coding for survey estimation and population mode choice."""

PT_QUALITY_CODES = {"A": 0, "B": 1, "C": 2, "D": 3, "NONE": 4}


def encode_pt_quality(values):
    """Encode a pandas Series; missing/unclassified service has quality 4."""
    return (values.astype("string").str.strip().str.upper()
            .map(PT_QUALITY_CODES).fillna(4).astype("int8"))


def pt_quality_expression(column):
    """Equivalent Polars expression without importing Polars in estimation."""
    import polars as pl

    return (pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase()
            .replace_strict(PT_QUALITY_CODES, default=4, return_dtype=pl.Int8))
