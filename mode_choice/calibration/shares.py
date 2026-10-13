"""Microcensus targets and distance-filtered synthetic trip shares."""
import numpy as np
import polars as pl

from mode_choice.defaults import MODES, CALIBRATION_MIN_DISTANCE_KM
from mode_choice.calibration.optimizer import share_vector


def load_targets(path):
    frame = pl.read_csv(path, columns=["mode", "mode_share"])
    if frame["mode"].null_count() or frame["mode"].n_unique() != frame.height:
        raise ValueError("Target mode shares contain null or duplicate mode names")
    unknown = set(frame["mode"]) - set(MODES)
    if unknown:
        raise ValueError(f"Unsupported calibration target modes: {sorted(unknown)}")
    values = dict(frame.iter_rows())
    # The survey output is rounded; normalize, with absent modes treated as zero.
    return share_vector([values.get(mode, 0.0) for mode in MODES])


def trip_counts(choices, distances):
    """Count eligible legs, not tours; input rows have the same stable order."""
    if choices.height != len(distances):
        raise ValueError("Choice and distance row counts differ")
    counts = choices.filter(distances.is_finite() & (distances > CALIBRATION_MIN_DISTANCE_KM)).group_by("mode").len()
    values = dict(counts.iter_rows())
    unknown = set(values) - set(MODES)
    if unknown:
        raise ValueError(f"Unsupported sampled modes: {unknown}")
    return np.array([values.get(mode, 0) for mode in MODES], dtype=np.int64)


def load_canton_targets(path):
    """Read the wide canton CSV produced by data.microcensus.shares.

    Older files can have an unnamed index column after external-population rows
    are appended. Accept that format as well as the explicit canton_name header.
    """
    frame = pl.read_csv(path)
    if "canton_name" not in frame.columns:
        if frame.columns[0] not in ("", "Unnamed: 0"):
            raise ValueError("Cantonal targets must contain a canton_name column")
        frame = frame.rename({frame.columns[0]: "canton_name"})
    if frame["canton_name"].null_count() or frame["canton_name"].n_unique() != frame.height:
        raise ValueError("Cantonal targets contain null or duplicate canton names")
    missing = set(MODES) - set(frame.columns)
    if missing:
        frame = frame.with_columns([pl.lit(0.0).alias(mode) for mode in sorted(missing)])
    result = {}
    for row in frame.select("canton_name", *[pl.col(mode).cast(pl.Float64).fill_null(0.0) for mode in MODES]).iter_rows():
        try:
            result[row[0]] = share_vector(row[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid mode shares for canton {row[0]}: {exc}") from exc
    return result
