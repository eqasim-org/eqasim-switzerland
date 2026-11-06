"""Compact daily schedule encoding and exact candidate checks for RDA."""

import json

import numpy as np
import pandas as pd
from numba import njit


def encode_schedules(values):
    """Encode each unique JSON schedule once. None is unknown, [] is closed."""
    parsed, rows = {}, []
    for value in values:
        key = None if pd.isna(value) else value
        if key not in parsed:
            intervals = None if key is None else json.loads(key)
            if intervals is None:
                parsed[key] = [(-1., -1.)]
            else:
                merged = []
                for start, end in sorted(intervals):
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(end, merged[-1][1])
                    else:
                        merged.append([start, end])
                if len(merged) > 1 and merged[0][0] == 0 and merged[-1][1] == 1440:
                    merged[-1][1] += merged[0][1]
                    merged.pop(0)
                parsed[key] = [(start * 60., end * 60.) for start, end in merged]
        rows.append(parsed[key])
    width = max(1, max((len(row) for row in rows), default=0))
    encoded = np.full((len(rows), width, 2), np.nan)
    for index, row in enumerate(rows):
        if row:
            encoded[index, :len(row)] = row
    return encoded


@njit(cache=True)
def eligible_candidates(schedules, candidates, start, end, tolerance_seconds=600.):
    """Test complete stays against daily intervals, including overnight sessions."""
    result = np.zeros(len(candidates), dtype=np.bool_)
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        return result
    day = np.floor(start / 86400.) * 86400.
    for position in range(len(candidates)):
        schedule = schedules[candidates[position]]
        if schedule[0, 0] == -1.:  # Explicitly unknown: no evidence to exclude.
            result[position] = True
            continue
        for interval in schedule:
            left, right = interval
            if np.isnan(left):
                break
            if right - left >= 86400.:
                result[position] = True
                break
            for shift in (-86400., 0., 86400.):
                opening = day + shift + left
                closing = day + shift + right
                mismatch = max(opening - start, 0.) + max(end - closing, 0.)
                if mismatch <= tolerance_seconds:
                    result[position] = True
                    break
            if result[position]:
                break
    return result
