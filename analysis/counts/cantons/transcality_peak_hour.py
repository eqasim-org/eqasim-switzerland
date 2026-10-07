"""Transcality totals and bound envelopes for a configured peak-hour window."""

import json
import os

import numpy as np

from . import transcality


def configure_peak_hours(context):
    context.config("analysis.counts.peak_hour_start", default=7)
    context.config("analysis.counts.peak_hour_end", default=10)
    return get_peak_hours(context)


def get_peak_hours(context):
    start = context.config("analysis.counts.peak_hour_start")
    end = context.config("analysis.counts.peak_hour_end")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not 0 <= start < end <= 24 or int(start) != start or int(end) != end):
        raise ValueError("Peak hours must be whole hours with 0 <= start < end <= 24 (end exclusive).")
    return int(start), int(end)


def configure(context):
    transcality.configure(context)
    configure_peak_hours(context)


def select_peak_counts(counts, start, end):
    """Keep full-day profiles for hover charts, but sum only selected hours."""
    result = counts.copy()
    for index, row in result.iterrows():
        profile = json.loads(row["profile_json"])
        hours = np.asarray(profile["hour"])
        selected = (hours >= start) & (hours < end)
        if sorted(hours[selected].tolist()) != list(range(start, end)):
            raise ValueError("Transcality profile is missing or duplicates a peak hour.")
        for column, field in (("flow", "flow"), ("flow_lower", "lower"), ("flow_upper", "upper")):
            result.loc[index, column] = np.asarray(profile[field], dtype=float)[selected].sum()
    result["period_hours"] = end - start
    result["flow_period"] = f"{start:02d}:00–{end:02d}:00"
    return result


def execute(context):
    start, end = get_peak_hours(context)
    counts = transcality.prepare_transcality_counts(
        os.path.join(context.config("counts_path"), "Transcality")
    )
    counts = select_peak_counts(counts, start, end)
    output = os.path.join(context.path(), "processed_data.gpkg")
    counts.to_file(output, driver="GPKG")
    counts.drop(columns="geometry").to_csv(
        os.path.join(context.path(), "processed_peak_counts.csv"), index=False
    )
    return output
