"""Read MOBIS walking legs and add endpoint elevations."""

import logging
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series
from matsim.scenario.network.utils.elevation_estimator import ElevationEstimator

COLUMNS = ["participant_id", "leg_id", "mode", "in_switzerland", "implausible", "started_at", "finished_at",
           "start_x", "start_y", "end_x", "end_y", "length", "duration",]


def configure(context):
    context.config("data_path")


def execute(context):

    # Read the complete dataset and keep valid walking legs.

    archive = Path(context.config("data_path")) / "GPS_data/MOBIS_Covid_version_3.zip"
    with ZipFile(archive) as source, source.open("tracking/legs.csv") as stream:
        data = pd.read_csv(stream, usecols=COLUMNS)
    data = clean_legs(data)
    if data.empty:
        raise ValueError("No walking legs remain after filtering")

    # Let the elevation estimator handle cached heights and missing requests.

    data = add_elevations(data, context.config("data_path"))
    data = data.dropna(subset=["start_z", "end_z"]).copy()
    data["elevation_change_m"] = data.end_z - data.start_z
    data["endpoint_grade"] = data.elevation_change_m / data.euclidean_distance_m
    data = data.loc[data.endpoint_grade.abs() <= 0.3].reset_index(drop=True)

    # Save and return all retained legs, without sampling.

    data.to_parquet(Path(context.path()) / "walking_legs.parquet", index=False)
    logging.getLogger("synpp").info("Walking dataset: %d legs", len(data))
    return data


def clean_legs(data):
    walking = data["mode"].str.lower().isin(["mode::walk", "walk", "walking"])
    swiss = coerce_boolean_series(
        data.in_switzerland.fillna(False), name="in_switzerland"
    )
    implausible = coerce_boolean_series(
        data.implausible.fillna(True), name="implausible"
    )
    data = data.loc[walking & swiss & ~implausible].copy()
    data["started_at"] = pd.to_datetime(data.started_at, utc=True, format="mixed")
    data["finished_at"] = pd.to_datetime(data.finished_at, utc=True, format="mixed")
    data["euclidean_distance_m"] = np.hypot(data.end_x - data.start_x, data.end_y - data.start_y)
    data["observed_speed_m_s"] = data.length / data.duration
    data["observed_detour"] = data.length / data.euclidean_distance_m.replace(0, np.nan)

    valid = (
        data.finished_at.ge(data.started_at)
        & data.euclidean_distance_m.ge(300)
        & data.duration.between(30, 14400)
        & data.observed_speed_m_s.between(0.2, 3)
        & data.observed_detour.between(0.8, 4)
        & np.isfinite(
            data[["start_x", "start_y", "end_x", "end_y", "length", "duration"]]
        ).all(axis=1)
        & data.participant_id.notna()
        & data.leg_id.notna()
    )
    data = data.loc[valid].drop_duplicates(COLUMNS)
    ambiguous = data.duplicated(["participant_id", "leg_id", "started_at"], keep=False)
    return data.loc[~ambiguous].reset_index(drop=True)


def add_elevations(data, data_path):
    endpoints = np.vstack([data[["start_x", "start_y"]], data[["end_x", "end_y"]]])
    coordinates, inverse = np.unique(endpoints, axis=0, return_inverse=True)
    estimator = ElevationEstimator(
        data_path=str(data_path), coordinates=coordinates.tolist()
    )
    nodes = pd.DataFrame(coordinates, columns=["x", "y"]).assign(z=np.nan)
    heights = estimator.run_on_nodes(nodes, interpolate_missing=False).z.to_numpy()[
        inverse
    ]
    data = data.copy()
    data["start_z"], data["end_z"] = heights[: len(data)], heights[len(data) :]
    return data


def validate(context):
    archive = Path(context.config("data_path")) / "GPS_data/MOBIS_Covid_version_3.zip"
    return archive.stat().st_size, archive.stat().st_mtime_ns
