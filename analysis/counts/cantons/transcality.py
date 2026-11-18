"""
This data is sent by Lukas to Milos. The counts are already processes by Transcality.
Prepare matched Transcality detector profiles for count comparison.
"""

import json
import logging
import os
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..matching.matcher import MATCH_ENDPOINT_COLUMNS


logger = logging.getLogger("synpp")

SOURCE_FILES = {
    "ASTRA_LOOPS": "measurements_astra.csv",
    "CAMERA": "measurements_camera.csv",
    "CANTON_LOOPS_EDGE_BASED": "measurements_induction_loops.csv",
    "CANTON_LOOPS_LANE_BASED": "measurements_induction_loops.csv",
}

# Daily corrections supplied from the Geneva traffic-count map:
# https://www.arcgis.com/apps/mapviewer/index.html?url=https://vector.sitg.ge.ch/arcgis/rest/services/OTC_COMPTAGE_TRAFIC/FeatureServer/0&source=sd
# Each tuple receives ONE corrected total, shared by its detector profiles.
DAILY_FLOW_OVERRIDES = {
    ("astra_200_4", "astra_200_5", "astra_200_6", "2071.2"): 28327,
    ("2017.1",): 11207,
    ("315.2",): 6691,
    ("017.0.102",): 5533,
    ("017.0.103",): 11196,
    ("035.0.106",): 2276,
}
EXCLUDED_DETECTOR_IDS = {"316.2"}
SITG_OVERRIDES_FILE = os.path.join("sitg", "transcality_overrides_2025.csv")


def configure(context):
    context.config("correct_transcality_counts", default=True)
    context.config("data_path")
    context.config(
        "counts_path",
        default=os.path.join(context.config("data_path"), "traffic_counts"),
    )


def execute(context):
    input_path = os.path.join(context.config("counts_path"), "Transcality")
    output_path = os.path.join(context.path(), "processed_data.gpkg")
    counts = prepare_transcality_counts(
        input_path, correct_transcality_counts=context.config("correct_transcality_counts")
    )

    counts.to_file(output_path, driver="GPKG")
    counts.drop(columns="geometry").to_csv(
        os.path.join(context.path(), "processed_daily_counts.csv"), index=False
    )

    figure, axis = plt.subplots(figsize=(10, 10))
    counts.plot(ax=axis, column="source", categorical=True, legend=True, markersize=8)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(os.path.join(context.path(), "fig.png"), dpi=200)
    plt.close(figure)
    return output_path


def prepare_transcality_counts(input_path, *, correct_transcality_counts=True, apply_sitg_overrides=True):
    """Build directional daily counts and 24-hour profiles from source rates."""
    input_path = os.fspath(input_path)
    matched = _read_matches(os.path.join(input_path, "matched_points.csv"))
    if correct_transcality_counts:
        matched = matched[~matched.detid.isin(EXCLUDED_DETECTOR_IDS)].copy()
    measurements = _read_measurements(
        os.path.join(input_path, "transcality_counts"), matched
    )
    hourly = _to_hourly_vehicle_counts(measurements)
    hourly = hourly.merge(
        matched[
            [
                "source",
                "detid",
                "geometry",
                "osm_id",
                "angle",
                *MATCH_ENDPOINT_COLUMNS,
            ]
        ],
        on=["source", "detid"],
        how="inner",
        validate="many_to_one",
    )

    # Match each detector to the current network before summing observations.
    # Exported MATSim IDs are not stable across network generations.
    hourly["OBJECTID"] = hourly["source"] + ":" + hourly["detid"]

    profile = hourly.groupby(["OBJECTID", "hour"], as_index=False).agg(
        hourly_flow=("hourly_flow", "sum"),
        hourly_lower=("hourly_lower", "sum"),
        hourly_upper=("hourly_upper", "sum"),
    )
    complete = profile.groupby("OBJECTID")["hour"].nunique()
    incomplete = complete[complete != 24]
    if not incomplete.empty:
        logger.warning(
            "Dropping %d Transcality observations without all 24 hours.",
            len(incomplete),
        )
        profile = profile[~profile["OBJECTID"].isin(incomplete.index)]

    if correct_transcality_counts:
        overrides = {**(_read_sitg_overrides(input_path) if apply_sitg_overrides else {}), **DAILY_FLOW_OVERRIDES}
        profile = _apply_daily_flow_overrides(profile, overrides)

    metadata = hourly.groupby("OBJECTID", as_index=False).agg(
        source=("source", "first"),
        detector_ids=("detid", lambda values: ", ".join(dict.fromkeys(values))),
        geometry=("geometry", "first"),
        osm_id=("osm_id", "first"),
        angle=("angle", "first"),
        **{column: (column, "first") for column in MATCH_ENDPOINT_COLUMNS},
    )
    daily = profile.groupby("OBJECTID", as_index=False).agg(
        flow=("hourly_flow", "sum"),
        flow_lower=("hourly_lower", "sum"),
        flow_upper=("hourly_upper", "sum"),
    )
    profiles = profile.groupby("OBJECTID").apply(
        _profile_as_json, include_groups=False
    ).rename("profile_json").reset_index()

    result = metadata.merge(daily, on="OBJECTID", how="inner").merge(
        profiles, on="OBJECTID", how="inner"
    )
    numeric = ["flow", "flow_lower", "flow_upper", "angle"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(
        subset=[*numeric, "osm_id", "geometry"]
    )
    result[["flow", "flow_lower", "flow_upper"]] = result[
        ["flow", "flow_lower", "flow_upper"]
    ].round(2)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:2056")


def _read_sitg_overrides(input_path):
    """Load reviewed 2025 weekday corrections from the local SITG snapshot."""
    path = os.path.join(input_path, SITG_OVERRIDES_FILE)
    if not os.path.exists(path):
        return {}
    rows = pd.read_csv(path, dtype={"detector_id": str})
    if rows.detector_id.isna().any() or rows.detector_id.duplicated().any():
        raise ValueError("SITG corrections require unique nonempty detector IDs")
    if not (pd.to_numeric(rows.count_year).eq(2025).all()
            and pd.to_numeric(rows.daily_flow).gt(0).all()
            and np.isfinite(pd.to_numeric(rows.daily_flow)).all()):
        raise ValueError("SITG corrections require positive 2025 weekday totals")
    protected = {detector for group in DAILY_FLOW_OVERRIDES for detector in group} | EXCLUDED_DETECTOR_IDS
    if rows.detector_id.isin(protected).any():
        raise ValueError("SITG corrections must not replace explicit manual corrections/exclusions")
    return {(row.detector_id,): float(row.daily_flow) for row in rows.itertuples()}


def _apply_daily_flow_overrides(profile, overrides=None):
    """Preserve profile shapes while correcting totals before link aggregation."""
    profile = profile.copy()
    detector_ids = profile.OBJECTID.str.split(":", n=1).str[1]
    for detectors, daily_flow in (DAILY_FLOW_OVERRIDES if overrides is None else overrides).items():
        selected = detector_ids.isin(detectors)
        present = set(detector_ids[selected])
        if not present:
            continue
        if present != set(detectors):
            raise ValueError(f"Incomplete Transcality correction group: missing {sorted(set(detectors) - present)}")
        original = profile.loc[selected, "hourly_flow"].sum()
        if not np.isfinite(original) or original <= 0:
            raise ValueError(f"Cannot rescale nonpositive Transcality flow for {detectors}")
        factor = daily_flow / original
        columns = ["hourly_flow", "hourly_lower", "hourly_upper"]
        profile.loc[selected, columns] *= factor

        # Allocate hundredths across hours/detectors so the exported profiles
        # and their aggregated daily count both sum to the exact correction.
        values = profile.loc[selected, "hourly_flow"].to_numpy() * 100
        cents = np.floor(values).astype(np.int64)
        remainder = int(round(daily_flow * 100)) - int(cents.sum())
        order = np.argsort(-(values - cents), kind="stable")
        cents[order[:remainder]] += 1
        profile.loc[selected, "hourly_flow"] = cents / 100
        profile.loc[selected, ["hourly_lower", "hourly_upper"]] = profile.loc[
            selected, ["hourly_lower", "hourly_upper"]
        ].round(2)
        logger.info("Corrected Transcality %s total: %.2f -> %.2f vehicles/day.",
                    ", ".join(detectors), original, daily_flow)
    return profile


def validate(context):
    if not context.config("correct_transcality_counts"):
        return "transcality-corrections-disabled"
    path = Path(context.config("counts_path")) / "Transcality" / SITG_OVERRIDES_FILE
    return hashlib.sha256(path.read_bytes() if path.exists() else b"no-sitg-overrides").hexdigest()


def _read_matches(path):
    matched = pd.read_csv(path, encoding="utf-8-sig", dtype={"detid": str})
    required = {
        "detid",
        "source",
        "geometry",
        "link_osm_id",
        "link_x_from_node",
        "link_y_from_node",
        "link_x_to_node",
        "link_y_to_node",
    }
    missing = required.difference(matched.columns)
    if missing:
        raise ValueError(f"matched_points.csv is missing: {sorted(missing)}")

    # A detector ID shared by different sources is ambiguous: discard every
    # occurrence, rather than choosing one source or double-counting it.
    source_counts = matched.groupby("detid")["source"].nunique()
    ambiguous_ids = source_counts.index[source_counts > 1]
    if len(ambiguous_ids):
        logger.warning(
            "Dropping %d Transcality detector IDs shared across sources (%d matches).",
            len(ambiguous_ids), matched["detid"].isin(ambiguous_ids).sum(),
        )
        matched = matched[~matched["detid"].isin(ambiguous_ids)].copy()

    matched["matched_at"] = pd.to_datetime(
        matched.get("matched_at"), errors="coerce", utc=True
    )
    duplicate_mask = matched.duplicated(["source", "detid"], keep=False)
    if duplicate_mask.any():
        logger.warning(
            "%d Transcality detector IDs have multiple matches; keeping the latest match.",
            matched.loc[duplicate_mask, ["source", "detid"]].drop_duplicates().shape[0],
        )
    matched = matched.sort_values("matched_at", na_position="first").drop_duplicates(
        ["source", "detid"], keep="last"
    )
    matched["geometry"] = gpd.GeoSeries.from_wkt(matched["geometry"], crs="EPSG:4326")
    matched = gpd.GeoDataFrame(matched, geometry="geometry", crs="EPSG:4326").to_crs(
        "EPSG:2056"
    )
    matched["angle"] = np.degrees(
        np.arctan2(
            matched["link_y_to_node"] - matched["link_y_from_node"],
            matched["link_x_to_node"] - matched["link_x_from_node"],
        )
    )
    return matched.rename(columns={"link_osm_id": "osm_id"})


def _read_measurements(directory, matched):
    # Check the raw files before restricting to matched stations. An ID is
    # ambiguous even if only one of its sources has a network match.
    # Edge/lane loop sources share a file: read and count that file only once.
    raw = {
        filename: pd.read_csv(os.path.join(directory, filename), dtype={"detid": str})
        for filename in dict.fromkeys(SOURCE_FILES.values())
    }
    occurrences = pd.concat([data["detid"].drop_duplicates() for data in raw.values()])
    ambiguous_ids = set(occurrences[occurrences.duplicated(keep=False)])
    if ambiguous_ids:
        logger.warning(
            "Dropping %d Transcality detector IDs shared across measurement files.",
            len(ambiguous_ids),
        )
    frames = []
    for source, filename in SOURCE_FILES.items():
        source_ids = set(matched.loc[matched["source"] == source, "detid"]) - ambiguous_ids
        data = raw[filename]
        if "type" in data.columns:
            data = data[data["type"].eq("all")]
        data = data[data["detid"].isin(source_ids)].copy()
        data = data.rename(columns={"flow": "value", "median_value": "value"})
        required = {"interval", "detid", "value", "quantile_25", "quantile_75"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{filename} is missing: {sorted(missing)}")
        data["source"] = source
        frames.append(
            data[
                [
                    "interval",
                    "detid",
                    "value",
                    "quantile_25",
                    "quantile_75",
                    "source",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _to_hourly_vehicle_counts(measurements):
    measurements = measurements.copy()
    value_columns = ["value", "quantile_25", "quantile_75", "interval"]
    measurements[value_columns] = measurements[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    measurements = measurements.dropna(subset=value_columns)
    measurements = measurements.sort_values(["source", "detid", "interval"])

    def durations(intervals):
        values = intervals.to_numpy(dtype=float)
        return np.diff(np.append(values, 24 * 3600))

    measurements["duration_seconds"] = measurements.groupby(
        ["source", "detid"], sort=False
    )["interval"].transform(durations)
    invalid = measurements["duration_seconds"].le(0)
    if invalid.any():
        raise ValueError("Transcality intervals must be unique and strictly increasing.")
    crosses_hour = (
        measurements["interval"] % 3600 + measurements["duration_seconds"] > 3600
    )
    if crosses_hour.any():
        raise ValueError("Transcality intervals must not cross hourly boundaries.")

    factor = measurements["duration_seconds"] / 3600
    measurements["hour"] = (measurements["interval"] // 3600).astype(int)
    measurements["hourly_flow"] = measurements["value"] * factor
    measurements["hourly_lower"] = measurements["quantile_25"] * factor
    measurements["hourly_upper"] = measurements["quantile_75"] * factor
    return measurements.groupby(
        ["source", "detid", "hour"], as_index=False
    )[["hourly_flow", "hourly_lower", "hourly_upper"]].sum()


def _profile_as_json(group):
    group = group.sort_values("hour")
    return json.dumps(
        {
            "hour": group["hour"].astype(int).tolist(),
            "flow": group["hourly_flow"].round(2).tolist(),
            "lower": group["hourly_lower"].round(2).tolist(),
            "upper": group["hourly_upper"].round(2).tolist(),
        },
        separators=(",", ":"),
    )
