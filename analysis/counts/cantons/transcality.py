"""Prepare matched Transcality detector profiles for count comparison."""

import json
import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger("synpp")

SOURCE_FILES = {
    "ASTRA_LOOPS": "measurements_astra.csv",
    "CAMERA": "measurements_camera.csv",
    "CANTON_LOOPS_EDGE_BASED": "measurements_induction_loops.csv",
    "CANTON_LOOPS_LANE_BASED": "measurements_induction_loops.csv",
}


def configure(context):
    context.config("data_path")
    context.config(
        "counts_path",
        default=os.path.join(context.config("data_path"), "traffic_counts"),
    )


def execute(context):
    input_path = os.path.join(context.config("counts_path"), "Transcality")
    output_path = os.path.join(context.path(), "processed_data.gpkg")
    counts = prepare_transcality_counts(input_path)

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


def prepare_transcality_counts(input_path):
    """Build directional daily counts and 24-hour profiles from source rates."""
    input_path = os.fspath(input_path)
    matched = _read_matches(os.path.join(input_path, "matched_points.csv"))
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

    metadata = hourly.groupby("OBJECTID", as_index=False).agg(
        source=("source", "first"),
        detector_ids=("detid", lambda values: ", ".join(dict.fromkeys(values))),
        geometry=("geometry", "first"),
        osm_id=("osm_id", "first"),
        angle=("angle", "first"),
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
    frames = []
    for source, filename in SOURCE_FILES.items():
        source_ids = set(matched.loc[matched["source"] == source, "detid"])
        data = pd.read_csv(
            os.path.join(directory, filename), dtype={"detid": str}
        )
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
