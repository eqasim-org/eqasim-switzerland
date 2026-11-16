"""Adapt network or interview-place border crossings for NPVM freight."""

import logging
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import covers

from matsim.scenario.network.utils.border_crossings import shift_right

logger = logging.getLogger("synpp")
NETWORK_STAGE = "matsim.scenario.network.convert_osm"
INTERVIEW_STAGE = "data.cross_border.road_candidates"

# Smaller ranks mean more important roads; ramps share their parent category.
ROAD_HIERARCHY = {road: rank for rank, roads in enumerate([
    ("motorway", "motorway_link", "trunk", "trunk_link"),
    ("primary", "primary_link"),
    ("secondary", "secondary_link"),
    ("tertiary", "tertiary_link"),
    ("unclassified", "residential", "living_street", "service", "track", "road"),
]) for road in roads}


def configure(context):
    context.config("data_path")
    if context.config("npvm_freight_use_interview_places", default=False):
        context.stage(INTERVIEW_STAGE)
    elif context.config("npvm_freight_border_crossings_file", default=None) is None:
        context.stage(NETWORK_STAGE)


def execute(context):

    # Adapt surveyed road places and reuse their existing directional network matches.

    if context.config("npvm_freight_use_interview_places"):
        places = context.stage(INTERVIEW_STAGE)
        path = Path(context.path(INTERVIEW_STAGE)) / "directional_links.csv"
        matches = pd.read_csv(path, dtype={"facility_id": str, "link_id": str})
        matches["link_geometry"] = gpd.GeoSeries.from_wkt(matches.link_geometry, crs=2056)
        crossings = from_interview_places(places, matches)
        logger.info("Loaded %d directional interview-place crossings", len(crossings))
    else:

        # Load the explicit GeoParquet or configured network conversion's crossing export.

        configured = context.config("npvm_freight_border_crossings_file")
        if configured is None:
            context.stage(NETWORK_STAGE)
            path = Path(context.path(NETWORK_STAGE)) / "border_crossing_links.parquet"
        else:
            path = Path(context.config("data_path")) / configured
        if not path.is_file():
            raise FileNotFoundError(f"Missing road border crossings: {path}. Rebuild {NETWORK_STAGE} "
                                    "or configure npvm_freight_border_crossings_file with its exported GeoParquet.")
        crossings = read_crossings(path)
        logger.info("Loaded %d eligible road crossing records (%s mode) from %s",
                    len(crossings), crossings.assignment_mode.iloc[0], path)

    # Return LV95 point geometry alongside source link IDs, hierarchy and modes.

    return crossings


def validate(context):
    if context.config("npvm_freight_use_interview_places"):
        return None
    configured = context.config("npvm_freight_border_crossings_file")
    if configured is None:
        return None
    path = Path(context.config("data_path")) / configured
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def read_crossings(path):
    frame = gpd.read_parquet(path)
    if frame.crs is None or frame.crs.to_epsg() != 2056:
        raise ValueError(f"Border crossing GeoParquet must declare EPSG:2056: {path}")
    if not frame.geometry.geom_type.eq("Point").all() or frame.geometry.is_empty.any():
        raise ValueError(f"Border crossing GeoParquet must contain nonempty crossing points: {path}")
    required = {"link_id", "crossing_x", "crossing_y", "highway", "modes", "direction", "border_x", "border_y", "right_offset_m"}
    if not required.issubset(frame):
        raise ValueError(f"Border crossing GeoParquet missing columns {sorted(required - set(frame))}: {path}")
    if not frame.direction.isin(["in", "out"]).all():
        raise ValueError(f"Border crossing direction must be 'in' or 'out': {path}")
    xy = frame[["crossing_x", "crossing_y"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(xy.to_numpy()).all():
        raise ValueError(f"Border crossing GeoParquet has invalid coordinates: {path}")
    frame[["crossing_x", "crossing_y"]] = xy
    if not np.allclose(np.column_stack([frame.geometry.x, frame.geometry.y]), xy.to_numpy(), rtol=0, atol=1e-6):
        raise ValueError(f"Border crossing geometry disagrees with crossing coordinates: {path}")
    border_xy = frame[["border_x", "border_y"]].to_numpy(dtype=float)
    if not np.isfinite(border_xy).all() or not np.allclose(frame.right_offset_m, 1.0):
        raise ValueError(f"Border crossings must have finite boundary points and a one-metre right offset: {path}")
    if not np.allclose(np.linalg.norm(xy.to_numpy() - border_xy, axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"Border crossing coordinates must be one metre from the boundary intersection: {path}")
    frame["link_id"] = frame.link_id.astype(str)
    # Use truck wherever that mode exists in the export; car is for car-only networks.
    modes = frame.modes.fillna("").map(lambda value: set(re.split(r"[,;\s]+", str(value))))
    mode = "truck" if modes.map(lambda values: "truck" in values).any() else "car"
    frame["highway"] = frame.highway.map(highway_type)
    frame["hierarchy"] = frame.highway.map(ROAD_HIERARCHY)
    eligible = frame.hierarchy.lt(max(ROAD_HIERARCHY.values()))
    frame = frame.loc[eligible & modes.map(lambda values: mode in values)].copy()
    if frame.empty:
        raise ValueError(f"No {mode}-permitted eligible road crossings found in {path}")
    if set(frame.direction) != {"in", "out"}:
        raise ValueError(f"Eligible {mode} road crossings must provide both in and out directions: {path}")
    frame["assignment_mode"] = mode
    frame["crossing_source"] = "network"
    return frame.reset_index(drop=True)


def select_crossings(zones, crossings, direction):
    """Return candidate points and normalized probabilities for each zone/direction."""
    if direction not in {"in", "out"}:
        raise ValueError(f"Unknown border crossing direction: {direction}")
    crossings = crossings.loc[crossings.direction.eq(direction)].copy()
    if crossings.empty:
        raise ValueError(f"No eligible '{direction}' crossings available")
    # Split links can represent the same physical crossing; retain its best road class.
    crossings = crossings.sort_values(["hierarchy", "link_id"], kind="stable")
    crossings = crossings.loc[~crossings[["border_x", "border_y"]].round(3).duplicated()].copy()
    # Zone matching uses the unshifted intersection so lane offsets cannot change eligibility.
    boundary_points = gpd.GeoSeries(gpd.points_from_xy(crossings.border_x, crossings.border_y),
                                    index=crossings.index, crs=2056)
    choices = []
    approach = None
    if {"foreign_x", "foreign_y"}.issubset(crossings):
        xy = crossings[["foreign_x", "foreign_y"]].to_numpy(dtype=float)
        if not np.isfinite(xy).all():
            raise ValueError("Crossing GeoParquet contains invalid foreign approach coordinates")
        approach = gpd.points_from_xy(xy[:, 0], xy[:, 1], crs=2056)
    for zone in zones.itertuples():
        inside = covers(zone.geometry, boundary_points.array)
        if approach is not None:
            inside |= covers(zone.geometry, approach)
        if inside.any():
            candidates = crossings.loc[inside].copy()
            best = candidates.hierarchy.min()
            candidates = candidates.loc[candidates.hierarchy.le(best + 1)]
            method = "best_two_levels_in_zone"
        else:
            distances = boundary_points.distance(zone.geometry)
            candidates = crossings.assign(distance_to_zone_m=distances).sort_values(
                ["distance_to_zone_m", "hierarchy", "link_id"], kind="stable").head(2).copy()
            method = "nearest_two_crossings"
        candidates = candidates.sort_values(["hierarchy", "link_id"], kind="stable").copy()
        # Use interview importance when supplied; otherwise road weights are 8:4:2:1.
        weights = (candidates.sampling_weight.to_numpy(dtype=float).copy()
                   if "sampling_weight" in candidates else
                   2.0 ** (-candidates.hierarchy.to_numpy(dtype=float)))
        if method == "nearest_two_crossings":
            # Distance is from the exact intersection to the region, measured in kilometres.
            weights /= 1.0 + candidates.distance_to_zone_m.to_numpy(dtype=float) / 1000.0
        candidates["probability"] = weights / weights.sum()
        candidates["zone_id"] = zone.zone_id
        candidates["selection_method"] = method
        choices.append(candidates.drop(columns=["geometry", "link_geometry", "border_geometry"], errors="ignore"))
    return pd.concat(choices, ignore_index=True) if choices else pd.DataFrame()


def highway_type(value):
    """Choose the highest freight road class if the network stores multiple tags."""
    parts = re.split(r"[,;\s]+", str(value).strip().lower())
    return min((part for part in parts if part in ROAD_HIERARCHY),
               key=lambda part: ROAD_HIERARCHY[part], default="unknown")


def from_interview_places(places, directional_links=None):
    """Adapt road interview points to the common crossing schema without changing inputs.

    Importance is the sampling weight; its distance below the maximum is the
    hierarchy rank. Matched entry/exit curves provide directed right offsets.
    Unmatched points are bidirectional at their original location (no bearing
    or one-metre offset is invented). PT references and zero scores are omitted.
    """
    roads = places.loc[places.label.eq("road")].copy().to_crs(2056)
    if "is_candidate" in roads:
        roads = roads.loc[roads.is_candidate].copy()
    roads["importance"] = pd.to_numeric(roads.importance, errors="raise")
    if not (np.isfinite(roads.importance) & roads.importance.ge(0)).all():
        raise ValueError("Road interview importance must be finite and nonnegative")
    roads = roads.loc[roads.importance.gt(0)].copy()
    if roads.empty:
        raise ValueError("No road interview points with positive importance")
    if (roads.geometry.isna().any() or roads.geometry.is_empty.any()
            or not roads.geometry.geom_type.eq("Point").all()):
        raise ValueError("Road interview places must have nonempty point geometries")
    if not np.isfinite(np.column_stack([roads.geometry.x, roads.geometry.y])).all():
        raise ValueError("Road interview places have non-finite coordinates")
    if roads.border_crossing_point_id.isna().any() or roads.border_crossing_point_id.duplicated().any():
        raise ValueError("Road interview point IDs must be nonempty and unique")
    matches = None
    if directional_links is not None:
        if directional_links.facility_id.duplicated().any():
            raise ValueError("Directional interview matches have duplicate facility IDs")
        matches = directional_links.set_index("facility_id")
    maximum = roads.importance.max()
    records = []
    for place in roads.itertuples():
        declared = getattr(place, "direction", None)
        if pd.notna(declared):
            declared = str(declared).strip().lower()
            declared = {"entry": "in", "exit": "out", "both": None, "bidirectional": None, "": None}.get(declared, declared)
            if declared not in {"in", "out", None}:
                raise ValueError(f"Unknown interview crossing direction: {declared}")
        else:
            declared = None
        for direction, suffix in (("in", "entry"), ("out", "exit")):
            if declared is not None and direction != declared:
                continue
            facility_id = f"{place.border_crossing_point_id}_{suffix}"
            match = matches.loc[facility_id] if matches is not None and facility_id in matches.index else None
            curve = match.get("link_geometry") if match is not None else None
            matched = curve is not None and not curve.is_empty and curve.geom_type == "LineString" and curve.length > 0
            anchor = curve.interpolate(curve.project(place.geometry)) if matched else place.geometry
            point = shift_right(curve, curve.project(place.geometry)) if matched else anchor
            records.append({
                "source_point_id": str(place.border_crossing_point_id),
                "interview_place": str(place.interview_place), "importance": place.importance,
                "hierarchy": maximum - place.importance, "sampling_weight": place.importance,
                "link_id": str(match.link_id) if matched else f"interview:{facility_id}",
                "direction": direction, "direction_source": "network_match" if matched else "provided" if declared else "assumed_bidirectional",
                "crossing_source": "interview_places", "crossing_x": point.x, "crossing_y": point.y,
                "border_x": anchor.x, "border_y": anchor.y, "right_offset_m": 1.0 if matched else 0.0,
                "foreign_x": place.geometry.x, "foreign_y": place.geometry.y,
                "highway": str(match.get("road_class", "interview_road")) if matched else "interview_road",
                "modes": "car", "assignment_mode": "car", "geometry": point,
                "border_geometry": anchor, "link_geometry": curve if matched else None,
            })
    result = gpd.GeoDataFrame(records, geometry="geometry", crs=2056)
    for column in ("border_geometry", "link_geometry"):
        result[column] = gpd.GeoSeries(result[column], crs=2056)
    return result
