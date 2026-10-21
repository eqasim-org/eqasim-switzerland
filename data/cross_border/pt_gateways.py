import gzip
import logging
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

logger = logging.getLogger("synpp")

# Distance between an interview place and a border-crossing segment of a train
# for that train to be considered crossing there.
CROSSING_RADIUS = 3000.0

# Long-distance services, by line name. Stations served by one of them are the
# gateways of the long-distance travellers of the survey.
LONG_DISTANCE_LINE = re.compile(r"^(IC|ICE|ICN|EC|EN|TGV|RJ|RJX|NJ|IR|IRLEX|GEX|BEX|D)\d*$")
LONG_DISTANCE_ROUTE = re.compile(r":LYR:|TGV|OUIGO", re.IGNORECASE)

RAIL_MODES = {"rail"}

# Gateways with little service are dropped: an agent sent there would wait for one
# of a handful of trains (e.g. Nurieux, 2 TGV a day, for Paris via La Plaine, rather
# than Bellegarde). A gateway is kept if its departures (summed over the crossing
# trains stopping there) reach this share of the best-served gateway of the same
# interview place, and at least MIN_DEPARTURES.
MIN_SERVICE_SHARE = 0.1
MIN_DEPARTURES = 5


def configure(context):
    context.config("border_offset", default = 50000)
    context.config("cross_border_pt_gateways_file", default = None)

    context.stage("data.cross_border.interview_places")
    context.stage("data.spatial.swiss_border")
    context.stage("matsim.scenario.network.convert_pt_schedule")


def read_schedule(path):
    """ Stops and routes of a MATSim transit schedule (only what is needed here). """
    stops, routes = [], []
    line_name = None

    with gzip.open(path, "rb") as stream:
        for event, element in ET.iterparse(stream, events = ("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]

            if event == "start":
                if tag == "transitLine":
                    line_name = element.get("name") or element.get("id")
                continue

            if tag == "stopFacility":
                stops.append((element.get("id"), float(element.get("x")), float(element.get("y")), element.get("name")))
                element.clear()

            elif tag == "transitRoute":
                mode = (element.findtext("transportMode") or "").strip()

                if mode in RAIL_MODES:
                    profile = element.find("routeProfile")
                    departures = element.find("departures")

                    routes.append(dict(
                        line_name = str(line_name),
                        route_id = element.get("id"),
                        stops = [stop.get("refId") for stop in profile],
                        departures = 0 if departures is None else len(departures),
                    ))

                element.clear()

    df_stops = pd.DataFrame(stops, columns = ["stop_id", "x", "y", "name"]).drop_duplicates("stop_id")
    return df_stops, routes


def is_long_distance(route):
    return bool(LONG_DISTANCE_LINE.match(route["line_name"])) or bool(LONG_DISTANCE_ROUTE.search(route["route_id"]))


def find_gateways(df_stops, routes, df_points, swiss_border, border_offset):
    stops = df_stops.set_index("stop_id")
    stops["in_ch"] = shapely.contains_xy(swiss_border, stops["x"].values, stops["y"].values)

    border_line = swiss_border.boundary.simplify(50)
    stops["distance_to_border"] = shapely.distance(border_line, shapely.points(stops["x"].values, stops["y"].values))

    point_geometries = df_points.geometry.values
    rows = []

    for route in routes:
        route_stops = [stop_id for stop_id in route["stops"] if stop_id in stops.index]
        if len(route_stops) < 2:
            continue

        in_ch = stops.loc[route_stops, "in_ch"].values
        if in_ch.all() or not in_ch.any():
            continue

        x = stops.loc[route_stops, "x"].values
        y = stops.loc[route_stops, "y"].values
        long_distance = is_long_distance(route)

        for index in np.flatnonzero(in_ch[:-1] != in_ch[1:]):
            # A crossing belongs to one interview place only (in Basel, several lines
            # cross a few km apart): among the places close to the segment, the one
            # closest to where it crosses the border. The segment alone is not enough,
            # it may start right next to another place (Schaffhausen -> Singen passes
            # Neuhausen but crosses at Thayngen).
            segment = shapely.linestrings([[x[index], y[index]], [x[index + 1], y[index + 1]]])
            eligible = np.flatnonzero(shapely.distance(point_geometries, segment) <= CROSSING_RADIUS)

            if len(eligible) == 0:
                continue

            crossing = shapely.intersection(segment, border_line)
            if crossing.is_empty:
                crossing = segment
            point_index = eligible[np.argmin(shapely.distance(point_geometries[eligible], crossing))]

            # The stops abroad from the crossing on, away from Switzerland, as long as
            # they stay inside the simulated perimeter.
            step = 1 if in_ch[index] else -1
            position = index + 1 if in_ch[index] else index
            abroad = []
            # whether the route comes from beyond the perimeter, as the trains of the travellers living there do
            from_far = False

            while 0 <= position < len(route_stops) and not in_ch[position]:
                stop_id = route_stops[position]
                if stops.at[stop_id, "distance_to_border"] > border_offset:
                    from_far = True
                    break
                abroad.append(stop_id)
                position += step

            for stop_id in abroad:
                rows.append((point_index, stop_id, route["departures"], route["departures"] if from_far else 0, long_distance))

    df = pd.DataFrame(rows, columns = ["point_index", "stop_id", "departures", "far_departures", "has_long_distance"])
    if len(df) == 0:
        return pd.DataFrame(columns = ["interview_place", "border_crossing_point_id", "gateway_name", "x", "y",
                                       "departures", "far_departures", "has_long_distance", "distance_to_border"])

    df["gateway_name"] = stops.loc[df["stop_id"], "name"].values

    # One gateway per station: the platforms of a station share its name.
    df = df.groupby(["point_index", "gateway_name"]).agg(
        departures = ("departures", "sum"), far_departures = ("far_departures", "sum"), has_long_distance = ("has_long_distance", "any"), stop_id = ("stop_id", "first"),
    ).reset_index()

    max_departures = df.groupby("point_index")["departures"].transform("max")
    df = df[df["departures"] >= np.maximum(MIN_SERVICE_SHARE * max_departures, MIN_DEPARTURES)].copy()

    df["x"] = stops.loc[df["stop_id"], "x"].values
    df["y"] = stops.loc[df["stop_id"], "y"].values
    df["distance_to_border"] = stops.loc[df["stop_id"], "distance_to_border"].values
    df["interview_place"] = df_points["interview_place"].values[df["point_index"]]
    df["border_crossing_point_id"] = df_points["border_crossing_point_id"].values[df["point_index"]]

    return df[["interview_place", "border_crossing_point_id", "gateway_name", "x", "y",
               "departures", "far_departures", "has_long_distance", "distance_to_border"]]


def apply_overrides(df_gateways, df_points, path):
    df_override = pd.read_csv(path)

    missing = {"interview_place", "gateway_name", "x", "y", "has_long_distance"} - set(df_override.columns)
    if missing:
        raise RuntimeError(f"cross_border_pt_gateways_file is missing the columns {sorted(missing)}")

    unknown = set(df_override["interview_place"]) - set(df_points["interview_place"])
    if unknown:
        raise RuntimeError(f"cross_border_pt_gateways_file lists unknown pt interview places: {sorted(unknown)}")

    df_override = df_override.merge(df_points[["interview_place", "border_crossing_point_id"]], on = "interview_place")
    df_override["departures"] = np.nan
    df_override["far_departures"] = np.nan
    df_override["distance_to_border"] = np.nan
    df_override["has_long_distance"] = df_override["has_long_distance"].astype(bool)

    df_gateways = df_gateways[~df_gateways["interview_place"].isin(df_override["interview_place"])]
    return pd.concat([df_gateways, df_override[df_gateways.columns]], ignore_index = True)


def execute(context):
    border_offset = context.config("border_offset")
    swiss_border = context.stage("data.spatial.swiss_border").copy().iloc[0]

    df_points = context.stage("data.cross_border.interview_places")
    df_points = df_points[df_points["label"] == "pt"].reset_index(drop = True)

    schedule_path = context.stage("matsim.scenario.network.convert_pt_schedule")["schedule"]
    df_stops, routes = read_schedule(schedule_path)
    logger.info("Read %d stops and %d rail routes from %s", len(df_stops), len(routes), schedule_path)

    df_gateways = find_gateways(df_stops, routes, df_points, swiss_border, border_offset)

    override_path = context.config("cross_border_pt_gateways_file")
    if override_path:
        df_gateways = apply_overrides(df_gateways, df_points, override_path)

    df_gateways = df_gateways.sort_values(["interview_place", "distance_to_border"]).reset_index(drop = True)
    df_gateways["gateway_id"] = "PTG_" + df_gateways.index.astype(str)

    df_gateways = gpd.GeoDataFrame(df_gateways, geometry = gpd.points_from_xy(df_gateways["x"], df_gateways["y"]), crs = "EPSG:2056")
    df_gateways = df_gateways.drop(columns = ["x", "y"])

    for place in df_points["interview_place"]:
        gateways = df_gateways[df_gateways["interview_place"] == place]
        if len(gateways) == 0:
            logger.warning("pt interview place %s: no gateway found, its far-away travellers stay on the interview place.", place)
        else:
            logger.info("pt interview place %s: gateways %s (long distance: %s)", place,
                        ", ".join(gateways["gateway_name"]),
                        ", ".join(gateways.loc[gateways["has_long_distance"], "gateway_name"]) or "none")

    df_gateways.to_file(os.path.join(context.path(), "pt_gateways.gpkg"), driver = "GPKG")

    return df_gateways
