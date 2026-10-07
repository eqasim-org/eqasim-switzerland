import gzip
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point

REQUIRED_SLOTS = [
    "agency", "stops", "routes", "trips", "stop_times"
]

OPTIONAL_SLOTS = [
    "calendar", "calendar_dates", "fare_attributes", "fare_rules",
    "shapes", "frequencies", "transfers", "pathways", "levels",
    "feed_info", "translations", "attributions"
]


def read_gtfs(gtfs_path):
    feed = {}

    with ZipFile(gtfs_path, "r") as zf:
        available_slots = zf.namelist()
        prefix = None

        if "agency.txt" in available_slots:
            prefix = ""
        else:
            for slot in available_slots:
                if slot.endswith("agency.txt"):
                    prefix = slot.replace("agency.txt", "")
                    print(f"GTFS files seem to be located in: {prefix}")
                    break

            if prefix is None:
                raise RuntimeError("No GTFS data found in archive")

        for slot in REQUIRED_SLOTS:
            if not "%s%s.txt" % (prefix, slot) in available_slots:
                raise RuntimeError("Missing GTFS information: %s" % slot)

        if not "%scalendar.txt" % prefix in available_slots and not "%scalendar_dates.txt" % prefix in available_slots:
            raise RuntimeError("At least calendar.txt or calendar_dates.txt must be specified.")

        print(f"Loading GTFS data from {gtfs_path} ...")

        for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            if "%s%s.txt" % (prefix, slot) in available_slots:
                print(f"  Loading {slot}.txt ...")

                with zf.open("%s%s.txt" % (prefix, slot)) as f:
                    feed[slot] = pd.read_csv(f, skipinitialspace = True)
            else:
                print(f"  Not loading {slot}.txt")

    if "stops" in feed:
        df_stops = feed["stops"]

        if "parent_station" not in df_stops:
            print("Missing parent_station in stops, setting to NaN")
            df_stops["parent_station"] = np.nan

        df_stops["location_type"]  = df_stops["location_type"].fillna(0).astype(int)
        df_stops["parent_station"] = df_stops["parent_station"].fillna("").astype(str)

        gtfs_geometry = [Point(xy) for xy in zip(df_stops["stop_lon"], df_stops["stop_lat"])]
        gdf = gpd.GeoDataFrame(df_stops, geometry = gtfs_geometry, crs = "EPSG:4326")
        gdf = gdf.to_crs("EPSG:2056")

        return gdf

    raise RuntimeError("GTFS archive did not contain stops.txt")


def add_missing_base_stops(gtfs_stops):
    base_ids = gtfs_stops["stop_id"].str.split(":").str[0]
    missing  = ~base_ids.isin(set(gtfs_stops["stop_id"]))

    if not missing.any():
        return gtfs_stops

    synthetic = gtfs_stops[missing].copy()
    synthetic["stop_id"] = base_ids[missing]
    synthetic = synthetic.drop_duplicates("stop_id")

    return pd.concat([gtfs_stops, synthetic], ignore_index = True)


def filter_stops_in_shapefile(stops_gdf, shapefile_path):
    polygon_gdf = gpd.read_file(shapefile_path)

    if stops_gdf.crs != polygon_gdf.crs:
        stops_gdf = stops_gdf.to_crs(polygon_gdf.crs)

    filtered = gpd.sjoin(stops_gdf, polygon_gdf, predicate = "within", how = "inner")
    filtered = filtered[["stop_id", "geometry"]]

    return filtered


def _normalize_stop_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def find_schedule_file(matsim_output_folder):
    for name in ("output_transitSchedule.xml.gz", "output_transitSchedule.xml"):
        path = os.path.join(matsim_output_folder, name)
        if os.path.exists(path):
            return path
    return None


def read_schedule_stops(schedule_path):
    """One row per stop of a MATSim transit schedule. The id is the one of the stop facility
    before any ".link:" suffix, i.e. the one used by the passenger counts."""
    opener = gzip.open if schedule_path.endswith(".gz") else open
    rows = {}

    with opener(schedule_path, "rb") as f:
        for _, element in ET.iterparse(f):
            if element.tag == "stopFacility":
                stop_id = element.attrib["id"].split(".")[0]
                if stop_id not in rows:
                    rows[stop_id] = (stop_id, element.attrib.get("name", stop_id),
                                     float(element.attrib["x"]), float(element.attrib["y"]))
                element.clear()

    df = pd.DataFrame(rows.values(), columns = ["stop_id", "stop_name", "x", "y"])
    return gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df["x"], df["y"]), crs = "EPSG:2056").drop(columns = ["x", "y"])


def supplement_with_schedule_stops(gtfs_stops, schedule_path, max_distance = 300.0):
    """The simulated schedule can contain stops that the GTFS used here does not have under the
    same id (e.g. the French stops of the Leman Express come from another feed and have other
    ids). Such a stop is matched to a stop of the GTFS with a similar name closer than
    max_distance (metres), so that both share the same id; otherwise it is added to the stops.

    Returns (stops, alias) where alias maps the schedule stop id to the GTFS stop id."""
    schedule_stops = read_schedule_stops(schedule_path)
    known = set(gtfs_stops["stop_id"])
    unknown = schedule_stops[~schedule_stops["stop_id"].isin(known)].reset_index(drop = True)

    if len(unknown) == 0:
        return gtfs_stops, {}

    from scipy.spatial import cKDTree

    reference = gtfs_stops.drop_duplicates("stop_id").reset_index(drop = True)
    tree = cKDTree(np.column_stack([reference.geometry.x, reference.geometry.y]))
    reference_names = [_normalize_stop_name(n) for n in reference["stop_name"]]

    # GTFS stops by UIC station code (the first 7 digits of the 8-digit SNCF code, which is
    # the number the Swiss stop ids start with)
    by_uic = {}
    for stop_id in reference["stop_id"]:
        match = re.match(r"^(?:Parent)?(\d{7})(?!\d)", stop_id)
        if match:
            by_uic.setdefault(match.group(1), []).append(stop_id)

    alias = {}
    for stop_id, name, geometry in zip(unknown["stop_id"], unknown["stop_name"], unknown.geometry):
        # 1. same station code, e.g. StopPoint:OCETrain:::TER-87745000 -> 8774500
        code = re.search(r"(?<!\d)(\d{8})$", stop_id)
        candidates = [c for c in by_uic.get(code.group(1)[:7], []) if not c.startswith("Parent")] if code else []
        if candidates:
            alias[stop_id] = min(candidates, key = lambda c: (len(c), c))
            continue

        # 2. similar name and close
        name = _normalize_stop_name(name)
        best = None
        for i in tree.query_ball_point([geometry.x, geometry.y], max_distance):
            other = reference_names[i]
            if len(name) >= 4 and len(other) >= 4 and (name == other or name in other or other in name):
                distance = np.hypot(reference.geometry.x.iloc[i] - geometry.x, reference.geometry.y.iloc[i] - geometry.y)
                # real stops are preferred to parent stations (ids "Parent<number>")
                key = (reference["location_type"].iloc[i] == 1 or reference["stop_id"].iloc[i].startswith("Parent"), distance)
                if best is None or key < best[0]:
                    best = (key, reference["stop_id"].iloc[i])
        if best is not None:
            alias[stop_id] = best[1]

    added = unknown[~unknown["stop_id"].isin(alias)].copy()
    wgs = added.to_crs("EPSG:4326")
    added["stop_lon"] = wgs.geometry.x.values
    added["stop_lat"] = wgs.geometry.y.values
    added["location_type"] = 0
    added["parent_station"] = ""

    print(f"Schedule stops unknown to the GTFS: {len(unknown)} -> {len(alias)} matched to a GTFS stop "
          f"by name and position, {len(added)} added as new stops")

    return pd.concat([gtfs_stops, added], ignore_index = True), alias
