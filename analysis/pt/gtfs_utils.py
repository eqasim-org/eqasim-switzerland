"""GTFS loading helpers, extracted from eqasim-switzerland's synpp stages."""

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
    """
    Some GTFS station families have no row for their own bare base id
    (the part of stop_id before the first ":") - only per-platform child
    rows like "8501013:0:1"/"8501013:0:2" exist, with no plain "8501013"
    row alongside them (unlike most families, which do have one). Checked
    against this feed: ~2100 base ids feed-wide are affected, including
    several Leman Express stations (Pont-Cearl, Mies, Tannay, Chambesy,
    Versoix, Coppet, Lancy-Pont-Rouge, Geneve-Champel, Geneve-Eaux-Vives,
    Chene-Bourg, Lancy-Bachet - see lemanis.py).

    Anything that joins on gtfs_code == stop_id (this whole pipeline's
    convention, e.g. tpg_data.py's crosswalks, lemanis.py's
    STOP_NAME_TO_GTFS_CODE, MATSim's stop_id_gtfs_base) silently fails to
    resolve a stop_name/coordinate for those ids. This adds one synthetic
    row per missing base id, copied from its first child (same
    name/coordinates - good enough for labeling and map placement; not
    meant to represent an exact platform location).
    """

    base_ids = gtfs_stops["stop_id"].str.split(":").str[0]
    missing = ~base_ids.isin(set(gtfs_stops["stop_id"]))

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
