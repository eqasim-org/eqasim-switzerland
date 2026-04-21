import zipfile, io
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
import os
import numpy as np
import logging

logger = logging.getLogger("synpp")

REQUIRED_SLOTS = [
    "agency", "stops", "routes", "trips", "stop_times"
]

OPTIONAL_SLOTS = [
    "calendar", "calendar_dates", "fare_attributes", "fare_rules",
    "shapes", "frequencies", "transfers", "pathways", "levels",
    "feed_info", "translations", "attributions"
]

def read_feed(path):
    feed = {}

    with zipfile.ZipFile(path, "r") as zip:
        available_slots = zip.namelist()
        prefix = None

        if "agency.txt" in available_slots:
            prefix = ""
        else:
            for slot in available_slots:
                if slot.endswith("agency.txt"):
                    prefix = slot.replace("agency.txt", "")
                    logger.warning(f"GTFS files seem to be located in: {prefix}")
                    break

            if prefix is None:
                raise RuntimeError("No GTFS data found in archive")

        for slot in REQUIRED_SLOTS:
            if not "%s%s.txt" % (prefix, slot) in available_slots:
                raise RuntimeError("Missing GTFS information: %s" % slot)

        if not "%scalendar.txt" % prefix in available_slots and not "%scalendar_dates.txt" % prefix in available_slots:
            raise RuntimeError("At least calendar.txt or calendar_dates.txt must be specified.")

        logger.info(f"Loading GTFS data from {path} ...")

        for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            if "%s%s.txt" % (prefix, slot) in available_slots:
                logger.info(f"  Loading {slot}.txt ...")

                with zip.open("%s%s.txt" % (prefix, slot)) as f:
                    feed[slot] = pd.read_csv(f, skipinitialspace = True)
            else:
                logger.info(f"  Not loading {slot}.txt")

    # Some cleanup

    for slot in ("calendar", "calendar_dates", "trips"):
        if slot in feed and "service_id" in feed[slot] and pd.api.types.is_string_dtype(feed[slot]["service_id"]):
            initial_count = len(feed[slot])
            feed[slot] = feed[slot][feed[slot]["service_id"].str.len() > 0]
            final_count = len(feed[slot])

            if final_count != initial_count:
                logger.warning(f"Removed {initial_count - final_count}/{initial_count} entries from {slot} with empty service_id")

    if "stops" in feed:
        df_stops = feed["stops"]

        if not "parent_station" in df_stops:
            logger.warning("Missing parent_station in stops, setting to NaN")
            df_stops["parent_station"] = np.nan

        df_stops["location_type"]  = df_stops["location_type"].fillna(0).astype(int)
        df_stops["parent_station"] = df_stops["parent_station"].fillna("").astype(str)

    if "transfers" in feed:
        df_transfers = feed["transfers"]

        if not "min_transfer_time" in df_transfers:
            df_transfers["min_transfer_time"] = 0

        f = df_transfers["min_transfer_time"].isna()
        if np.any(f):
            logger.warning("NaN numbers for min_transfer_time in transfers")
            df_transfers = df_transfers[~f]

        df_transfers["min_transfer_time"] = df_transfers["min_transfer_time"].astype(int)
        feed["transfers"] = df_transfers

    if "agency" in feed:
        df_agency = feed["agency"]
        df_agency.loc[df_agency["agency_id"].isna(), "agency_id"] = "generic"

    if "routes" in feed:
        df_routes = feed["routes"]
        agency_id = feed["agency"]["agency_id"].values[0]

        if not "agency_id" in df_routes:
            df_routes["agency_id"] = agency_id

        df_routes.loc[df_routes["agency_id"].isna(), "agency_id"] = agency_id

    if "shapes" in feed: del feed["shapes"]
    feed["trips"]["shape_id"] = np.nan

    # Fixes for Nantes PDL
    for item in feed.keys():
        feed[item] = feed[item].drop(columns = [
            c for c in feed[item].columns if c.startswith("ext_")
        ])

    return feed

def write_feed(feed, path):
    logger.info(f"Writing GTFS data to {path} ...")

    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as zip:
            for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
                if slot in feed:
                    if slot == "stops":
                        df_stops = feed[slot]
                    logger.info(f"  Writing {slot}.txt ...")

                    # We cannot write directly to the file handle as it
                    # is binary, but pandas only writes in text mode.
                    zip.writestr("%s.txt" % slot, feed[slot].to_csv(index = None))

    else:
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.isdir(path):
            raise RuntimeError("Should be a directory: %s" % path)

        for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
            if slot in feed:
                with open("%s/%s.txt" % (path, slot), "w+", encoding="utf-8") as f:
                    logger.info(f"  Writing {slot}.txt ...")
                    feed[slot].to_csv(f, index = None, lineterminator='\n')


def clean_feed(feed, crs = None):
    feed = copy_feed(feed)

    df_stops = feed["stops"]

    if np.count_nonzero(df_stops["location_type"] == 1) == 0:
        logger.warning("Location types seem to be malformatted. Keeping all stops.")
        df_stations = df_stops.copy()
    else:
        df_stations = df_stops[df_stops["location_type"] == 1].copy()

    df_stations["geometry"] = [
        geo.Point(*xy)
        for xy in zip(df_stations["stop_lon"], df_stations["stop_lat"])
    ]

    df_stations = gpd.GeoDataFrame(df_stations, crs = "EPSG:4326")

    if not crs is None:
        logger.info(f"Converting stops to custom CRS {crs}")
        df_stations = df_stations.to_crs(crs)
    #elif not df_area.crs is None:
    #    logger.info(f"Converting stops to area CRS {df_area.crs}")
    #    df_stations = df_stations.to_crs(df_area.crs)

    logger.info("Filtering stations ...")
    initial_count = len(df_stations)

    #df_stations = gpd.sjoin(df_stations, df_area, predicate = "within")
    final_count = len(df_stations)

    logger.info(f"Found {final_count}/{initial_count} stations inside the specified area")
    inside_stations = df_stations["stop_id"]

    # 1) Remove stations that are not inside stations and not have a parent stop
    df_stops = feed["stops"]

    #df_stops = df_stops[
    #    df_stops["parent_station"].isin(inside_stations) |
    #    (
    #        df_stops["parent_station"] == "" &
    #        df_stops["stop_id"].isin(inside_stations)
    #    )
    #]

    feed["stops"] = df_stops.copy()
    remaining_stops = feed["stops"]["stop_id"].unique()

    # 2) Remove stop times
    df_times = feed["stop_times"]
    df_times = df_times[df_times["stop_id"].astype(str).isin(remaining_stops.astype(str))]
    feed["stop_times"] = df_times.copy()

    # 3) Remove transfers
    if "transfers" in feed:
        df_transfers = feed["transfers"]
        df_transfers = df_transfers[
            df_transfers["from_stop_id"].isin(remaining_stops) &
            df_transfers["to_stop_id"].isin(remaining_stops)
        ]
        feed["transfers"] = df_transfers.copy()

    # 4) Remove pathways
    if "pathways" in feed:
        df_pathways = feed["pathways"]
        df_pathways = df_pathways[
            df_pathways["from_stop_id"].isin(remaining_stops) &
            df_pathways["to_stop_id"].isin(remaining_stops)
        ]
        feed["pathways"] = df_pathways.copy()

    # 5) Remove trips
    trip_counts = feed["stop_times"]["trip_id"].value_counts()
    remaining_trips = trip_counts[trip_counts > 1].index.values

    df_trips = feed["trips"]
    df_trips = df_trips[
        df_trips["trip_id"].isin(remaining_trips)
    ]
    feed["trips"] = df_trips.copy()

    feed["stop_times"] = feed["stop_times"][
        feed["stop_times"]["trip_id"].isin(df_trips["trip_id"].unique())
    ]

    # 6) Remove frequencies
    if "frequencies" in feed:
        df_frequencies = feed["frequencies"]
        df_frequencies = df_frequencies[
            df_frequencies["trip_id"].isin(remaining_trips)
        ]
        feed["frequencies"] = df_frequencies.copy()

    return feed

SLOT_COLLISIONS = [
    { "slot": "agency", "identifier": "agency_id", "references": [
        ("routes", "agency_id"), ("fare_attributes", "agency_id")] },
    { "slot": "stops", "identifier": "stop_id", "references": [
        ("stops", "parent_station"), ("stop_times", "stop_id"),
        ("transfers", "from_stop_id"), ("transfers", "to_stop_id"),
        ("pathways", "from_stop_id"), ("pathways", "to_stop_id")] },
    { "slot": "routes", "identifier": "route_id", "references": [
        ("trips", "route_id"), ("fare_rules", "route_id"),
        ("attributions", "route_id")] },
    { "slot": "trips", "identifier": "trip_id", "references": [
        ("stop_times", "trip_id"), ("frequencies", "trip_id"),
        ("attributions", "trip_id")] },
    { "slot": "calendar", "identifier": "service_id", "references": [
        ("calendar_dates", "service_id"), ("trips", "service_id")] },
    { "slot": "calendar_dates", "identifier": "service_id", "references": [
        ("trips", "service_id"), ("calendar", "service_id")] },
    { "slot": "fare_attributes", "identifier": "fare_id", "references": [
        ("fare_rules", "fare_id")] },
    { "slot": "shapes", "identifier": "shape_id", "references": [
        ("trips", "shape_id")] },
    { "slot": "pathways", "identifier": "pathway_id", "references": [] },
    { "slot": "levels", "identifier": "level_id", "references": [
        ("stops", "level_id")] },
    { "slot": "attributions", "identifier": "attribution_id" },
]

def validate_feed(feed):
    if "trips" not in feed:
        return

    trip_service_ids = set(feed["trips"]["service_id"].astype(str).unique())

    defined_ids = set()
    if "calendar" in feed:
        defined_ids |= set(feed["calendar"]["service_id"].astype(str).unique())
    if "calendar_dates" in feed:
        defined_ids |= set(feed["calendar_dates"]["service_id"].astype(str).unique())

    missing = trip_service_ids - defined_ids

    if missing:
        print(f"WARNING: {len(missing)} service_ids in trips not found in calendar/calendar_dates:")
        print(sorted(missing)[:20], "..." if len(missing) > 20 else "")
    else:
        print(f"OK: all {len(trip_service_ids)} service_ids are defined")


def copy_feed(feed):
    return {
        slot: feed[slot].copy() for slot in feed
    }

def merge_feeds(feeds):
    result = {}

    for k, feed in enumerate(feeds):
        print(f"\n--- Validating feed {k+1} before merge ---")
        validate_feed(feed)                          # <-- add this
        result = merge_two_feeds(result, feed, "_m{}".format(k + 1))
        print(f"--- Validating merged result after feed {k+1} ---")
        validate_feed(result)

    return result

def merge_two_feeds(first, second, suffix="_merged"):
    logger.info("Merging GTFS data ...")
    first  = copy_feed(first)
    second = copy_feed(second)

    # Convert all identifiers and references to str once upfront
    for collision in SLOT_COLLISIONS:
        col = collision["identifier"]
        if collision["slot"] in first:
            first[collision["slot"]][col] = first[collision["slot"]][col].astype(str)
        if collision["slot"] in second:
            second[collision["slot"]][col] = second[collision["slot"]][col].astype(str)
        for ref_slot, ref_col in collision.get("references", []):
            if ref_slot in first and ref_col in first[ref_slot].columns:
                first[ref_slot][ref_col] = first[ref_slot][ref_col].astype(str)
            if ref_slot in second and ref_col in second[ref_slot].columns:
                second[ref_slot][ref_col] = second[ref_slot][ref_col].astype(str)

    for collision in SLOT_COLLISIONS:
        if collision["slot"] not in first or collision["slot"] not in second:
            continue

        col       = collision["identifier"]
        df_first  = first[collision["slot"]]
        df_second = second[collision["slot"]]

        # Find duplicate IDs between the two feeds
        ids_first  = set(df_first[col].unique())
        ids_second = set(df_second[col].unique())
        duplicate_ids = ids_first & ids_second

        if len(duplicate_ids) > 0:
            logger.info(f"   Found {len(duplicate_ids)} duplicate identifiers in {collision['slot']}")

            replacement_map = {id_: id_ + suffix for id_ in duplicate_ids}

            # Remap in the second feed's main slot
            second[collision["slot"]][col] = second[collision["slot"]][col].map(
                lambda x: replacement_map.get(x, x)
            )

            # Remap all references in second feed
            for ref_slot, ref_col in collision.get("references", []):
                if ref_slot in second and ref_col in second[ref_slot].columns:
                    second[ref_slot][ref_col] = second[ref_slot][ref_col].map(
                        lambda x: replacement_map.get(x, x)
                    )

    # Concatenate all slots
    feed = {}
    for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
        if slot in first and slot in second:
            feed[slot] = pd.concat([first[slot], second[slot]], sort=True).drop_duplicates()
        elif slot in first:
            feed[slot] = first[slot].copy()
        elif slot in second:
            feed[slot] = second[slot].copy()

    defined = set()
    if "calendar" in feed:
        defined |= set(feed["calendar"]["service_id"])
    if "calendar_dates" in feed:
        defined |= set(feed["calendar_dates"]["service_id"])

    if defined:
        mask = feed["trips"]["service_id"].isin(defined)
        missing = feed["trips"][~mask]["service_id"].unique()
        if len(missing):
            print(f"Dropping {len(missing)} trips with undefined service_ids: {missing}")
            feed["trips"] = feed["trips"][mask]

    validate_feed(feed)
    return feed

def despace_stop_ids(feed, replacement = ":::"):
    feed = copy_feed(feed)

    references = None

    for item in SLOT_COLLISIONS:
        if item["slot"] == "stops":
            references = item["references"]

    df_stops = feed["stops"]
    df_stops["stop_id"] = df_stops["stop_id"].astype(str)

    search_ids = list(df_stops[df_stops["stop_id"].str.contains(" ")]["stop_id"].unique())
    replacement_ids = [item.replace(" ", replacement) for item in search_ids]

    df_stops["stop_id"] = df_stops["stop_id"].replace(search_ids, replacement_ids)

    for reference_slot, reference_field in references:
        if reference_slot in feed:
            feed[reference_slot][reference_field] = feed[reference_slot][reference_field].astype(str).replace(search_ids, replacement_ids)

    logger.info(f"De-spaced {len(search_ids)}/{len(df_stops)} stops")

    return feed