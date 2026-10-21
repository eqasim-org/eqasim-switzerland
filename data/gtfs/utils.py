import zipfile
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
import os
import numpy as np
import os, pathlib
import re
import itertools
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


def _as_id_string(column):
    """An identifier column as strings, keeping the missing values missing (an id read as a float because of
    missing values, 1.0, becomes "1")."""
    def convert(value):
        if pd.isna(value):
            return None
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)

    return column.map(convert).astype(object)


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
                    logger.info(f"GTFS files seem to be located in: {prefix}")
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
                    feed[slot] = pd.read_csv(f, skipinitialspace = True, low_memory = False)
            else:
                logger.info(f"  Not loading {slot}.txt")

    # Some cleanup

    for slot in ("calendar", "calendar_dates", "trips"):
        if slot in feed and "service_id" in feed[slot] and pd.api.types.is_string_dtype(feed[slot]["service_id"]):
            initial_count = len(feed[slot])
            feed[slot] = feed[slot][feed[slot]["service_id"].str.len() > 0]
            final_count = len(feed[slot])

            if final_count != initial_count:
                logger.info(f"Removed {initial_count - final_count}/{initial_count} entries from {slot} with empty service_id")

    if "stops" in feed:
        df_stops = feed["stops"]

        if not "parent_station" in df_stops:
            logger.info("Missing parent_station in stops, setting to NaN")
            df_stops["parent_station"] = np.nan

        df_stops["location_type"]  = df_stops["location_type"].fillna(0).astype(int)
        df_stops["parent_station"] = df_stops["parent_station"].fillna("").astype(str)

    if "transfers" in feed:
        df_transfers = feed["transfers"]

        if not "min_transfer_time" in df_transfers:
            df_transfers["min_transfer_time"] = 0

        f = df_transfers["min_transfer_time"].isna()
        if np.any(f):
            logger.info("NaN numbers for min_transfer_time in transfers")
            df_transfers = df_transfers[~f].copy()

        df_transfers["min_transfer_time"] = df_transfers["min_transfer_time"].astype(int)
        feed["transfers"] = df_transfers

    if "agency" in feed:
        df_agency = feed["agency"]
        df_agency["agency_id"] = _as_id_string(df_agency["agency_id"])
        df_agency.loc[df_agency["agency_id"].isna(), "agency_id"] = "generic"

    if "routes" in feed:
        df_routes = feed["routes"]
        agency_id = feed["agency"]["agency_id"].values[0]

        if not "agency_id" in df_routes:
            df_routes["agency_id"] = agency_id

        # the agency ids are strings in the agency table: same type in the routes
        df_routes["agency_id"] = _as_id_string(df_routes["agency_id"])
        df_routes.loc[df_routes["agency_id"].isna(), "agency_id"] = agency_id

    # The shapes are kept; the trips that refer to a shape that is not in shapes.txt (or a feed without shapes.txt) get
    # no shape, as pt2matsim expects every shape_id of trips.txt to be defined
    if "shape_id" not in feed["trips"]:
        feed["trips"]["shape_id"] = np.nan
    feed["trips"]["shape_id"] = _as_id_string(feed["trips"]["shape_id"])
    if "shapes" in feed:
        feed["shapes"]["shape_id"] = _as_id_string(feed["shapes"]["shape_id"])
        defined = set(feed["shapes"]["shape_id"])
    else:
        defined = set()
    undefined = feed["trips"]["shape_id"].notna() & ~feed["trips"]["shape_id"].isin(defined)
    if undefined.any():
        logger.info(f"  {undefined.sum()} trips refer to a shape that is not defined, their shape_id is removed")
        feed["trips"].loc[undefined, "shape_id"] = None

    # Fixes for Nantes PDL
    for item in feed.keys():
        feed[item] = feed[item].drop(columns = [
            c for c in feed[item].columns if c.startswith("ext_")
        ])

    return feed


def write_feed(feed, path):
    logger.info(f"Writing GTFS data to {path} ...")

    if path.endswith(".zip"):
        # Write to a temp file in the same directory and rename into place
        # at the end, so a killed/interrupted run never leaves a truncated,
        # corrupted zip sitting at `path` — either the old file is still
        # there untouched, or the new one is complete.
        tmp_path = f"{path}.tmp{os.getpid()}"
        try:
            with zipfile.ZipFile(tmp_path, "w") as zip:
                for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
                    if slot in feed:
                        if slot == "stops":
                            df_stops = feed[slot]
                        logger.info(f"  Writing {slot}.txt ...")

                        # We cannot write directly to the file handle as it
                        # is binary, but pandas only writes in text mode.
                        zip.writestr("%s.txt" % slot, feed[slot].to_csv(index = None))
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

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
        logger.info("Location types seem to be malformatted. Keeping all stops.")
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

    logger.info("Filtering stations ...")
    initial_count = len(df_stations)

    #df_stations = gpd.sjoin(df_stations, df_area, predicate = "within")
    final_count = len(df_stations)

    logger.info(f"Found {final_count}/{initial_count} stations inside the specified area")
    inside_stations = df_stations["stop_id"]

    # 1) Remove stations that are not inside stations and not have a parent stop
    df_stops = feed["stops"]

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
        logger.info(f"WARNING: {len(missing)} service_ids in trips not found in calendar/calendar_dates:")
        logger.info(f"{sorted(missing)[:20]}{' ...' if len(missing) > 20 else ''}")
    else:
        logger.info(f"OK: all {len(trip_service_ids)} service_ids are defined")


def copy_feed(feed):
    return {
        slot: feed[slot].copy() for slot in feed
    }


def merge_feeds(feeds):
    result = {}

    for k, feed in enumerate(feeds):
        logger.info(f"\n--- Validating feed {k+1} before merge ---")
        validate_feed(feed)                          # <-- add this
        result = merge_two_feeds(result, feed, "_m{}".format(k + 1))
        logger.info(f"--- Validating merged result after feed {k+1} ---")
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

    # Resolve service_id collisions across calendar + calendar_dates combined.
    # Handling each table separately would miss the case where one feed defines
    # a service_id only in calendar and the other only in calendar_dates.
    ids_first_svc = set()
    ids_second_svc = set()
    for t in ("calendar", "calendar_dates"):
        if t in first:
            ids_first_svc |= set(first[t]["service_id"].unique())
        if t in second:
            ids_second_svc |= set(second[t]["service_id"].unique())

    duplicate_service_ids = ids_first_svc & ids_second_svc
    if duplicate_service_ids:
        logger.info(f"   Found {len(duplicate_service_ids)} duplicate service_ids across calendar/calendar_dates")
        svc_replacement_map = {id_: id_ + suffix for id_ in duplicate_service_ids}
        for t in ("calendar", "calendar_dates"):
            if t in second and "service_id" in second[t].columns:
                second[t]["service_id"] = second[t]["service_id"].replace(svc_replacement_map)
        if "trips" in second and "service_id" in second["trips"].columns:
            second["trips"]["service_id"] = second["trips"]["service_id"].replace(svc_replacement_map)

    for collision in SLOT_COLLISIONS:
        # service_id collisions are handled above across both calendar tables
        if collision["slot"] in ("calendar", "calendar_dates"):
            continue

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
            second[collision["slot"]][col] = second[collision["slot"]][col].replace(replacement_map)

            # Remap all references in second feed
            for ref_slot, ref_col in collision.get("references", []):
                if ref_slot in second and ref_col in second[ref_slot].columns:
                    second[ref_slot][ref_col] = second[ref_slot][ref_col].replace(replacement_map)

    # Concatenate all slots. Large tables (stop_times, trips, shapes) are not
    # deduplicated — after ID remapping above there are no cross-feed duplicates,
    # and drop_duplicates() on millions of rows is very expensive.
    LARGE_SLOTS = {"stop_times", "trips", "shapes"}
    feed = {}
    for slot in REQUIRED_SLOTS + OPTIONAL_SLOTS:
        if slot in first and slot in second:
            combined = pd.concat([first[slot], second[slot]])
            feed[slot] = combined if slot in LARGE_SLOTS else combined.drop_duplicates()
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
            logger.info(f"Dropping {len(missing)} trips with undefined service_ids: {missing}")
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




def fill_calendar_from_dates(feed):
    """
    pt2matsim requires every service_id referenced in trips.txt to have a row
    in calendar.txt. For service_ids defined only in calendar_dates.txt, create
    a synthetic calendar.txt row with all weekday flags set to 0 and start/end
    dates derived from the actual dates in calendar_dates.txt.
    """
    if "calendar_dates" not in feed:
        return feed

    feed = copy_feed(feed)

    in_calendar = set(feed["calendar"]["service_id"].astype(str)) if "calendar" in feed else set()
    df_cal_dates = feed["calendar_dates"].copy()
    df_cal_dates["service_id"] = df_cal_dates["service_id"].astype(str)
    df_cal_dates["date"] = df_cal_dates["date"].astype(str)

    df_new = (
        df_cal_dates
        .groupby("service_id")["date"]
        .agg(start_date="min", end_date="max")
        .reset_index()
    )
    df_new = df_new[~df_new["service_id"].isin(in_calendar)]

    if df_new.empty:
        logger.info("No synthetic calendar entries needed.")
        return feed

    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        logger.info(day)
        df_new[day] = 0

    feed["calendar"] = pd.concat([feed["calendar"], df_new], ignore_index=True) if "calendar" in feed else df_new
    logger.info(f"Added {len(df_new)} synthetic calendar entries for service_ids only in calendar_dates.")
    
    return feed


def extract_station_code(stop_id):
    """
    Extract a normalized international station code (UIC/DIDOK) from a
    stop_id, used to detect the same physical station represented under
    different ID schemes by different source feeds. E.g. Swiss "8501023"
    and French SNCF "StopArea:OCE85010231" both encode Coppet: Swiss GTFS
    uses the bare 7-digit UIC code, SNCF embeds the 8-digit code (7 digits
    + check digit) in its own ID scheme.
    """
    matches = re.findall(r"\d{7,8}", str(stop_id))
    if not matches:
        return None

    code = matches[-1]
    if len(code) == 8:
        code = code[:7]  # drop UIC check digit
    return code


def bridge_cross_source_stations(feed, min_transfer_time=300, verbose=False):
    """
    Detect stations that are represented under different stop_id schemes by
    different source feeds (e.g. a cross-border rail station listed once in
    the Swiss GTFS and once in the French GTFS) and add synthetic
    transfers.txt entries linking every platform of one side to every
    platform of the other, so transfer routing doesn't treat them as
    unrelated, disconnected stops. Connecting at the platform level (rather
    than relying on parent_station resolution) is deliberate: not every
    downstream consumer resolves parent_station reachability.
    """
    feed = copy_feed(feed)
    df_stops = feed["stops"].copy()

    def station_of(row):
        if row["location_type"] == 1 or not row["parent_station"]:
            return row["stop_id"]
        return row["parent_station"]

    df_stops["_station_id"] = df_stops.apply(station_of, axis=1)

    code_to_stations = {}
    for station_id in df_stops["_station_id"].unique():
        code = extract_station_code(station_id)
        if code is None:
            continue
        code_to_stations.setdefault(code, set()).add(station_id)

    groups = {code: ids for code, ids in code_to_stations.items() if len(ids) > 1}

    if not groups:
        logger.info("No cross-source station matches found, no bridging transfers added.")
        return feed

    station_to_platforms = (
        df_stops.groupby("_station_id")["stop_id"]
        .apply(lambda s: sorted(s.astype(str).unique()))
        .to_dict()
    )

    existing_pairs = set()
    if "transfers" in feed:
        existing_pairs = set(zip(
            feed["transfers"]["from_stop_id"].astype(str),
            feed["transfers"]["to_stop_id"].astype(str),
        ))

    new_rows = []
    for ids in groups.values():
        platform_lists = [station_to_platforms.get(sid, [sid]) for sid in sorted(ids)]
        for platforms_a, platforms_b in itertools.permutations(platform_lists, 2):
            for a, b in itertools.product(platforms_a, platforms_b):
                if a != b and (a, b) not in existing_pairs:
                    new_rows.append({"from_stop_id": a, "to_stop_id": b, "transfer_type": 2, "min_transfer_time": min_transfer_time})
                    existing_pairs.add((a, b))

    if not new_rows:
        logger.info(f"Found {len(groups)} matching station group(s) but all transfers already present.")
        return feed

    df_new = pd.DataFrame(new_rows)
    base = feed["transfers"] if "transfers" in feed else pd.DataFrame(columns=df_new.columns)
    feed["transfers"] = pd.concat([base, df_new], ignore_index=True)

    logger.info(f"Bridged {len(groups)} cross-source station group(s) (platform-to-platform) with {len(new_rows)} new transfer(s).")
    if verbose:
        for code, ids in sorted(groups.items()):
            platforms = [p for sid in sorted(ids) for p in station_to_platforms.get(sid, [sid])]
            logger.info(f"  {code}: {platforms}")

    return feed


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between paired arrays of coordinates."""
    r = 6371000
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def filter_long_transfers(feed, max_distance_m=1500, max_transfer_time_s=1200):
    """
    Drop transfers.txt rows that are implausibly long: the two stops are
    more than max_distance_m apart, or min_transfer_time exceeds
    max_transfer_time_s. Catches false-positive station matches from
    bridge_cross_source_stations (e.g. a UIC/DIDOK code collision linking
    two unrelated stations) as well as any overly long transfers already
    present in the source feeds.
    """
    if "transfers" not in feed:
        return feed

    feed = copy_feed(feed)
    df_stops = feed["stops"].set_index("stop_id")[["stop_lat", "stop_lon"]]
    df_t = feed["transfers"]

    t = (
        df_t
        .merge(df_stops, left_on="from_stop_id", right_index=True, how="left")
        .rename(columns={"stop_lat": "lat1", "stop_lon": "lon1"})
        .merge(df_stops, left_on="to_stop_id", right_index=True, how="left")
        .rename(columns={"stop_lat": "lat2", "stop_lon": "lon2"})
    )

    has_coords = t[["lat1", "lon1", "lat2", "lon2"]].notna().all(axis=1)
    dist_m = pd.Series(np.nan, index=t.index)
    dist_m[has_coords] = haversine_m(
        t.loc[has_coords, "lat1"], t.loc[has_coords, "lon1"],
        t.loc[has_coords, "lat2"], t.loc[has_coords, "lon2"],
    )

    too_far = dist_m > max_distance_m
    too_long = df_t["min_transfer_time"].fillna(0) > max_transfer_time_s
    drop = too_far.fillna(False) | too_long

    logger.info(f"Dropping {drop.sum()}/{len(df_t)} transfers exceeding {max_distance_m}m or {max_transfer_time_s}s "
          f"({(too_far & ~too_long).sum()} by distance, {(too_long & ~too_far).sum()} by duration, "
          f"{(too_far & too_long).sum()} by both)")

    feed["transfers"] = df_t[~drop].reset_index(drop=True)
    return feed


def add_nearby_transfers(feed, max_distance_m = 300.0, walk_speed_m_s = 1.0, detour_factor = 1.3, base_time_s = 0.0,
                         max_transfer_time_s = 1200):
    """
    Adds a transfer time (transfer_type 2) between every two served stops at
    most max_distance_m apart that have no rule yet.

    MATSim (SwissRailRaptor) lets the passengers walk between any two stops
    within its beeline walk distance (300 m) and only takes the walking time
    when no minimal transfer time is given, while the feeds give times (often
    several minutes) between the stops of the same station only. Changing at a
    neighbouring stop then looked faster than at the station itself (e.g. bus 6
    to the m2 at Lausanne, St-Etienne rather than Sallaz, which has 240 s).

    The time of a pair (a, b) builds on the rules around it: a rule from a to a
    stop c plus walking from c to b, or a rule from a stop c to b plus walking
    from a to c; the lowest is kept, and at least walking from a to b. Walking
    takes distance x detour_factor / walk_speed_m_s. Without any rule around a
    or b, the time is base_time_s plus walking.

    Returns (feed, set of the added (from_stop_id, to_stop_id) pairs).
    """
    from scipy.spatial import cKDTree

    feed = copy_feed(feed)
    stops = feed["stops"].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors = "coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors = "coerce")
    stops = stops.dropna(subset = ["stop_lat", "stop_lon"])
    coordinates = stops.set_index("stop_id")[["stop_lat", "stop_lon"]]

    served = set(feed["stop_times"]["stop_id"].astype(str))
    platforms = stops[stops["stop_id"].isin(served)]
    if "location_type" in platforms.columns:
        platforms = platforms[pd.to_numeric(platforms["location_type"], errors = "coerce").fillna(0) == 0]

    # Pairs of served stops close to each other (local metric coordinates, exact distance below)
    lat, lon = platforms["stop_lat"].to_numpy(), platforms["stop_lon"].to_numpy()
    xy = np.c_[np.radians(lon) * np.cos(np.radians(lat)), np.radians(lat)] * 6371000
    pairs = cKDTree(xy).query_pairs(max_distance_m * 1.01, output_type = "ndarray")
    ids = platforms["stop_id"].to_numpy()
    pairs = pd.DataFrame({
        "a": np.r_[ids[pairs[:, 0]], ids[pairs[:, 1]]],
        "b": np.r_[ids[pairs[:, 1]], ids[pairs[:, 0]]],
    })

    transfers = feed["transfers"].copy() if "transfers" in feed else pd.DataFrame(columns = ["from_stop_id", "to_stop_id", "transfer_type", "min_transfer_time"])
    transfers["from_stop_id"], transfers["to_stop_id"] = transfers["from_stop_id"].astype(str), transfers["to_stop_id"].astype(str)
    existing = set(zip(transfers["from_stop_id"], transfers["to_stop_id"]))
    pairs = pairs[[pair not in existing for pair in zip(pairs["a"], pairs["b"])]]

    def distance(from_ids, to_ids):
        p, q = coordinates.reindex(from_ids.to_numpy()), coordinates.reindex(to_ids.to_numpy())
        return pd.Series(haversine_m(p["stop_lat"].to_numpy(), p["stop_lon"].to_numpy(),
                                     q["stop_lat"].to_numpy(), q["stop_lon"].to_numpy()), index = from_ids.index)

    walk = lambda d: d * detour_factor / walk_speed_m_s
    pairs["distance"] = distance(pairs["a"], pairs["b"])
    pairs = pairs[pairs["distance"] <= max_distance_m]

    # The rules the times build on (not the forbidden transfers, type 3)
    rules = transfers.copy()
    if "transfer_type" in rules:
        rules = rules[pd.to_numeric(rules["transfer_type"], errors = "coerce").fillna(2) != 3]
    rules["time"] = pd.to_numeric(rules["min_transfer_time"], errors = "coerce").fillna(0)
    rules = rules[["from_stop_id", "to_stop_id", "time"]]

    from_a = pairs.merge(rules.rename(columns = {"from_stop_id": "a", "to_stop_id": "c"}), on = "a")
    from_a["candidate"] = from_a["time"] + walk(distance(from_a["c"], from_a["b"]))
    to_b = pairs.merge(rules.rename(columns = {"from_stop_id": "c", "to_stop_id": "b"}), on = "b")
    to_b["candidate"] = to_b["time"] + walk(distance(to_b["a"], to_b["c"]))
    anchored = pd.concat([from_a, to_b]).dropna(subset = ["candidate"]).groupby(["a", "b"])["candidate"].min()

    pairs = pairs.set_index(["a", "b"])
    pairs["anchored"] = anchored.reindex(pairs.index)
    time = pairs["anchored"].fillna(base_time_s + walk(pairs["distance"]))
    time = np.maximum(time, walk(pairs["distance"])).clip(upper = max_transfer_time_s)
    new = pd.DataFrame({
        "from_stop_id": pairs.index.get_level_values("a"), "to_stop_id": pairs.index.get_level_values("b"),
        "transfer_type": 2, "min_transfer_time": np.ceil(time.to_numpy()).astype(int),
    })

    feed["transfers"] = pd.concat([feed["transfers"], new], ignore_index = True) if "transfers" in feed else new
    logger.info(f"Added {len(new)} transfer times between stops at most {max_distance_m:.0f} m apart without a rule "
                f"({int(pairs['anchored'].notna().sum())} from the rules around them, the others from walking only)")
    return feed, set(zip(new["from_stop_id"], new["to_stop_id"]))


def time_to_seconds(t):
    """Parse a GTFS HH:MM:SS time (HH may exceed 23 for post-midnight trips) into seconds."""
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _times_to_seconds(times):
    """Vectorised time_to_seconds: NaN for an empty or malformed time."""
    parts = times.astype(str).str.split(":", expand = True)
    if parts.shape[1] < 3:
        return pd.Series(np.nan, index = times.index)
    return (pd.to_numeric(parts[0], errors = "coerce") * 3600 + pd.to_numeric(parts[1], errors = "coerce") * 60
            + pd.to_numeric(parts[2], errors = "coerce"))


def _seconds_to_time(seconds):
    seconds = int(round(seconds))
    return "%02d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def repair_backward_stop_times(feed, max_rounds = 5):
    """Repairs the stop times out of order inside a trip: a stop whose time is before the departure from the previous
    stop or after the arrival at the next one, while these two are in order, gets a time between them (by
    shape_dist_traveled when available, else halfway). E.g. the Haute-Savoie feed has trips with 00:00:00 at one stop
    in the middle of the morning (a missing time): MATSim then refuses the schedule ("Invalid PT stop sequence/offsets")."""
    st = feed["stop_times"]
    sequence = pd.to_numeric(st["stop_sequence"], errors = "coerce")
    order = st.assign(_seq = sequence).sort_values(["trip_id", "_seq"], kind = "stable").index

    trip = st.loc[order, "trip_id"].astype(str).to_numpy()
    arrival = _times_to_seconds(st.loc[order, "arrival_time"]).to_numpy(dtype = float)
    departure = _times_to_seconds(st.loc[order, "departure_time"]).to_numpy(dtype = float)
    distance = (pd.to_numeric(st.loc[order, "shape_dist_traveled"], errors = "coerce").to_numpy(dtype = float)
                if "shape_dist_traveled" in st.columns else np.full(len(order), np.nan))

    def shifted(values, k, fill):
        out = np.full(len(values), fill, dtype = values.dtype)
        if k > 0:
            out[k:] = values[:-k]
        else:
            out[:k] = values[-k:]
        return out

    same_previous = trip == shifted(trip, 1, None)
    same_next = trip == shifted(trip, -1, None)
    repaired = np.zeros(len(order), dtype = bool)

    for _ in range(max_rounds):
        previous_departure, next_arrival = shifted(departure, 1, np.nan), shifted(arrival, -1, np.nan)
        with np.errstate(invalid = "ignore"):
            bad = same_previous & same_next & (previous_departure <= next_arrival) & (
                (arrival < previous_departure) | (departure > next_arrival))
        if not bad.any():
            break

        previous_distance, next_distance = shifted(distance, 1, np.nan), shifted(distance, -1, np.nan)
        with np.errstate(invalid = "ignore", divide = "ignore"):
            share = (distance - previous_distance) / (next_distance - previous_distance)
        share = np.where(np.isfinite(share), np.clip(share, 0.0, 1.0), 0.5)
        times = previous_departure + share * (next_arrival - previous_departure)
        arrival[bad] = departure[bad] = times[bad]
        repaired |= bad

    if repaired.any():
        rows = order[repaired]
        new_times = [_seconds_to_time(t) for t in arrival[repaired]]
        feed = dict(feed)
        feed["stop_times"] = st = st.copy()
        st.loc[rows, "arrival_time"] = new_times
        st.loc[rows, "departure_time"] = new_times
        trips = sorted(set(trip[repaired]))
        logger.warning("Stop times out of order repaired (interpolated between the neighbouring stops): %d stop(s) of %d trip(s), e.g. %s",
                       repaired.sum(), len(trips), ", ".join(trips[:5]))

    return feed


def expand_service_dates(feed):
    """
    Expand calendar.txt (weekday flags between start_date/end_date) and
    calendar_dates.txt (exceptions) into a long table of (service_id, date)
    pairs, one row per day a service is actually active.
    """
    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    records = []

    if "calendar" in feed:
        df_cal = feed["calendar"].copy()
        df_cal["start_date"] = pd.to_datetime(df_cal["start_date"], format="%Y%m%d")
        df_cal["end_date"] = pd.to_datetime(df_cal["end_date"], format="%Y%m%d")

        for row in df_cal.itertuples(index=False):
            dates = pd.date_range(row.start_date, row.end_date, freq="D")
            if len(dates) == 0:
                continue
            flags = np.array([getattr(row, day) for day in weekday_cols])[dates.weekday]
            active_dates = dates[flags.astype(bool)]
            records.extend((row.service_id, d) for d in active_dates)

    df_active = pd.DataFrame(records, columns=["service_id", "date"])

    if "calendar_dates" in feed:
        df_cd = feed["calendar_dates"].copy()
        df_cd["service_id"] = df_cd["service_id"].astype(str)
        df_cd["date"] = pd.to_datetime(df_cd["date"].astype(str), format="%Y%m%d")

        added = df_cd[df_cd["exception_type"] == 1][["service_id", "date"]]
        removed = df_cd[df_cd["exception_type"] == 2][["service_id", "date"]]

        df_active["service_id"] = df_active["service_id"].astype(str)
        df_active = pd.concat([df_active, added], ignore_index=True).drop_duplicates()

        if not removed.empty:
            df_active = df_active.merge(removed, on=["service_id", "date"], how="left", indicator=True)
            df_active = df_active[df_active["_merge"] == "left_only"].drop(columns="_merge")

    return df_active


def merge_cross_agency_blocks(feed, max_gap_s=600, min_gap_s=-60):
    """
    Some source feeds split one physical vehicle's journey into multiple
    GTFS trips at an agency handoff (e.g. a train or bus crossing into
    another operator's network mid-route), linked only via a shared native
    block_id — trips.txt's standard "same vehicle, next trip" mechanism.
    As separate GTFS trips under different route_ids/agencies, routing
    tools see a required transfer where none actually exists.

    For every block_id shared by trips from more than one agency, orders
    that block's trips chronologically (by first departure) and merges any
    *adjacent* pair whose agencies differ, whose last/first stop_id match,
    and whose gap falls within [min_gap_s, max_gap_s] (default: up to 10
    minutes after, up to 1 minute before — enough for a real scheduled
    dwell at a handoff point without accidentally bridging an unrelated
    same-block deadhead). Adjacent trips that stay within one agency are
    left untouched — they're already correctly linked by the native
    block_id and aren't the "fake transfer" problem this targets.

    A merged run keeps its earliest trip's identity throughout (route_id,
    trip_id, agency, direction, etc.) — i.e. whatever was valid when the
    physical vehicle's journey began — and gets a fresh calendar_dates-only
    service_id covering exactly the dates every trip in the run actually
    operated (since each original trip's calendar may only partially
    overlap the others').

    Since block_id is only meaningful within a single feed, this is meant
    to run per-source-feed, before merging feeds together (see execute()).
    Stop_id here is compared as-is (no cross-scheme UIC/DIDOK
    normalization) since within one feed all trips already share one
    stop_id scheme.
    """
    if "block_id" not in feed["trips"].columns:
        logger.info("No block_id column found, skipping cross-agency block merging.")
        return feed

    feed = copy_feed(feed)
    df_trips = feed["trips"]
    df_st = feed["stop_times"]

    trip_agency = pd.Series("", index=df_trips["trip_id"], dtype=object)
    if "routes" in feed and "agency" in feed:
        df_routes = feed["routes"][["route_id", "agency_id"]].copy()
        df_routes["route_id"] = df_routes["route_id"].astype(str)
        df_routes["agency_id"] = df_routes["agency_id"].astype(str)
        df_agency = feed["agency"][["agency_id", "agency_name"]].copy()
        df_agency["agency_id"] = df_agency["agency_id"].astype(str)
        route_agency = df_routes.merge(df_agency, on="agency_id", how="left").set_index("route_id")["agency_name"]
        trip_route = df_trips.set_index("trip_id")["route_id"].astype(str)
        trip_agency = trip_route.map(route_agency).fillna("")

    trips_with_block = df_trips[df_trips["block_id"].notna()][["trip_id", "block_id"]].copy()
    if trips_with_block.empty:
        logger.info("No trips have a block_id, skipping cross-agency block merging.")
        return feed
    trips_with_block["agency_name"] = trips_with_block["trip_id"].map(trip_agency)

    agency_counts = trips_with_block.groupby("block_id")["agency_name"].nunique()
    cross_agency_block_ids = set(agency_counts[agency_counts > 1].index)
    if not cross_agency_block_ids:
        logger.info("No cross-agency block_id groups found, skipping cross-agency block merging.")
        return feed

    relevant = trips_with_block[trips_with_block["block_id"].isin(cross_agency_block_ids)]
    relevant_trip_ids = set(relevant["trip_id"])
    logger.info(f"merge_cross_agency_blocks: {len(cross_agency_block_ids)} cross-agency block_id group(s), "
          f"{len(relevant_trip_ids)} trip(s).")

    st = (
        df_st[df_st["trip_id"].isin(relevant_trip_ids)]
        [["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time"]]
        .sort_values(["trip_id", "stop_sequence"])
    )
    first_of = {
        tid: (row.stop_id, row.arrival_time, row.departure_time)
        for tid, row in st.groupby("trip_id", sort=False).first().iterrows()
    }
    last_of = {
        tid: (row.stop_id, row.arrival_time, row.departure_time)
        for tid, row in st.groupby("trip_id", sort=False).last().iterrows()
    }
    first_dep_s = {tid: time_to_seconds(v[2]) for tid, v in first_of.items()}
    last_arr_s = {tid: time_to_seconds(v[1]) for tid, v in last_of.items()}

    trip_service = df_trips.set_index("trip_id")["service_id"].astype(str)
    service_dates = expand_service_dates(feed).groupby("service_id")["date"].apply(set).to_dict()

    runs = []
    for block_id, group in relevant.groupby("block_id"):
        ordered = sorted(group["trip_id"], key=lambda t: first_dep_s[t])
        run = [ordered[0]]
        for t in ordered[1:]:
            prev = run[-1]
            same_stop = last_of[prev][0] == first_of[t][0]
            gap = first_dep_s[t] - last_arr_s[prev]
            different_agency = trip_agency.get(prev, "") != trip_agency.get(t, "")
            overlaps = bool(
                service_dates.get(trip_service.get(prev), set())
                & service_dates.get(trip_service.get(t), set())
            )
            if same_stop and different_agency and overlaps and min_gap_s <= gap <= max_gap_s:
                run.append(t)
            else:
                if len(run) > 1:
                    runs.append(run)
                run = [t]
        if len(run) > 1:
            runs.append(run)

    if not runs:
        logger.info("No adjacent cross-agency trips connect cleanly, skipping cross-agency block merging.")
        return feed

    st_by_trip = {
        tid: g.sort_values("stop_sequence")
        for tid, g in df_st[df_st["trip_id"].isin({t for run in runs for t in run})].groupby("trip_id")
    }

    combined_frames = []
    drop_trip_ids = set()
    new_service_of = {}
    new_cal_dates_rows = []
    merged_run_trip_ids = set()

    skipped_no_overlap = 0
    for counter, run in enumerate(runs, start=1):
        keep = run[0]

        dates = None
        for t in run:
            d = service_dates.get(trip_service.get(t), set())
            dates = d if dates is None else (dates & d)

        # Adjacent trips in the run were each checked pairwise for calendar
        # overlap while building runs above, but that doesn't guarantee the
        # *whole* run (3+ trips) shares a common date. Merging here would
        # drop the constituent trips' own valid service dates in favor of a
        # trip that never actually runs, so skip it entirely instead.
        if not dates:
            skipped_no_overlap += 1
            continue

        pieces = [st_by_trip[keep]] + [st_by_trip[t].iloc[1:] for t in run[1:]]
        combined = pd.concat(pieces, ignore_index=True)
        combined["trip_id"] = keep
        combined["stop_sequence"] = range(len(combined))
        if "shape_dist_traveled" in combined.columns:
            combined["shape_dist_traveled"] = np.nan  # the distances of the pieces do not add up
        combined_frames.append(combined)
        drop_trip_ids.update(run[1:])
        merged_run_trip_ids.update(run)

        svc_id = f"BLOCK_SVC_{counter}"
        new_service_of[keep] = svc_id
        new_cal_dates_rows.extend(
            {"service_id": svc_id, "date": date.strftime("%Y%m%d"), "exception_type": 1}
            for date in dates
        )

        logger.info(f"  merged: {' + '.join(run[1:])} into {keep} "
              f"({' -> '.join(trip_agency.get(t, '?') for t in run)}) — "
              f"{len(dates)} active date(s), {len(combined)} stops total")

    if skipped_no_overlap:
        logger.info(f"merge_cross_agency_blocks: skipped {skipped_no_overlap} run(s) with no common active date "
              f"across the whole run.")

    feed["stop_times"] = pd.concat(
        [df_st[~df_st["trip_id"].isin(merged_run_trip_ids)]] + combined_frames,
        ignore_index=True,
    )

    df_trips = df_trips[~df_trips["trip_id"].isin(drop_trip_ids)].copy()
    df_trips["service_id"] = df_trips["trip_id"].map(new_service_of).fillna(df_trips["service_id"])
    if "shape_id" in df_trips.columns:  # the shape of the first trip only covers the first part of the merged one
        df_trips.loc[df_trips["trip_id"].isin(new_service_of), "shape_id"] = None
    feed["trips"] = df_trips

    if new_cal_dates_rows:
        df_new_cd = pd.DataFrame(new_cal_dates_rows)
        feed["calendar_dates"] = (
            pd.concat([feed["calendar_dates"], df_new_cd], ignore_index=True)
            if "calendar_dates" in feed else df_new_cd
        )

    if "frequencies" in feed:
        feed["frequencies"] = feed["frequencies"][~feed["frequencies"]["trip_id"].isin(drop_trip_ids)]

    logger.info(f"Merged {len(combined_frames)} cross-agency block run(s) into single trips — no fake transfer left behind.")

    return feed


def extend_swiss_trips_with_sncf(swiss_feed, sncf_feed, max_gap_s=600, min_gap_s=-60,
                                  min_match_stops=3, match_time_tol_s=180):
    """
    merge_cross_agency_blocks() fixes the split-journey problem *within* a
    single feed's own block_id chains. But some cross-border corridors are
    split across two independently-published feeds instead: the Swiss
    consolidated feed only models its own operator's portion (SBB, or an
    SNCF-attributed route embedded in the Swiss feed), while the
    standalone SNCF feed separately publishes the continuation across the
    border. Neither side's block_id spans the other feed, so this split is
    otherwise invisible to routing tools, which instead see
    bridge_cross_source_stations()'s artificial minimum-transfer treatment
    at the border stop.

    Detects boundary handoffs between a trip in swiss_feed and a trip in
    sncf_feed — same station (matched via extract_station_code, since the
    two feeds use different stop_id schemes), corresponding
    route_short_name (exact match, or one is "R" + the other), overlapping
    service calendar, and a departure/arrival gap within [min_gap_s,
    max_gap_s] — and extends the Swiss trip's stop_times with the SNCF
    trip's continuation on whichever dates the pairing is unambiguous.
    Some Swiss trips have more than one candidate SNCF continuation overall
    (e.g. SNCF publishes a normal daily variant and a rare schedule-
    exception variant departing the same boundary station at the same
    time, active on different, non-overlapping dates) — ambiguity is
    resolved per date rather than by rejecting the pairing feed-wide, so a
    trip still gets extended on the dates where exactly one candidate is
    actually active, and only the genuinely conflicting dates (both
    candidates active at once) are left unextended. A trip's active dates
    are split into an extended copy (new trip_id, fresh calendar_dates-only
    service_id) per distinct SNCF continuation used, plus — if any dates
    couldn't be resolved — the original Swiss trip kept for just those
    leftover dates. Same on the SNCF side: a consumed trip is dropped only
    for the dates actually spliced onto a Swiss trip; any leftover dates
    keep the original SNCF trip under a reduced calendar.

    The SNCF trip covering a Swiss trip's continuation isn't always a
    standalone segment starting exactly at the boundary station — some
    corridors (e.g. RL3/L3 to St-Gervais) are only ever published by SNCF
    as a single trip covering the *entire* physical journey, Swiss-side
    portion included. So a Swiss trip's boundary station is searched for
    anywhere within an SNCF trip's itinerary, not just at one of its ends —
    and in both directions: a Swiss trip can be missing its tail (exiting
    Switzerland — the SNCF continuation is appended after the Swiss trip's
    last stop) or missing its head (entering Switzerland — the SNCF prefix
    is prepended before the Swiss trip's first stop). Both are searched for
    independently; a trip could in principle get extended on both ends.

    Matching on a single shared station + time + name is a weak signal on
    its own — it can accidentally pair a Swiss trip ending at a station
    with an unrelated SNCF trip that simply happens to depart the same
    station a few minutes later (e.g. a vehicle turnaround into a new
    direction, not a continuation). So besides the boundary station,
    min_match_stops-1 more of the Swiss trip's stops adjacent to the
    boundary (station id and departure time, within match_time_tol_s) must
    also line up against the SNCF trip's stops on the matching side of the
    matched position — i.e. a whole subsequence of the schedule has to
    agree, not just one coincidental stop — before a candidate is even
    considered. As a further safety check, a candidate extension is
    rejected if the SNCF trip's spliced-in stops would revisit any station
    already on the Swiss trip's own itinerary (a genuine continuation only
    ever moves forward).

    Returns (swiss_feed, sncf_feed): swiss_feed with matched trips
    extended, sncf_feed with the consumed trips removed.
    """
    swiss_feed = copy_feed(swiss_feed)
    sncf_feed = copy_feed(sncf_feed)

    def trip_meta(feed):
        df_trips = feed["trips"]
        trip_agency = pd.Series("", index=df_trips["trip_id"], dtype=object)
        trip_short_name = pd.Series("", index=df_trips["trip_id"], dtype=object)
        if "routes" in feed and "agency" in feed:
            df_routes = feed["routes"][["route_id", "route_short_name", "agency_id"]].copy()
            df_routes["route_id"] = df_routes["route_id"].astype(str)
            df_routes["agency_id"] = df_routes["agency_id"].astype(str)
            df_agency = feed["agency"][["agency_id", "agency_name"]].copy()
            df_agency["agency_id"] = df_agency["agency_id"].astype(str)
            df_routes = df_routes.merge(df_agency, on="agency_id", how="left")
            route_agency = df_routes.set_index("route_id")["agency_name"]
            route_short = df_routes.set_index("route_id")["route_short_name"]
            trip_route = df_trips.set_index("trip_id")["route_id"].astype(str)
            trip_agency = trip_route.map(route_agency).fillna("")
            trip_short_name = trip_route.map(route_short).fillna("")
        return trip_agency, trip_short_name

    def tokens(feed):
        st = (
            feed["stop_times"][["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time"]]
            .sort_values(["trip_id", "stop_sequence"])
        )
        codes = st["stop_id"].astype(str).map(extract_station_code)
        st = st.assign(norm_id=codes.fillna(st["stop_id"].astype(str)))
        entries = st.groupby("trip_id", sort=False).apply(
            lambda g: list(zip(g["norm_id"], g["arrival_time"], g["departure_time"])), include_groups=False
        )
        return entries.to_dict()

    swiss_agency, swiss_name = trip_meta(swiss_feed)
    sncf_agency, sncf_name = trip_meta(sncf_feed)
    swiss_entries = tokens(swiss_feed)
    sncf_entries = tokens(sncf_feed)

    # Swiss route_short_name isn't always a specific line identifier — for
    # international/long-distance services (TGV Lyria, EuroCity, etc.) it's
    # often just a generic vehicle-type label shared across many unrelated
    # physical lines (e.g. "TGV" alone covers Lausanne-Paris, Genève-
    # Bellegarde, Zürich-Mulhouse...). Requiring it to equal (or be an
    # "R"-prefixed variant of) the SNCF train number is meaningless there
    # and blocks genuine continuations (e.g. Swiss "TGV" ending at Frasne
    # never gets stitched to SNCF's "612E" running the same physical train
    # on to Paris) — so these labels are treated as wildcards, relying
    # instead on the multi-stop trailing/leading match already required
    # below to confirm it's really the same physical train.
    GENERIC_ROUTE_LABELS = {"TGV", "EC", "IC", "ICE", "IR", "RE", "R", "TER", "EXT", "SN", "NJ", "RJ"}

    def names_correspond(a, b):
        a, b = str(a).strip().upper(), str(b).strip().upper()
        if not a or not b:
            return False
        if a in GENERIC_ROUTE_LABELS or b in GENERIC_ROUTE_LABELS:
            return True
        return a == b or a == "R" + b or b == "R" + a

    swiss_service_dates = expand_service_dates(swiss_feed).groupby("service_id")["date"].apply(set).to_dict()
    sncf_service_dates = expand_service_dates(sncf_feed).groupby("service_id")["date"].apply(set).to_dict()
    swiss_trip_service = swiss_feed["trips"].set_index("trip_id")["service_id"].astype(str)
    sncf_trip_service = sncf_feed["trips"].set_index("trip_id")["service_id"].astype(str)

    def overlapping_dates(t_swiss, t_sncf):
        return (
            swiss_service_dates.get(swiss_trip_service.get(t_swiss), set())
            & sncf_service_dates.get(sncf_trip_service.get(t_sncf), set())
        )

    # Every occurrence of a station within an SNCF trip's itinerary is a
    # candidate boundary position — not just one of the trip's ends.
    sncf_positions_by_station = {}
    for t, entry in sncf_entries.items():
        for idx, (code, _, _) in enumerate(entry):
            sncf_positions_by_station.setdefault(code, []).append((t, idx))

    candidates = []
    for t_swiss, entry in swiss_entries.items():
        if not entry:
            continue
        name_a = swiss_name.get(t_swiss, "")
        if not name_a:
            continue
        swiss_stations = {e[0] for e in entry}

        # --- append: Swiss trip is missing its tail (exiting Switzerland) ---
        code, arr_str, _ = entry[-1]
        arr_s = time_to_seconds(arr_str)
        for t_sncf, idx in sncf_positions_by_station.get(code, []):
            sncf_entry = sncf_entries[t_sncf]
            if idx >= len(sncf_entry) - 1:
                continue  # nothing after idx to extend with
            name_b = sncf_name.get(t_sncf, "")
            if not names_correspond(name_a, name_b):
                continue

            # Require a subset of stops adjacent to the boundary — not
            # just the boundary stop itself — to agree on station id and
            # departure time between the two trips' schedules. Falls back
            # to just the boundary stop when either trip is too short to
            # offer more (e.g. a genuine handoff at the SNCF trip's very
            # first stop, idx == 0).
            k = min(min_match_stops, len(entry), idx + 1)
            trailing_ok = True
            for o in range(1, k):
                swiss_stop, sncf_stop = entry[-1 - o], sncf_entry[idx - o]
                if swiss_stop[0] != sncf_stop[0]:
                    trailing_ok = False
                    break
                if abs(time_to_seconds(swiss_stop[2]) - time_to_seconds(sncf_stop[2])) > match_time_tol_s:
                    trailing_ok = False
                    break
            if not trailing_ok:
                continue

            gap = time_to_seconds(sncf_entry[idx][2]) - arr_s
            if not (min_gap_s <= gap <= max_gap_s):
                continue
            dates = overlapping_dates(t_swiss, t_sncf)
            if not dates:
                continue
            # Backtracking guard: reject if the SNCF continuation revisits
            # a station already on the Swiss trip's own itinerary.
            if swiss_stations & {e[0] for e in sncf_entry[idx + 1:]}:
                continue
            candidates.append((t_swiss, t_sncf, idx, dates, "append"))

        # --- prepend: Swiss trip is missing its head (entering Switzerland) ---
        code0, _, dep_str0 = entry[0]
        dep_s0 = time_to_seconds(dep_str0)
        for t_sncf, idx in sncf_positions_by_station.get(code0, []):
            if idx < 1:
                continue  # nothing before idx to prepend
            sncf_entry = sncf_entries[t_sncf]
            name_b = sncf_name.get(t_sncf, "")
            if not names_correspond(name_a, name_b):
                continue

            k = min(min_match_stops, len(entry), len(sncf_entry) - idx)
            leading_ok = True
            for o in range(1, k):
                swiss_stop, sncf_stop = entry[o], sncf_entry[idx + o]
                if swiss_stop[0] != sncf_stop[0]:
                    leading_ok = False
                    break
                if abs(time_to_seconds(swiss_stop[2]) - time_to_seconds(sncf_stop[2])) > match_time_tol_s:
                    leading_ok = False
                    break
            if not leading_ok:
                continue

            gap = dep_s0 - time_to_seconds(sncf_entry[idx][1])
            if not (min_gap_s <= gap <= max_gap_s):
                continue
            dates = overlapping_dates(t_swiss, t_sncf)
            if not dates:
                continue
            # Backtracking guard: reject if the SNCF prefix revisits a
            # station already on the Swiss trip's own itinerary.
            if swiss_stations & {e[0] for e in sncf_entry[:idx]}:
                continue
            candidates.append((t_swiss, t_sncf, idx, dates, "prepend"))

    if not candidates:
        logger.info("extend_swiss_trips_with_sncf: no Swiss<->SNCF boundary continuations found.")
        return swiss_feed, sncf_feed

    # Resolve ambiguity per date rather than feed-wide: for each (Swiss
    # trip, date), a candidate SNCF continuation only wins if it's the sole
    # one active that specific date. A candidate active on 150 dates but
    # contested on 3 of them still wins the other 147. An append and a
    # prepend candidate proposed for the same (Swiss trip, date) count as
    # two competing options too — a trip needing both ends resolved
    # simultaneously on the same date isn't handled, so it's left alone.
    date_candidates = {}
    for t_swiss, t_sncf, idx, dates, mode in candidates:
        for d in dates:
            date_candidates.setdefault((t_swiss, d), []).append((t_sncf, idx, mode))

    swiss_date_winner = {
        key: opts[0] for key, opts in date_candidates.items() if len(opts) == 1
    }

    # An SNCF trip can only be the genuine continuation/prefix of one Swiss
    # trip on a given date — if two different Swiss trips both (each
    # unambiguously on their own side) claim the same SNCF trip for the
    # same date, that's itself a conflict, so drop both.
    sncf_date_claimants = {}
    for (t_swiss, d), (t_sncf, idx, mode) in swiss_date_winner.items():
        sncf_date_claimants.setdefault((t_sncf, d), []).append(t_swiss)
    contested_sncf_dates = {key for key, claimants in sncf_date_claimants.items() if len(claimants) > 1}
    if contested_sncf_dates:
        swiss_date_winner = {
            key: (t_sncf, idx, mode)
            for key, (t_sncf, idx, mode) in swiss_date_winner.items()
            if (t_sncf, key[1]) not in contested_sncf_dates
        }

    if not swiss_date_winner:
        logger.info("extend_swiss_trips_with_sncf: no Swiss<->SNCF boundary continuations were unambiguous on any date.")
        return swiss_feed, sncf_feed

    groups = {}
    for (t_swiss, d), (t_sncf, idx, mode) in swiss_date_winner.items():
        groups.setdefault((t_swiss, t_sncf, idx, mode), set()).add(d)

    involved_swiss = {key[0] for key in groups}
    involved_sncf = {key[1] for key in groups}

    swiss_st = swiss_feed["stop_times"]
    sncf_st = sncf_feed["stop_times"]
    swiss_st_by_trip = {
        tid: g.sort_values("stop_sequence")
        for tid, g in swiss_st[swiss_st["trip_id"].isin(involved_swiss)].groupby("trip_id")
    }
    sncf_st_by_trip = {
        tid: g.sort_values("stop_sequence")
        for tid, g in sncf_st[sncf_st["trip_id"].isin(involved_sncf)].groupby("trip_id")
    }
    swiss_trip_row = swiss_feed["trips"].set_index("trip_id", drop=False)

    combined_frames = []
    new_trip_rows = []
    new_cal_dates_rows = []
    consumed_dates_by_swiss = {}
    consumed_dates_by_sncf = {}

    for counter, ((t_swiss, t_sncf, idx, mode), dates) in enumerate(groups.items(), start=1):
        new_tid = f"{t_swiss}_EXT{counter}"
        if mode == "append":
            combined = pd.concat([swiss_st_by_trip[t_swiss], sncf_st_by_trip[t_sncf].iloc[idx + 1:]], ignore_index=True)
        else:
            combined = pd.concat([sncf_st_by_trip[t_sncf].iloc[:idx], swiss_st_by_trip[t_swiss]], ignore_index=True)
        combined["trip_id"] = new_tid
        combined["stop_sequence"] = range(len(combined))
        if "shape_dist_traveled" in combined.columns:
            combined["shape_dist_traveled"] = np.nan
        combined_frames.append(combined)

        svc_id = f"EXT_SVC_{counter}"
        new_cal_dates_rows.extend(
            {"service_id": svc_id, "date": date.strftime("%Y%m%d"), "exception_type": 1}
            for date in dates
        )

        new_row = swiss_trip_row.loc[t_swiss].copy()
        new_row["trip_id"] = new_tid
        new_row["service_id"] = svc_id
        if "shape_id" in new_row.index:  # the Swiss shape does not cover the SNCF part
            new_row["shape_id"] = None
        new_trip_rows.append(new_row)

        consumed_dates_by_swiss.setdefault(t_swiss, set()).update(dates)
        consumed_dates_by_sncf.setdefault(t_sncf, set()).update(dates)

        arrow = f"{t_swiss} -> {new_tid}" if mode == "append" else f"{new_tid} <- {t_swiss}"
        logger.info(f"  {mode}ed: {arrow} ({swiss_name.get(t_swiss, '?')}, {swiss_agency.get(t_swiss, '?')}) "
              f"+ {t_sncf} ({sncf_name.get(t_sncf, '?')}, {sncf_agency.get(t_sncf, '?')}) — "
              f"{len(dates)} active date(s), {len(combined)} stops total")

    # Swiss side: trips fully consumed (no leftover dates) are dropped
    # outright; trips with some unresolved/leftover dates keep their
    # original trip_id and stop_times but get a calendar restricted to
    # just those dates.
    swiss_trip_ids_to_drop = set()
    updated_swiss_service = {}
    leftover_counter = 0
    for t_swiss in involved_swiss:
        total_dates = swiss_service_dates.get(swiss_trip_service.get(t_swiss), set())
        leftover = total_dates - consumed_dates_by_swiss.get(t_swiss, set())
        if not leftover:
            swiss_trip_ids_to_drop.add(t_swiss)
        else:
            leftover_counter += 1
            svc_id = f"EXT_LEFTOVER_SVC_{leftover_counter}"
            updated_swiss_service[t_swiss] = svc_id
            new_cal_dates_rows.extend(
                {"service_id": svc_id, "date": date.strftime("%Y%m%d"), "exception_type": 1}
                for date in leftover
            )

    df_swiss_trips = swiss_feed["trips"].copy()
    df_swiss_trips = df_swiss_trips[~df_swiss_trips["trip_id"].isin(swiss_trip_ids_to_drop)]
    if updated_swiss_service:
        df_swiss_trips["service_id"] = df_swiss_trips["trip_id"].map(updated_swiss_service).fillna(df_swiss_trips["service_id"])
    if new_trip_rows:
        df_swiss_trips = pd.concat([df_swiss_trips, pd.DataFrame(new_trip_rows)], ignore_index=True)
    swiss_feed["trips"] = df_swiss_trips

    swiss_feed["stop_times"] = pd.concat(
        [swiss_st[~swiss_st["trip_id"].isin(swiss_trip_ids_to_drop)]] + combined_frames,
        ignore_index=True,
    )
    if new_cal_dates_rows:
        df_new_cd = pd.DataFrame(new_cal_dates_rows)
        swiss_feed["calendar_dates"] = (
            pd.concat([swiss_feed["calendar_dates"], df_new_cd], ignore_index=True)
            if "calendar_dates" in swiss_feed else df_new_cd
        )

    # SNCF side: same leftover-dates treatment as the Swiss side above.
    sncf_trip_ids_to_drop = set()
    updated_sncf_service = {}
    sncf_leftover_cal_rows = []
    sncf_leftover_counter = 0
    for t_sncf in involved_sncf:
        total_dates = sncf_service_dates.get(sncf_trip_service.get(t_sncf), set())
        leftover = total_dates - consumed_dates_by_sncf.get(t_sncf, set())
        if not leftover:
            sncf_trip_ids_to_drop.add(t_sncf)
        else:
            sncf_leftover_counter += 1
            svc_id = f"SNCF_LEFTOVER_SVC_{sncf_leftover_counter}"
            updated_sncf_service[t_sncf] = svc_id
            sncf_leftover_cal_rows.extend(
                {"service_id": svc_id, "date": date.strftime("%Y%m%d"), "exception_type": 1}
                for date in leftover
            )

    df_sncf_trips = sncf_feed["trips"].copy()
    df_sncf_trips = df_sncf_trips[~df_sncf_trips["trip_id"].isin(sncf_trip_ids_to_drop)]
    if updated_sncf_service:
        df_sncf_trips["service_id"] = df_sncf_trips["trip_id"].map(updated_sncf_service).fillna(df_sncf_trips["service_id"])
    sncf_feed["trips"] = df_sncf_trips
    sncf_feed["stop_times"] = sncf_st[~sncf_st["trip_id"].isin(sncf_trip_ids_to_drop)]
    if sncf_leftover_cal_rows:
        df_new_cd = pd.DataFrame(sncf_leftover_cal_rows)
        sncf_feed["calendar_dates"] = (
            pd.concat([sncf_feed["calendar_dates"], df_new_cd], ignore_index=True)
            if "calendar_dates" in sncf_feed else df_new_cd
        )
    if "frequencies" in sncf_feed:
        sncf_feed["frequencies"] = sncf_feed["frequencies"][~sncf_feed["frequencies"]["trip_id"].isin(sncf_trip_ids_to_drop)]

    logger.info(f"extend_swiss_trips_with_sncf: extended {len(involved_swiss)} Swiss trip(s) into "
          f"{len(groups)} date-scoped continuation(s).")

    return swiss_feed, sncf_feed


def drop_duplicate_sncf_trips(swiss_feed, sncf_feed, max_stop_time_diff_s=180):
    """
    Some Léman Express trips are published in full by *both* the Swiss
    consolidated feed and the standalone SNCF feed independently — the
    Swiss trip already covers the entire physical journey end-to-end (no
    block_id split, nothing for extend_swiss_trips_with_sncf() to stitch
    on), while the SNCF feed separately re-publishes the exact same train
    under its own trip_id. This isn't a split journey (one trip covering
    part of a route, needing the other feed's continuation) — it's the
    same journey reported twice — so instead of extending anything, the
    redundant SNCF trip is simply dropped.

    Matches trips whose entire normalized station sequence (via
    extract_station_code) is identical and whose per-stop times are all
    within max_stop_time_diff_s of each other — not required to be
    byte-identical, since the two publishers occasionally round a couple
    of intermediate stops a minute or two apart even for the same physical
    train. The same schedule slot can be published as several Swiss
    trip_ids (one per calendar variant — weekday, Saturday, Sunday — each
    its own service_id), so a station-sequence match alone isn't unique.
    As in extend_swiss_trips_with_sncf(), ambiguity is resolved per date
    rather than feed-wide: a candidate pairing only wins a given date if
    it's the sole one active both on the Swiss side and the SNCF side that
    date, so an SNCF trip whose calendar spans several Swiss calendar
    variants still gets matched correctly on each variant's own dates. An
    SNCF trip is dropped entirely once every one of its active dates has
    been matched; if only some dates matched (the rest genuinely
    ambiguous), it's kept under a reduced calendar covering just the
    leftover dates. The Swiss side is left untouched.

    Returns (swiss_feed, sncf_feed).
    """
    swiss_feed = copy_feed(swiss_feed)
    sncf_feed = copy_feed(sncf_feed)

    def trip_stations_and_times(feed):
        st = (
            feed["stop_times"][["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time"]]
            .sort_values(["trip_id", "stop_sequence"])
        )
        codes = st["stop_id"].astype(str).map(extract_station_code)
        st = st.assign(norm_id=codes.fillna(st["stop_id"].astype(str)))
        st = st.assign(dep_s=st["departure_time"].map(time_to_seconds))

        stations = {}
        times = {}
        for tid, g in st.groupby("trip_id", sort=False):
            stations[tid] = tuple(g["norm_id"])
            times[tid] = np.array(g["dep_s"])
        return stations, times

    swiss_stations, swiss_times = trip_stations_and_times(swiss_feed)
    sncf_stations, sncf_times = trip_stations_and_times(sncf_feed)

    swiss_by_seq = {}
    for tid, seq in swiss_stations.items():
        swiss_by_seq.setdefault(seq, []).append(tid)
    sncf_by_seq = {}
    for tid, seq in sncf_stations.items():
        sncf_by_seq.setdefault(seq, []).append(tid)

    swiss_trip_service = swiss_feed["trips"].set_index("trip_id")["service_id"].astype(str)
    sncf_trip_service = sncf_feed["trips"].set_index("trip_id")["service_id"].astype(str)
    swiss_service_dates = expand_service_dates(swiss_feed).groupby("service_id")["date"].apply(set).to_dict()
    sncf_service_dates = expand_service_dates(sncf_feed).groupby("service_id")["date"].apply(set).to_dict()

    # Gather, per (SNCF trip, date), every Swiss trip in the same exact
    # signature group that's also within the time tolerance and active
    # that date.
    date_candidates = {}
    for seq, swiss_tids in swiss_by_seq.items():
        sncf_tids = sncf_by_seq.get(seq)
        if not sncf_tids:
            continue

        for t_sncf in sncf_tids:
            sncf_dates = sncf_service_dates.get(sncf_trip_service.get(t_sncf), set())
            for t_swiss in swiss_tids:
                overlap = swiss_service_dates.get(swiss_trip_service.get(t_swiss), set()) & sncf_dates
                if not overlap:
                    continue
                max_diff = np.abs(swiss_times[t_swiss] - sncf_times[t_sncf]).max()
                if max_diff > max_stop_time_diff_s:
                    continue
                for d in overlap:
                    date_candidates.setdefault((t_sncf, d), []).append(t_swiss)

    sncf_date_winner = {
        key: cands[0] for key, cands in date_candidates.items() if len(cands) == 1
    }

    # A Swiss trip can only be duplicated by one SNCF trip on a given date —
    # if two different SNCF trips both (each unambiguously on their own
    # side) claim the same Swiss trip for the same date, that's itself a
    # conflict, so drop both.
    swiss_date_claimants = {}
    for (t_sncf, d), t_swiss in sncf_date_winner.items():
        swiss_date_claimants.setdefault((t_swiss, d), []).append(t_sncf)
    contested_swiss_dates = {key for key, claimants in swiss_date_claimants.items() if len(claimants) > 1}
    if contested_swiss_dates:
        sncf_date_winner = {
            key: t_swiss
            for key, t_swiss in sncf_date_winner.items()
            if (t_swiss, key[1]) not in contested_swiss_dates
        }

    if not sncf_date_winner:
        logger.info("drop_duplicate_sncf_trips: no fully-duplicated Swiss<->SNCF trips found.")
        return swiss_feed, sncf_feed

    consumed_dates_by_sncf = {}
    matched_swiss_of = {}
    for (t_sncf, d), t_swiss in sncf_date_winner.items():
        consumed_dates_by_sncf.setdefault(t_sncf, set()).add(d)
        matched_swiss_of.setdefault(t_sncf, set()).add(t_swiss)

    dup_sncf_trip_ids = set()
    updated_sncf_service = {}
    leftover_cal_rows = []
    leftover_counter = 0
    for t_sncf, consumed in consumed_dates_by_sncf.items():
        total_dates = sncf_service_dates.get(sncf_trip_service.get(t_sncf), set())
        leftover = total_dates - consumed
        matched = ", ".join(sorted(matched_swiss_of[t_sncf]))
        if not leftover:
            dup_sncf_trip_ids.add(t_sncf)
            logger.info(f"  duplicate: dropping SNCF {t_sncf} — already fully covered by Swiss trip(s) {matched} "
                  f"({len(consumed)} shared active date(s))")
        else:
            leftover_counter += 1
            svc_id = f"SNCF_DUP_LEFTOVER_SVC_{leftover_counter}"
            updated_sncf_service[t_sncf] = svc_id
            leftover_cal_rows.extend(
                {"service_id": svc_id, "date": date.strftime("%Y%m%d"), "exception_type": 1}
                for date in leftover
            )
            logger.info(f"  duplicate: SNCF {t_sncf} covered by Swiss trip(s) {matched} on {len(consumed)} date(s) — "
                  f"kept for {len(leftover)} leftover, unmatched date(s)")

    df_sncf_trips = sncf_feed["trips"].copy()
    df_sncf_trips = df_sncf_trips[~df_sncf_trips["trip_id"].isin(dup_sncf_trip_ids)]
    if updated_sncf_service:
        df_sncf_trips["service_id"] = df_sncf_trips["trip_id"].map(updated_sncf_service).fillna(df_sncf_trips["service_id"])
    sncf_feed["trips"] = df_sncf_trips
    sncf_feed["stop_times"] = sncf_feed["stop_times"][~sncf_feed["stop_times"]["trip_id"].isin(dup_sncf_trip_ids)]
    if leftover_cal_rows:
        df_new_cd = pd.DataFrame(leftover_cal_rows)
        sncf_feed["calendar_dates"] = (
            pd.concat([sncf_feed["calendar_dates"], df_new_cd], ignore_index=True)
            if "calendar_dates" in sncf_feed else df_new_cd
        )
    if "frequencies" in sncf_feed:
        sncf_feed["frequencies"] = sncf_feed["frequencies"][~sncf_feed["frequencies"]["trip_id"].isin(dup_sncf_trip_ids)]

    logger.info(f"drop_duplicate_sncf_trips: dropped {len(dup_sncf_trip_ids)} SNCF trip(s) fully duplicated by the "
          f"Swiss feed" + (f", trimmed {leftover_counter} to their leftover dates" if leftover_counter else "") + ".")
    return swiss_feed, sncf_feed


def rename_swiss_leman_routes(feed):
    """
    The Swiss feed labels the Léman Express lines "RL1".."RL7" (an
    internal "Regio-something-Léman" convention), while SNCF and the
    official regional branding use just "L1".."L7". The "R" prefix is
    also exactly what names_correspond() in extend_swiss_trips_with_sncf()
    strips to match a Swiss trip against its SNCF counterpart — so by the
    time this runs (after all cross-source merging is done), it's purely
    cosmetic to drop it too, for a single consistent line name across the
    merged feed regardless of which source a given trip came from.

    Restricted to SBB-attributed routes matching exactly "RL" + digits, to
    avoid renaming an unrelated "RL"-prefixed route from a different
    agency/region that happens to share the pattern.
    """
    feed = copy_feed(feed)
    df_routes = feed["routes"].copy()

    sbb_agency_ids = set()
    if "agency" in feed:
        df_agency = feed["agency"]
        sbb_agency_ids = set(
            df_agency.loc[
                df_agency["agency_name"].astype(str).str.contains("Schweizerische Bundesbahnen", case=False, na=False),
                "agency_id",
            ].astype(str)
        )

    mask = (
        df_routes["route_short_name"].astype(str).str.match(r"^RL\d+$", na=False)
        & df_routes["agency_id"].astype(str).isin(sbb_agency_ids)
    )

    if not mask.any():
        logger.info("rename_swiss_leman_routes: no matching RLn routes found.")
        return feed

    df_routes.loc[mask, "route_short_name"] = df_routes.loc[mask, "route_short_name"].str.replace(r"^RL", "L", regex=True)
    feed["routes"] = df_routes
    logger.info(f"rename_swiss_leman_routes: renamed {int(mask.sum())} SBB route(s) from RLn to Ln.")
    return feed


def get_active_service_ids(feed, date):
    """
    Return the set of service_ids (as str) active on the given date,
    combining calendar.txt (weekday flag within start/end range) with
    calendar_dates.txt exceptions (type 1 = added, type 2 = removed).
    """
    date = pd.Timestamp(date)
    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    active = set()

    if "calendar" in feed:
        df_cal = feed["calendar"]
        start = pd.to_datetime(df_cal["start_date"], format="%Y%m%d")
        end = pd.to_datetime(df_cal["end_date"], format="%Y%m%d")
        weekday_col = weekday_cols[date.weekday()]

        mask = (start <= date) & (end >= date) & (df_cal[weekday_col].astype(int) == 1)
        active |= set(df_cal.loc[mask, "service_id"].astype(str))

    if "calendar_dates" in feed:
        df_cd = feed["calendar_dates"].copy()
        df_cd["service_id"] = df_cd["service_id"].astype(str)
        df_cd["date"] = pd.to_datetime(df_cd["date"].astype(str), format="%Y%m%d")

        same_day = df_cd[df_cd["date"] == date]
        active |= set(same_day.loc[same_day["exception_type"] == 1, "service_id"])
        active -= set(same_day.loc[same_day["exception_type"] == 2, "service_id"])

    return active


def filter_feed_to_date(feed, date):
    """
    Keep only the services (and their trips/stop_times/frequencies) that are
    active on the given date. Used to build a "typical day" feed, e.g. for
    accessibility computations that reference a single representative date.
    """
    feed = copy_feed(feed)

    active_service_ids = get_active_service_ids(feed, date)
    logger.info(f"Filtering feed to services active on {pd.Timestamp(date).date()}: "
          f"{len(active_service_ids)} service_id(s) kept")

    if not active_service_ids:
        logger.info("WARNING: no services are active on this date, feed will be empty.")

    if "calendar" in feed:
        feed["calendar"] = feed["calendar"][feed["calendar"]["service_id"].astype(str).isin(active_service_ids)]

    if "calendar_dates" in feed:
        feed["calendar_dates"] = feed["calendar_dates"][feed["calendar_dates"]["service_id"].astype(str).isin(active_service_ids)]

    feed["trips"] = feed["trips"][feed["trips"]["service_id"].astype(str).isin(active_service_ids)]
    remaining_trip_ids = set(feed["trips"]["trip_id"].astype(str))

    feed["stop_times"] = feed["stop_times"][feed["stop_times"]["trip_id"].astype(str).isin(remaining_trip_ids)]

    if "frequencies" in feed:
        feed["frequencies"] = feed["frequencies"][feed["frequencies"]["trip_id"].astype(str).isin(remaining_trip_ids)]

    return feed






# --- Replacing the lines of an agency of the Swiss feed by the operator's own feed ----------------------------------

def _str_ids(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = _as_id_string(df[column])
    return df


def map_operator_stops(operator_stops, swiss_stops, max_distance_m = 150):
    """{operator stop_id: Swiss stop_id} for the stops of an operator feed that uses the SLOIDs (ch:1:sloid:...), in this
    order: the Swiss stop whose original_stop_id is that SLOID, the Swiss stop of the same station (didok) and platform
    code, the closest Swiss stop of the station within max_distance_m. Stops without a match are not in the result."""
    swiss = swiss_stops.copy()
    swiss["stop_id"] = swiss["stop_id"].astype(str)
    swiss["_code"] = swiss["stop_id"].str.split(":").str[0]

    by_original = {}
    if "original_stop_id" in swiss.columns:
        by_original = dict(zip(swiss["original_stop_id"].astype(str), swiss["stop_id"]))
    swiss_platforms = swiss[swiss["location_type"] == 0]
    swiss_ids = set(swiss["stop_id"])

    mapping, methods = {}, {"original_stop_id": 0, "platform": 0, "closest": 0, "station": 0}
    for row in operator_stops.itertuples():
        stop_id = str(row.stop_id)
        if stop_id in by_original:
            mapping[stop_id] = by_original[stop_id]
            methods["original_stop_id"] += 1
            continue

        didok = str(getattr(row, "didok", "") or "").split(".")[0]
        if row.location_type == 1:
            if f"Parent{didok}" in swiss_ids:
                mapping[stop_id] = f"Parent{didok}"
                methods["station"] += 1
            continue

        candidates = swiss_platforms[swiss_platforms["_code"] == didok]
        if candidates.empty:
            continue

        platform = str(getattr(row, "platform_code", "") or "")
        if platform and "platform_code" in candidates.columns:
            same = candidates[candidates["platform_code"].astype(str) == platform]
            if len(same):
                mapping[stop_id] = same["stop_id"].iloc[0]
                methods["platform"] += 1
                continue

        distances = haversine_m(row.stop_lat, row.stop_lon, candidates["stop_lat"].values, candidates["stop_lon"].values)
        if distances.min() <= max_distance_m:
            mapping[stop_id] = candidates["stop_id"].iloc[int(np.argmin(distances))]
            methods["closest"] += 1

    logger.info(f"  {len(mapping)}/{len(operator_stops)} operator stops matched to Swiss stops ({methods})")
    return mapping


def replace_agency_lines(swiss_feed, operator_feed, agency_id = "881", service_prefix = "TPG_", removed_lines = ()):
    """Replaces the lines of agency_id in the Swiss feed by the ones of the operator feed (e.g. the TPG own feed).

    - the routes, trips and stop times of the agency are removed from the Swiss feed, except the lines (short names) that
      the operator feed does not have, which are kept unless they are in removed_lines (e.g. the TPG airport lines A1 to
      A6, which the operator asked to remove);
    - the operator stops are matched to the Swiss stops (see map_operator_stops), so that the other operators of the same
      stops and the analyses keep the Swiss stop ids; the unmatched ones are added with their own ids;
    - the operator feed describes one typical working day (one service valid on a single date): its services are made
      valid every Monday to Friday of the period of the Swiss feed;
    - the operator transfers are added on the matched stops."""
    swiss = copy_feed(swiss_feed)
    operator = copy_feed(operator_feed)

    for feed in (swiss, operator):
        _str_ids(feed["routes"], ["route_id", "agency_id"])
        _str_ids(feed["trips"], ["route_id", "trip_id", "service_id"])
        _str_ids(feed["stop_times"], ["trip_id", "stop_id"])
        _str_ids(feed["stops"], ["stop_id", "parent_station"])
        if "transfers" in feed:
            _str_ids(feed["transfers"], ["from_stop_id", "to_stop_id", "from_route_id", "to_route_id", "from_trip_id", "to_trip_id"])

    # 1) the lines of the agency in the Swiss feed, the ones replaced
    agency_routes = swiss["routes"][swiss["routes"]["agency_id"] == agency_id]
    operator_names = set(operator["routes"]["route_short_name"].astype(str))
    removed_names = set(map(str, removed_lines)) - operator_names
    agency_names = set(agency_routes["route_short_name"].astype(str))
    replaced = agency_routes[agency_routes["route_short_name"].astype(str).isin(operator_names | removed_names)]
    kept = sorted(agency_names - operator_names - removed_names)
    added = sorted(operator_names - agency_names)
    logger.info(f"Replacing {len(replaced)} routes of agency {agency_id} by the operator feed "
                f"({operator['routes']['route_id'].nunique()} routes); kept from the Swiss feed: {kept}; new: {added}; "
                f"removed: {sorted(removed_names & agency_names)}")

    replaced_routes = set(replaced["route_id"])
    removed_trips = set(swiss["trips"].loc[swiss["trips"]["route_id"].isin(replaced_routes), "trip_id"])
    swiss["routes"] = swiss["routes"][~swiss["routes"]["route_id"].isin(replaced_routes)]
    swiss["trips"] = swiss["trips"][~swiss["trips"]["trip_id"].isin(removed_trips)]
    swiss["stop_times"] = swiss["stop_times"][~swiss["stop_times"]["trip_id"].isin(removed_trips)]
    if "frequencies" in swiss:
        swiss["frequencies"] = swiss["frequencies"][~swiss["frequencies"]["trip_id"].isin(removed_trips)]
    if "transfers" in swiss:
        t = swiss["transfers"]
        specific = pd.Series(False, index = t.index)
        for column, ids in (("from_route_id", replaced_routes), ("to_route_id", replaced_routes), ("from_trip_id", removed_trips), ("to_trip_id", removed_trips)):
            if column in t.columns:
                specific |= t[column].isin(ids)
        swiss["transfers"] = t[~specific]
    logger.info(f"  {len(removed_trips)} Swiss trips removed")

    # 2) the stops
    mapping = map_operator_stops(operator["stops"], swiss["stops"])
    unmatched = operator["stops"][~operator["stops"]["stop_id"].isin(mapping)].copy()
    unmatched["parent_station"] = unmatched["parent_station"].map(lambda p: mapping.get(p, p) if p else p)
    if len(unmatched):
        logger.info(f"  {len(unmatched)} operator stops without a Swiss stop, added with their own id: "
                    f"{sorted(set(unmatched['stop_name']))[:10]}")
    remap = lambda ids: ids.map(lambda s: mapping.get(s, s))
    operator["stop_times"]["stop_id"] = remap(operator["stop_times"]["stop_id"])

    # 3) the calendar: every Monday to Friday of the Swiss feed period
    starts, ends = [], []
    if "calendar" in swiss:
        starts.append(swiss["calendar"]["start_date"].astype(str).min()); ends.append(swiss["calendar"]["end_date"].astype(str).max())
    if "calendar_dates" in swiss:
        dates = swiss["calendar_dates"]["date"].astype(str)
        starts.append(dates.min()); ends.append(dates.max())
    services = sorted(set(operator["trips"]["service_id"]))
    calendar = pd.DataFrame({
        "service_id": [service_prefix + s for s in services],
        "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
        "start_date": int(min(starts)), "end_date": int(max(ends)),
    })
    operator["trips"]["service_id"] = service_prefix + operator["trips"]["service_id"]
    logger.info(f"  operator services {services} made valid every Monday to Friday from {min(starts)} to {max(ends)}")

    # 4) everything into the Swiss feed
    trip_columns = [c for c in operator["trips"].columns if c in swiss["trips"].columns]
    swiss["routes"] = pd.concat([swiss["routes"], operator["routes"]], ignore_index = True)
    swiss["trips"] = pd.concat([swiss["trips"], operator["trips"][trip_columns]], ignore_index = True)
    swiss["stop_times"] = pd.concat([swiss["stop_times"], operator["stop_times"]], ignore_index = True)
    swiss["calendar"] = pd.concat([swiss["calendar"], calendar], ignore_index = True) if "calendar" in swiss else calendar
    swiss["stops"] = pd.concat([swiss["stops"], unmatched[[c for c in unmatched.columns if c in swiss["stops"].columns]]], ignore_index = True)
    if "shapes" in operator:
        swiss["shapes"] = pd.concat([swiss["shapes"], operator["shapes"]], ignore_index = True) if "shapes" in swiss else operator["shapes"]
        logger.info(f"  {operator['shapes']['shape_id'].nunique()} operator shapes added")

    if "transfers" in operator:
        t = operator["transfers"].copy()
        t["from_stop_id"], t["to_stop_id"] = remap(t["from_stop_id"]), remap(t["to_stop_id"])
        t = t[t["from_stop_id"] != t["to_stop_id"]]
        existing = set(zip(swiss["transfers"]["from_stop_id"], swiss["transfers"]["to_stop_id"])) if "transfers" in swiss else set()
        t = t[[pair not in existing for pair in zip(t["from_stop_id"], t["to_stop_id"])]]
        swiss["transfers"] = pd.concat([swiss.get("transfers", pd.DataFrame()), t], ignore_index = True)
        logger.info(f"  {len(t)} operator transfers added")

    return swiss


def drop_unused_shapes(feed):
    """Removes the shapes that no trip uses any more (trips removed or merged), and the shape ids of the trips that are
    not in shapes.txt."""
    if "shapes" not in feed:
        if "shape_id" in feed["trips"]:
            feed["trips"]["shape_id"] = None
        return feed

    used = set(feed["trips"]["shape_id"].dropna().astype(str))
    before = feed["shapes"]["shape_id"].nunique()
    feed["shapes"] = feed["shapes"][feed["shapes"]["shape_id"].astype(str).isin(used)]
    defined = set(feed["shapes"]["shape_id"].astype(str))
    feed["trips"].loc[~feed["trips"]["shape_id"].astype(str).isin(defined), "shape_id"] = None
    logger.info(f"Shapes: {feed['shapes']['shape_id'].nunique()}/{before} used by the trips")
    if feed["shapes"].empty:
        del feed["shapes"]
    return feed
