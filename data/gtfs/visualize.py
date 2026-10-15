import json
import logging
import os
import re
import zipfile
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import data.gtfs.utils as gtfs_utils
from data.gtfs import visualize_pages

logger = logging.getLogger("synpp")

# Normalized modes (basic GTFS route types and the extended ones) and their colors, the same in every map
MODES = ["Rail", "Metro", "Tram", "Bus", "Boat", "Funicular", "Cable car", "Other"]
MODE_COLORS = {
    "Rail": "#33508f", "Metro": "#8a4fa8", "Tram": "#c2571a", "Bus": "#2a9d8f",
    "Boat": "#1f8fbf", "Funicular": "#6b8e23", "Cable car": "#8d6e63", "Other": "#888888",
}
SOURCE_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def mode_of(route_type):
    t = int(route_type)
    basic = {0: "Tram", 1: "Metro", 2: "Rail", 3: "Bus", 4: "Boat", 5: "Cable car", 6: "Cable car", 7: "Funicular", 11: "Bus", 12: "Rail"}
    if t in basic:
        return basic[t]
    for low, high, mode in ((100, 199, "Rail"), (200, 299, "Bus"), (400, 499, "Metro"), (700, 899, "Bus"), (900, 999, "Tram"),
                            (1000, 1099, "Boat"), (1300, 1399, "Cable car"), (1400, 1499, "Funicular")):
        if low <= t <= high:
            return mode
    return "Other"


def time_to_seconds(values):
    parts = values.astype(str).str.split(":", expand = True)
    return parts[0].astype(int) * 3600 + parts[1].astype(int) * 60 + parts[2].astype(int)


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between paired arrays of coordinates."""
    r = 6371000
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def extract_station_code(stop_id):
    """Normalized UIC/DIDOK station code of a stop id (same logic as data.gtfs.utils), None if there is none."""
    matches = re.findall(r"\d{7,8}", str(stop_id))
    if not matches:
        return None
    code = matches[-1]
    return code[:7] if len(code) == 8 else code


# ---------------------------------------------------------------------------
# Reading the feed
# ---------------------------------------------------------------------------

class Feed:
    """The tables of a GTFS feed (a folder of .txt files or a zip), read once and shared by the maps. With a date, only
    the trips active on that date are kept."""

    def __init__(self, path, date = None, diagnostics_path = None):
        self.path = path
        self.date = date
        self.diagnostics_path = diagnostics_path
        self._cache = {}

    def table(self, name, usecols = None, optional = False, chunk_filter = None):
        key = (name, tuple(usecols) if usecols else None)
        if key in self._cache:
            return self._cache[key]

        def read(handle):
            reader = pd.read_csv(handle, dtype = str, usecols = (lambda c: c in usecols) if usecols else None,
                                 keep_default_na = False, na_values = [""], chunksize = 2_000_000)
            parts = [chunk_filter(chunk) if chunk_filter else chunk for chunk in reader]
            return pd.concat(parts, ignore_index = True) if parts else pd.DataFrame(columns = usecols or [])

        if os.path.isdir(self.path):
            file = os.path.join(self.path, f"{name}.txt")
            if not os.path.exists(file):
                if optional:
                    return None
                raise FileNotFoundError(file)
            df = read(file)
        else:
            with zipfile.ZipFile(self.path) as z:
                if f"{name}.txt" not in z.namelist():
                    if optional:
                        return None
                    raise FileNotFoundError(f"{name}.txt in {self.path}")
                with z.open(f"{name}.txt") as f:
                    df = read(f)

        self._cache[key] = df
        return df

    @property
    def stops(self):
        if "stops" not in self._cache:
            stops = self.table("stops", ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"])
            stops = stops.dropna(subset = ["stop_lat", "stop_lon"]).copy()
            stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
            self._cache["stops"] = stops
        return self._cache["stops"]

    @property
    def routes(self):
        if "routes" not in self._cache:
            routes = self.table("routes", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
            agency = self.table("agency", ["agency_id", "agency_name"], optional = True)
            if agency is not None and "agency_id" in routes.columns:
                routes = routes.merge(agency.drop_duplicates("agency_id"), on = "agency_id", how = "left")
            if "agency_name" not in routes.columns:
                routes["agency_name"] = ""
            routes["agency_name"] = routes["agency_name"].fillna("")
            routes["name"] = routes["route_short_name"].fillna(routes["route_long_name"]).fillna(routes["route_id"])
            routes["mode"] = routes["route_type"].fillna("3").map(mode_of)
            sources = self.sources("routes")
            routes["source"] = routes["route_id"].map(sources).fillna("") if sources else ""
            self._cache["routes"] = routes
        return self._cache["routes"]

    def sources(self, slot):
        """{id: source file} written by data.gtfs.cleaned, empty when it is not there."""
        if not self.diagnostics_path:
            return {}
        path = os.path.join(self.diagnostics_path, f"{slot}_sources.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path, dtype = str, keep_default_na = False)
        return dict(zip(df.iloc[:, 0], df["source"]))

    @property
    def trips(self):
        if "trips" not in self._cache:
            trips = self.table("trips", ["route_id", "service_id", "trip_id", "direction_id", "shape_id", "trip_headsign"])
            if self.date:
                feed = {}
                for name in ("calendar", "calendar_dates"):
                    table = self.table(name, optional = True)
                    if table is not None:
                        feed[name] = table.astype({"exception_type": int}) if name == "calendar_dates" else table
                active = gtfs_utils.get_active_service_ids(feed, pd.Timestamp(self.date))
                trips = trips[trips["service_id"].isin(active)]
                logger.info(f"  {len(trips)} trips active on {self.date}")
            self._cache["trips"] = trips.reset_index(drop = True)
        return self._cache["trips"]

    @property
    def stop_times(self):
        """The stop times of the trips (of the date), with the time in seconds and the order in the trip."""
        if "stop_times" not in self._cache:
            trip_ids = set(self.trips["trip_id"])
            logger.info("  Reading stop_times ...")
            st = self.table("stop_times", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
                            chunk_filter = lambda chunk: chunk[chunk["trip_id"].isin(trip_ids)])
            st["stop_sequence"] = st["stop_sequence"].astype(int)
            st = st.sort_values(["trip_id", "stop_sequence"]).reset_index(drop = True)
            st["departure_time"] = st["departure_time"].fillna(st["arrival_time"])
            st["arrival_time"] = st["arrival_time"].fillna(st["departure_time"])
            st["dep_s"] = time_to_seconds(st["departure_time"])
            st["arr_s"] = time_to_seconds(st["arrival_time"])
            logger.info(f"  {len(st)} stop times")
            self._cache["stop_times"] = st
        return self._cache["stop_times"]


def read_perimeter(path):
    """The perimeter as a GeoJSON-like dict (simplified, WGS84) and the polygon, or (None, None)."""
    if not path:
        return None, None
    shape = gpd.read_file(path)
    shape = shape.to_crs(2056) if shape.crs is not None else shape.set_crs(4326).to_crs(2056)
    union = shape.union_all().simplify(20)
    union_wgs = gpd.GeoSeries([union], crs = 2056).to_crs(4326).iloc[0]
    return json.loads(gpd.GeoSeries([union_wgs]).to_json()), union_wgs


def stops_inside(stops, polygon):
    if polygon is None:
        return set(stops["stop_id"])
    points = gpd.GeoSeries(gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs = 4326)
    return set(stops["stop_id"][points.within(polygon).values])


def write_page(output_path, kind, title, data):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok = True)
    html = visualize_pages.page(kind, title, data)
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(html)
    logger.info(f"  -> {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def _direction_of(trips):
    if "direction_id" in trips.columns:
        direction = trips["direction_id"].fillna("").astype(str).str.replace(".0", "", regex = False)
    else:
        direction = pd.Series("", index = trips.index)
    # without a direction_id, the Swiss trip ids end with H or R
    fallback = trips["trip_id"].str[-1].str.upper().where(trips["trip_id"].str[-1].str.upper().isin(["H", "R"]), "0")
    return direction.where(direction != "", fallback)


def build_lines(feed, perimeter = None):
    logger.info("Lines map")
    routes, trips, stops, st = feed.routes, feed.trips, feed.stops, feed.stop_times
    perimeter_geojson, polygon = read_perimeter(perimeter)

    trips = trips.merge(routes[["route_id", "agency_name", "name", "mode", "source"]], on = "route_id")
    trips["line"] = trips["agency_name"] + "|" + trips["name"]
    trips["direction"] = _direction_of(trips)

    if polygon is not None:
        inside = stops_inside(stops, polygon)
        keep = set(st.loc[st["stop_id"].isin(inside), "trip_id"])
        trips = trips[trips["trip_id"].isin(keep)]
        logger.info(f"  {trips['line'].nunique()} lines with a stop in the perimeter")

    st = st[st["trip_id"].isin(set(trips["trip_id"]))]
    per_trip = st.groupby("trip_id").agg(n_stops = ("stop_id", "nunique"), first_dep = ("dep_s", "first"))
    trips = trips.join(per_trip, on = "trip_id").dropna(subset = ["n_stops"])

    # one representative trip per line and direction: the one with the most stops, then the most frequent start time
    representative = trips.sort_values(["n_stops", "first_dep"], ascending = [False, True]).drop_duplicates(["line", "direction"])
    stop_lists = st[st["trip_id"].isin(set(representative["trip_id"]))].groupby("trip_id")["stop_id"].apply(list).to_dict()

    shapes = None
    if "shape_id" in representative.columns and representative["shape_id"].notna().any():
        wanted = set(representative["shape_id"].dropna())
        table = feed.table("shapes", ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], optional = True,
                           chunk_filter = lambda chunk: chunk[chunk["shape_id"].isin(wanted)])
        if table is not None and len(table):
            table = table.astype({"shape_pt_lat": float, "shape_pt_lon": float, "shape_pt_sequence": int}).sort_values(["shape_id", "shape_pt_sequence"])
            shapes = {sid: [[round(a, 5), round(b, 5)] for a, b in zip(g["shape_pt_lat"], g["shape_pt_lon"])] for sid, g in table.groupby("shape_id")}

    hours = (trips["first_dep"] // 3600 % 24).astype(int)
    departures = pd.crosstab([trips["line"], trips["direction"]], hours)
    counts = trips.groupby(["line", "direction"]).size()

    stop_index, stop_rows = {}, []
    stop_info = stops.drop_duplicates("stop_id").set_index("stop_id")

    def index_of(stop_id):
        if stop_id not in stop_index:
            if stop_id not in stop_info.index:
                return None
            row = stop_info.loc[stop_id]
            stop_index[stop_id] = len(stop_rows)
            stop_rows.append([row["stop_name"], round(row["stop_lat"], 5), round(row["stop_lon"], 5)])
        return stop_index[stop_id]

    sources = sorted({s for s in routes["source"] if s}) or []
    agencies = sorted(set(trips["agency_name"]))
    agency_index = {a: k for k, a in enumerate(agencies)}
    source_index = {s: k for k, s in enumerate(sources)}

    lines = []
    for line, group in representative.groupby("line", sort = False):
        first = group.iloc[0]
        directions = []
        for row in group.itertuples():
            ids = [index_of(s) for s in stop_lists.get(row.trip_id, [])]
            ids = [i for i in ids if i is not None]
            if len(ids) < 2:
                continue
            hourly = departures.loc[(line, row.direction)] if (line, row.direction) in departures.index else None
            entry = {
                "l": f"{stop_rows[ids[0]][0]} → {stop_rows[ids[-1]][0]}",
                "t": int(counts.get((line, row.direction), 0)),
                "h": [int(hourly.get(h, 0)) if hourly is not None else 0 for h in range(24)],
                "s": ids,
            }
            if shapes is not None and isinstance(row.shape_id, str) and row.shape_id in shapes:
                entry["p"] = shapes[row.shape_id]
            directions.append(entry)
        if directions:
            lines.append({"n": first["name"], "m": first["mode"], "a": agency_index[first["agency_name"]],
                          "s": source_index.get(first["source"], -1), "d": directions})

    logger.info(f"  {len(lines)} lines, {len(stop_rows)} stops")
    return {
        "date": feed.date or "", "lines": lines, "stops": stop_rows, "agencies": agencies, "sources": sources,
        "modes": MODES, "modeColors": MODE_COLORS, "sourceColors": SOURCE_COLORS, "perimeter": perimeter_geojson,
    }


# ---------------------------------------------------------------------------
# Stops and transfers
# ---------------------------------------------------------------------------

FAST_SPEED_M_S = 2.0       # a rule faster than this (distance / time) does not leave the time to walk
EXTRA_TIME_S   = 300       # a rule this much longer than walking at 1 m/s
RULE_ORIGINS   = {"feed": 0, "bridge": 1, "manual": 2, "nearby": 3}
NEARBY_M       = 300       # stops of different modes this close are expected to have a rule when their station has rules to that mode
REMOVED_DISTANCE_M, REMOVED_DURATION_S = 1500, 1200   # the thresholds of filter_long_transfers


def build_transfers(feed, perimeter = None):
    logger.info("Stops and transfers map")
    stops, routes, st, trips = feed.stops, feed.routes, feed.stop_times, feed.trips
    perimeter_geojson, polygon = read_perimeter(perimeter)

    rules_path = os.path.join(feed.diagnostics_path, "transfers.csv") if feed.diagnostics_path else None
    if rules_path and os.path.exists(rules_path):
        transfers = pd.read_csv(rules_path, dtype = {"from_stop_id": str, "to_stop_id": str})
        transfers["kept"] = transfers["kept"].astype(str).str.lower() == "true"
        has_rules = True
    else:
        transfers = feed.table("transfers", ["from_stop_id", "to_stop_id", "min_transfer_time"], optional = True)
        transfers = pd.DataFrame(columns = ["from_stop_id", "to_stop_id", "min_transfer_time"]) if transfers is None else transfers.copy()
        transfers["rule"], transfers["kept"] = "feed", True
        has_rules = False
        logger.info("  no diagnostics of data.gtfs.cleaned (re-run it): the origin of the rules and the removed ones are unknown")

    transfers["min_transfer_time"] = pd.to_numeric(transfers["min_transfer_time"], errors = "coerce").fillna(0)
    position = stops.drop_duplicates("stop_id").set_index("stop_id")[["stop_lat", "stop_lon"]]
    t = transfers.join(position, on = "from_stop_id").join(position, on = "to_stop_id", rsuffix = "_to").dropna(subset = ["stop_lat", "stop_lat_to"])
    t["dist"] = haversine_m(t["stop_lat"], t["stop_lon"], t["stop_lat_to"], t["stop_lon_to"])

    # modes serving each stop (on the date)
    stop_modes = (st[["trip_id", "stop_id"]].drop_duplicates()
                  .merge(trips[["trip_id", "route_id"]], on = "trip_id").merge(routes[["route_id", "mode"]], on = "route_id")
                  .groupby("stop_id")["mode"].agg(lambda m: sorted(set(m))))

    served = stops[stops["stop_id"].isin(stop_modes.index)].drop_duplicates("stop_id").reset_index(drop = True)
    if polygon is not None:
        served = served[served["stop_id"].isin(stops_inside(served, polygon))].reset_index(drop = True)
        t = t[t["from_stop_id"].isin(set(served["stop_id"])) | t["to_stop_id"].isin(set(served["stop_id"]))]

    # the stops shown: the served ones and the ones of the rules
    shown = pd.concat([served, stops[stops["stop_id"].isin(set(t["from_stop_id"]) | set(t["to_stop_id"]))]]).drop_duplicates("stop_id").reset_index(drop = True)
    index = {s: k for k, s in enumerate(shown["stop_id"])}
    stop_sources = feed.sources("stops")
    sources = sorted({s for s in stop_sources.values() if s})
    source_index = {s: k for k, s in enumerate(sources)}
    mode_bits = {m: 1 << k for k, m in enumerate(MODES)}

    stop_rows = [
        [row.stop_name, round(row.stop_lat, 5), round(row.stop_lon, 5), row.stop_id, source_index.get(stop_sources.get(row.stop_id, ""), -1),
         sum(mode_bits[m] for m in stop_modes.get(row.stop_id, []))]
        for row in shown.itertuples()
    ]

    # rules: [from, to, time, origin (0 source feed, 1 bridge, 2 manual, 3 nearby: added by gtfs.add_nearby_transfers),
    # status (0 kept, 1 removed for the distance, 2 for the time), distance]
    t = t[t["from_stop_id"] != t["to_stop_id"]]
    status = np.where(t["kept"], 0, np.where(t["dist"] > REMOVED_DISTANCE_M, 1, 2))
    rules = [[index[a], index[b], int(time), RULE_ORIGINS.get(rule, 0), int(s), int(round(d))]
             for a, b, time, rule, s, d in zip(t["from_stop_id"], t["to_stop_id"], t["min_transfer_time"], t["rule"], status, t["dist"])]

    # stops of different modes close to each other without a rule between them, when one of them has rules to other
    # stops of the mode of the second: the station has rules to some of its stops of that mode but not to this one, which
    # makes it faster to change there (e.g. in the 2024 feed, the rail platforms of Lausanne had 300 s to the metro at
    # Lausanne, gare but nothing to the metro at Lausanne, Grancy)
    pairs = set(zip(t["from_stop_id"], t["to_stop_id"])) | set(zip(t["to_stop_id"], t["from_stop_id"]))
    modes_with_rules = defaultdict(set)   # stop -> the modes of the stops it has rules to
    for a, b in pairs:
        modes_with_rules[a].update(stop_modes.get(b, []))
    lat0 = np.radians(served["stop_lat"].mean()) if len(served) else 0
    xy = np.column_stack([served["stop_lon"] * 111_320 * np.cos(lat0), served["stop_lat"] * 111_320])
    missing = []
    if len(served) > 1:
        ids = served["stop_id"].tolist()
        for i, j in cKDTree(xy).query_pairs(NEARBY_M):
            a, b = ids[i], ids[j]
            modes_a, modes_b = set(stop_modes.get(a, [])), set(stop_modes.get(b, []))
            if modes_a == modes_b or (a, b) in pairs:
                continue
            for x, y, modes_y in ((a, b, modes_b), (b, a, modes_a)):
                if modes_with_rules[x] & (modes_y - set(stop_modes.get(x, []))):
                    missing.append([index[x], index[y], int(round(float(np.hypot(*(xy[i] - xy[j])))))])
                    break

    logger.info(f"  {len(stop_rows)} stops, {len(rules)} rules between different stops, {len(missing)} nearby stops without a rule while their station has rules to that mode")
    return {
        "stops": stop_rows, "rules": rules, "missing": missing, "sources": sources, "sourceColors": SOURCE_COLORS,
        "modes": MODES, "hasRules": has_rules, "perimeter": perimeter_geojson,
        "thresholds": {"fast": FAST_SPEED_M_S, "extra": EXTRA_TIME_S, "nearby": NEARBY_M,
                       "removedDistance": REMOVED_DISTANCE_M, "removedDuration": REMOVED_DURATION_S},
    }


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def build_schedules(feed, perimeter = None):
    logger.info("Schedules map")
    stops, routes, trips, st = feed.stops, feed.routes, feed.trips, feed.stop_times
    perimeter_geojson, polygon = read_perimeter(perimeter)
    inside = stops_inside(stops, polygon)

    stop_info = stops.drop_duplicates("stop_id").set_index("stop_id")
    parent = stop_info["parent_station"] if "parent_station" in stop_info.columns else pd.Series(dtype = str)

    def station_of(stop_id):
        p = parent.get(stop_id)
        return p if isinstance(p, str) and p in stop_info.index else stop_id

    # one marker per station: the platforms of a station, and the stations of different feeds with the same code
    group_of = {}
    for stop_id in inside:
        station = station_of(stop_id)
        group_of[stop_id] = extract_station_code(station) or station

    # the trips that stop in the perimeter
    trips = trips.merge(routes[["route_id", "name", "mode"]], on = "route_id")
    trips = trips[trips["trip_id"].isin(set(st.loc[st["stop_id"].isin(inside), "trip_id"]))]
    st = st[st["trip_id"].isin(set(trips["trip_id"]))]
    first = st.groupby("trip_id").first()
    last = st.groupby("trip_id").last()

    names, name_index = [], {}

    def name_id(name):
        if name not in name_index:
            name_index[name] = len(names)
            names.append(name)
        return name_index[name]

    stop_name = stop_info["stop_name"]
    trip_index = {trip_id: k for k, trip_id in enumerate(first.index)}
    trip_rows = [
        [name_id(stop_name.get(a, "?")), name_id(stop_name.get(b, "?")), int(t0 // 60), int(t1 // 60)]
        for a, b, t0, t1 in zip(first["stop_id"], last["stop_id"], first["dep_s"], last["arr_s"])
    ]

    local = st[st["stop_id"].isin(inside)].copy()
    local["group"] = local["stop_id"].map(group_of)
    local["is_last"] = local["stop_sequence"] == local["trip_id"].map(last["stop_sequence"])
    local = local.merge(trips[["trip_id", "name", "mode"]], on = "trip_id")
    local["destination"] = local["trip_id"].map(last["stop_id"]).map(stop_name)
    local["origin"] = local["trip_id"].map(first["stop_id"]).map(stop_name)

    # the route drawn for a line and destination: the stops of its longest trip
    lengths = st.groupby("trip_id").size()
    path_stops, path_rows, paths = {}, [], {}

    def stop_id_of(stop_id):
        if stop_id not in path_stops:
            path_stops[stop_id] = len(path_rows)
            path_rows.append([round(stop_info.loc[stop_id, "stop_lat"], 5), round(stop_info.loc[stop_id, "stop_lon"], 5)])
        return path_stops[stop_id]

    # the longest trip of each line, origin and destination: its stops are the route drawn
    local["key"] = local["name"] + "|" + local["origin"].fillna("?") + "|" + local["destination"].fillna("?")
    longest = (local[["key", "trip_id"]].drop_duplicates().assign(length = lambda df: df["trip_id"].map(lengths))
               .sort_values("length", ascending = False).drop_duplicates("key").set_index("key")["trip_id"])
    stop_lists = st[st["trip_id"].isin(set(longest))].groupby("trip_id")["stop_id"].apply(list).to_dict()
    for key, trip_id in longest.items():
        paths[key] = [stop_id_of(s) for s in stop_lists.get(trip_id, []) if s in stop_info.index]

    groups = []
    for group, rows in local.groupby("group"):
        members = [s for s in rows["stop_id"].unique() if s in stop_info.index]
        stations = {station_of(s) for s in members}
        lat = float(np.mean([stop_info.loc[s, "stop_lat"] for s in stations if s in stop_info.index] or [stop_info.loc[members[0], "stop_lat"]]))
        lon = float(np.mean([stop_info.loc[s, "stop_lon"] for s in stations if s in stop_info.index] or [stop_info.loc[members[0], "stop_lon"]]))
        label = Counter(stop_name.get(s) for s in stations if s in stop_info.index).most_common(1)
        label = label[0][0] if label else stop_name.get(members[0], group)

        services = []
        for (line, destination), g in rows.groupby(["name", "destination"]):
            terminal = bool(g["is_last"].mean() >= 0.5)
            times = (g["arr_s"] if terminal else g["dep_s"]) // 60
            origin = Counter(g["origin"]).most_common(1)[0][0]
            key = Counter(g["key"]).most_common(1)[0][0]
            departures = sorted(zip(times.astype(int), g["trip_id"].map(trip_index)))
            services.append([name_id(line), name_id(origin if terminal else destination), int(terminal), MODES.index(g["mode"].iloc[0]),
                             key, [[int(m), int(k)] for m, k in departures]])
        groups.append([label, round(lat, 5), round(lon, 5), services])

    logger.info(f"  {len(groups)} stops, {len(trip_rows)} trips")
    return {
        "date": feed.date or "", "groups": groups, "names": names, "trips": trip_rows, "paths": paths, "points": path_rows,
        "modes": MODES, "modeColors": MODE_COLORS, "perimeter": perimeter_geojson,
    }


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

MAPS = ("lines", "stops_transfers", "schedules")


def configure(context):
    context.stage("data.gtfs.cleaned")

    context.config("data_path")

    # The maps to draw: any of lines, stops_transfers, schedules
    context.config("gtfs_visualize_maps", default = ["lines", "stops_transfers"])

    # Date (YYYYMMDD) the maps are restricted to; empty: all the services of the feed
    context.config("gtfs_visualize_date", default = "")

    # Optional area (shapefile or gpkg, absolute or relative to data_path) the maps of lines and transfers are restricted
    # to, and the one of the schedules map (a row for every departure: restrict it to an area and a date).
    context.config("gtfs_visualize_perimeter", default = "")
    context.config("gtfs_visualize_schedules_perimeter", default = "")


def _perimeter(data_path, value):
    if not value:
        return ""
    return value if os.path.isabs(value) else os.path.join(data_path, value)


def execute(context):
    cleaned   = context.path("data.gtfs.cleaned")
    data_path = context.config("data_path")
    date      = str(context.config("gtfs_visualize_date") or "")
    maps      = context.config("gtfs_visualize_maps")
    if isinstance(maps, str):
        maps = [m.strip() for m in maps.split(",") if m.strip()]

    unknown = set(maps) - set(MAPS)
    if unknown:
        raise RuntimeError(f"Unknown maps {sorted(unknown)}, choose among {list(MAPS)}")

    feed = Feed("%s/output" % cleaned, date or None, diagnostics_path = os.path.join(cleaned, "diagnostics"))
    perimeter = _perimeter(data_path, context.config("gtfs_visualize_perimeter"))
    output = {}

    if "lines" in maps:
        output["lines"] = "%s/gtfs_lines.html" % context.path()
        write_page(output["lines"], "lines", "GTFS lines", build_lines(feed, perimeter))

    if "stops_transfers" in maps:
        output["stops_transfers"] = "%s/stops_transfers_map.html" % context.path()
        write_page(output["stops_transfers"], "transfers", "GTFS transfers", build_transfers(feed, perimeter))

    if "schedules" in maps:
        output["schedules"] = "%s/gtfs_schedules.html" % context.path()
        schedules_perimeter = _perimeter(data_path, context.config("gtfs_visualize_schedules_perimeter")) or perimeter
        write_page(output["schedules"], "schedules", "GTFS schedules", build_schedules(feed, schedules_perimeter))

    return output
