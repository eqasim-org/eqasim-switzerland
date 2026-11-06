"""Reading a GTFS timetable and turning it into the arrays the router works on.

A GTFS feed is a set of tables (stops.txt, trips.txt, stop_times.txt, calendar.txt...). From them, this builds a Network:

- the stops that are served, with their names, positions and change times;
- the connections: every vehicle going from one stop to the next one of its trip, sorted by departure time (what the
  Connection Scan Algorithm goes through);
- the stops of each trip in order, with their times;
- the walks between stops close to each other.

The whole feed is read once, with the days each trip runs (the calendar of its service), and kept in the cache: for
Switzerland, about 30 s and 1.7 GB the first time. The network of one date (Network.on) then only keeps the connections
of the trips running that day, which takes a fraction of a second. The stops, trips and walks stay those of the whole
feed, so their numbers (indices) are the same on every date.

Many arrays are stored "flat": all the stops of all the trips in one array, with t_start telling where each trip begins
(the stops of trip t are t_stop[t_start[t]:t_start[t + 1]]). The walks from each stop are stored the same way.
"""
import dataclasses
import hashlib
import logging
import os
import pickle
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger("synpp")

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
VERSION = 2   # change it when the network built changes, so that the old cache files are not used


def mode_of(route_type):
    """The mode (rail, bus, tram...) of a GTFS route_type, of the basic types (0 to 12) or of the extended ones (100 to
    1499)."""
    t = int(route_type)
    basic = {0: "tram", 1: "metro", 2: "rail", 3: "bus", 4: "boat", 5: "cable car", 6: "cable car", 7: "funicular", 11: "bus", 12: "rail"}
    if t in basic:
        return basic[t]
    for low, high, mode in ((100, 199, "rail"), (200, 299, "bus"), (400, 499, "metro"), (700, 899, "bus"), (900, 999, "tram"),
                            (1000, 1099, "boat"), (1300, 1399, "cable car"), (1400, 1499, "funicular")):
        if low <= t <= high:
            return mode
    return "other"


def to_seconds(values):
    """GTFS times ("HH:MM:SS", the hours can go past 24 for the trips after midnight) as seconds after midnight."""
    if len(values) == 0:
        return np.zeros(0, np.int32)
    parts = values.astype(str).str.split(":", expand = True)
    return (parts[0].astype(int) * 3600 + parts[1].astype(int) * 60 + parts[2].astype(int)).to_numpy(np.int32)


def station_code(stop_id):
    """The station code of a stop id, the same for all the platforms of a station: 8501120:0:1 -> 8501120,
    Parent8501120 -> 8501120."""
    return str(stop_id).replace("Parent", "").split(":")[0]


def day_of(date):
    """A date as YYYYMMDD, given as YYYYMMDD, YYYY-MM-DD, a Timestamp..."""
    return pd.Timestamp(str(date)).strftime("%Y%m%d")


class _Tables:
    """The .txt tables of a feed, in a zip file or a folder, read as text (no conversion of ids or times). The tables can
    also be in a folder of the zip or of the folder (output/stops.txt..., as data.gtfs.cleaned of eqasim writes them)."""

    def __init__(self, path):
        self.path = path
        self.prefix = ""   # the folder of the tables in the zip (or the folder), "" at its root
        if os.path.isdir(path):
            names = [os.path.relpath(os.path.join(d, f), path).replace(os.sep, "/") for d, _, files in os.walk(path) for f in files]
        else:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        stops = sorted((n for n in names if n == "stops.txt" or n.endswith("/stops.txt")), key = lambda n: n.count("/"))
        if not stops:
            raise ValueError(f"No stops.txt in the GTFS feed {path}")
        self.prefix = stops[0][:-len("stops.txt")]

    def _name(self, name):
        """The path of a table (name without .txt) in the zip or the folder."""
        return f"{self.prefix}{name}.txt"

    def has(self, name):
        """Whether the feed has the table (name without .txt)."""
        if os.path.isdir(self.path):
            return os.path.exists(os.path.join(self.path, self._name(name)))
        with zipfile.ZipFile(self.path) as z:
            return self._name(name) in z.namelist()

    def read(self, name, usecols = None, chunk_filter = None):
        """A table as a DataFrame of strings. usecols: the columns wanted (the others are not read); chunk_filter: a
        function applied to each piece of 2 million rows as it is read, to keep a large table (stop_times) small."""
        def read(handle):
            reader = pd.read_csv(handle, dtype = str, usecols = (lambda c: c in usecols) if usecols else None,
                                 keep_default_na = False, na_values = [""], chunksize = 2_000_000)
            return pd.concat([chunk_filter(c) if chunk_filter else c for c in reader], ignore_index = True)

        if os.path.isdir(self.path):
            return read(os.path.join(self.path, self._name(name)))
        with zipfile.ZipFile(self.path) as z, z.open(self._name(name)) as f:
            return read(f)


def service_calendar(tables):
    """The days each service runs, from calendar.txt (weekdays between two dates) and calendar_dates.txt (days added or
    removed). Returns (service ids, the first day as YYYYMMDD, a table of booleans: one row per service, one column per
    day from the first one)."""
    cal = tables.read("calendar") if tables.has("calendar") else None
    cd = tables.read("calendar_dates", ["service_id", "date", "exception_type"]) if tables.has("calendar_dates") else None
    ids, bounds = [], []
    if cal is not None:
        ids.append(cal["service_id"])
        bounds += [pd.to_datetime(cal["start_date"], format = "%Y%m%d").min(), pd.to_datetime(cal["end_date"], format = "%Y%m%d").max()]
    if cd is not None:
        cd_day = pd.to_datetime(cd["date"], format = "%Y%m%d")
        ids.append(cd["service_id"])
        bounds += [cd_day.min(), cd_day.max()]
    if not ids:
        raise ValueError(f"The GTFS feed {tables.path} has neither calendar.txt nor calendar_dates.txt")
    services = pd.Index(pd.unique(pd.concat(ids, ignore_index = True)))
    first, last = min(bounds), max(bounds)
    days = pd.date_range(first, last)
    runs = np.zeros((len(services), len(days)), bool)
    if cal is not None:   # the weekdays of each service, between its start and end dates
        start = (pd.to_datetime(cal["start_date"], format = "%Y%m%d") - first).dt.days.to_numpy()
        end = (pd.to_datetime(cal["end_date"], format = "%Y%m%d") - first).dt.days.to_numpy()
        week = cal[WEEKDAYS].astype(int).to_numpy().astype(bool)
        d = np.arange(len(days))
        np.logical_or.at(runs, services.get_indexer(cal["service_id"]),
                         week[:, days.weekday] & (d >= start[:, None]) & (d <= end[:, None]))
    if cd is not None:    # the exceptions: 1 the service runs that day, 2 it does not
        s, d, kind = services.get_indexer(cd["service_id"]), (cd_day - first).dt.days.to_numpy(), cd["exception_type"].astype(int).to_numpy()
        runs[s[kind == 1], d[kind == 1]] = True
        runs[s[kind == 2], d[kind == 2]] = False
    return services, first.strftime("%Y%m%d"), runs


@dataclass
class Network:
    """The timetable as arrays, for the whole feed or for one date (see the top of this file).

    Stops, trips and routes are referred to by their position (index) in these arrays; times are seconds after the
    midnight of the date, and can go past 24:00."""
    date: Optional[str]         # YYYYMMDD for the network of a date (Network.on), None for the whole feed
    # the stops served by the feed
    stop_ids: np.ndarray
    stop_names: np.ndarray
    stop_lat: np.ndarray
    stop_lon: np.ndarray
    change_s: np.ndarray        # the time needed to change vehicles at the stop (transfers.txt, else a default)
    # the connections, sorted by departure time: a vehicle of trip c_trip leaving c_dep_stop at c_dep and arriving at
    # the next stop of its trip, c_arr_stop, at c_arr
    c_dep_stop: np.ndarray
    c_arr_stop: np.ndarray
    c_dep: np.ndarray
    c_arr: np.ndarray
    c_trip: np.ndarray
    c_pos: np.ndarray           # the position of the departure stop in its trip (0: its first stop)
    # the trips: their stops in order (flat, see t_start), their ids and routes
    t_start: np.ndarray         # the stops of trip t are at t_start[t] to t_start[t + 1] in t_stop, t_arr, t_dep
    t_stop: np.ndarray
    t_arr: np.ndarray
    t_dep: np.ndarray
    trip_ids: np.ndarray
    trip_route: np.ndarray      # the row of the trip's route in routes
    routes: pd.DataFrame        # route_id, name (the line, e.g. IC5 or 31), mode, agency
    # the walks between stops (flat): from stop s to f_to[f_start[s]:f_start[s + 1]], in f_dur seconds
    f_start: np.ndarray
    f_to: np.ndarray
    f_dur: np.ndarray
    # the days the trips run (for the whole feed only)
    first_day: Optional[str] = None          # YYYYMMDD, the day of the first column of service_days
    trip_service: Optional[np.ndarray] = None   # the row of service_days of each trip
    service_days: Optional[np.ndarray] = None   # booleans, one row per service, one column per day: whether it runs

    @property
    def dates(self):
        """(first, last): the first and last dates (YYYYMMDD) the network covers."""
        if self.service_days is None:
            return self.date, self.date
        last = pd.Timestamp(self.first_day) + pd.Timedelta(days = self.service_days.shape[1] - 1)
        return self.first_day, last.strftime("%Y%m%d")

    def on(self, date):
        """The network of a date: only the connections of the trips running that day, with the stops, trips and walks of
        the whole feed. A trip of the day before that runs after midnight (times after 24:00) is not included."""
        date = day_of(date)
        if self.service_days is None:
            if date != self.date:
                raise ValueError(f"The network is the one of {self.date}, not {date}")
            return self
        first, last = self.dates
        day = (pd.Timestamp(date) - pd.Timestamp(first)).days
        if not first <= date <= last:
            raise ValueError(f"{date} is outside the feed ({first} to {last})")
        keep = self.service_days[:, day][self.trip_service][self.c_trip]
        return dataclasses.replace(self, date = date, c_dep_stop = self.c_dep_stop[keep], c_arr_stop = self.c_arr_stop[keep],
                                   c_dep = self.c_dep[keep], c_arr = self.c_arr[keep], c_trip = self.c_trip[keep], c_pos = self.c_pos[keep],
                                   first_day = None, trip_service = None, service_days = None)

    def departures(self):
        """The number of departures from each stop in a day (for the whole feed: on average over its dates), to rank
        the stations by how busy they are."""
        if self.service_days is None:
            return np.bincount(self.c_dep_stop, minlength = len(self.stop_ids))
        per_day = self.service_days.mean(axis = 1)[self.trip_service][self.c_trip]
        return np.bincount(self.c_dep_stop, weights = per_day, minlength = len(self.stop_ids))

    def stops_near(self, lat, lon, radius_m):
        """The stops within radius_m of a point: (their indices, their straight-line distances in metres)."""
        k = 111_320
        c = np.cos(np.radians(lat))
        d = np.hypot((self.stop_lat - lat) * k, (self.stop_lon - lon) * k * c)
        near = np.flatnonzero(d <= radius_m)
        return near, d[near]

    def stop_index(self, stop_id):
        """The indices of a stop id, or, for a station code (8507000), of all the platforms of the station, plus the stop
        with the code itself (some feeds serve both 8507000 and 8507000:0:6)."""
        code = station_code(stop_id)
        exact = np.flatnonzero(self.stop_ids == stop_id)
        if len(exact) and code != str(stop_id):
            return exact
        return np.flatnonzero(np.array([station_code(s) == code for s in self.stop_ids]))


def _inside(area, lon, lat, margin_m):
    """Whether each point is within margin_m of the polygons of the file area."""
    import geopandas as gpd
    import shapely
    polygons = gpd.read_file(area).to_crs(2056).union_all().buffer(margin_m)
    points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs = 4326).to_crs(2056)
    shapely.prepare(polygons)
    return shapely.contains_xy(polygons, points.x.to_numpy(), points.y.to_numpy())


def build_network(gtfs_path, date = None, walk_radius_m = 400, walk_speed_m_s = 1.2, detour = 1.3, default_change_s = 60, cache_dir = None,
                  area = None, area_margin_m = 100):
    """The Network of a GTFS feed (a zip file or a folder): of the whole feed, or of one date (YYYYMMDD) when date is
    given (as Network.on).

    walk_radius_m: stops closer than this are linked by a walk, at walk_speed_m_s along the straight line × detour,
    unless transfers.txt gives the time. default_change_s: the time to change vehicles at a stop without a transfers.txt
    rule. cache_dir: where the whole feed is kept once read. area: a file of polygons (e.g.
    data/network/tarifverbundkarte.geojson, the Swiss municipalities); the stops further than area_margin_m from them are
    left out, and a trip leaving the area goes straight from its last stop inside to its next one."""
    network = _build(gtfs_path, walk_radius_m, walk_speed_m_s, detour, default_change_s, cache_dir, area, area_margin_m)
    return network.on(date) if date is not None else network


def _build(gtfs_path, walk_radius_m, walk_speed_m_s, detour, default_change_s, cache_dir, area, area_margin_m):
    """The Network of the whole feed, from the cache when it was built with the same file and settings."""
    if cache_dir:
        stamp = f"{VERSION}|{os.path.abspath(gtfs_path)}|{os.path.getmtime(gtfs_path)}|{walk_radius_m}|{walk_speed_m_s}|{detour}|{default_change_s}"
        if area:
            stamp += f"|{os.path.abspath(area)}|{os.path.getmtime(area)}|{area_margin_m}"
        name = os.path.splitext(os.path.basename(os.path.normpath(gtfs_path)))[0]
        cache = os.path.join(cache_dir, f"network_{name}_{hashlib.md5(stamp.encode()).hexdigest()[:10]}.pkl")
        if os.path.exists(cache):
            logger.info(f"Network from the cache {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)

    tables = _Tables(gtfs_path)
    logger.info(f"Reading {gtfs_path}")
    services, first_day, service_days = service_calendar(tables)
    logger.info(f"  {len(services)} services, {service_days.shape[1]} days from {first_day}")
    trips = tables.read("trips", ["route_id", "service_id", "trip_id"]).drop_duplicates("trip_id").reset_index(drop = True)
    all_stops = tables.read("stops", ["stop_id", "stop_name", "stop_lat", "stop_lon"]).drop_duplicates("stop_id").dropna(subset = ["stop_lat"])
    all_stops = all_stops.reset_index(drop = True)
    trip_codes, stop_codes = pd.Index(trips["trip_id"]), pd.Index(all_stops["stop_id"])
    logger.info(f"  {len(trips)} trips, {len(all_stops)} stops")

    def compact(c):
        """A piece of stop_times.txt as numbers: the trip and stop (their rows in trips and all_stops), the sequence and
        the times in seconds; the rows without a time or with an unknown trip or stop are dropped."""
        dep = c["departure_time"].fillna(c["arrival_time"])
        arr = c["arrival_time"].fillna(dep)
        trip, stop = trip_codes.get_indexer(c["trip_id"]), stop_codes.get_indexer(c["stop_id"])
        keep = dep.notna().to_numpy() & (trip >= 0) & (stop >= 0)
        return pd.DataFrame({"trip": trip[keep].astype(np.int32), "stop": stop[keep].astype(np.int32),
                             "seq": c["stop_sequence"][keep].astype(np.int32).to_numpy(), "arr": to_seconds(arr[keep]), "dep": to_seconds(dep[keep])})

    st = tables.read("stop_times", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], chunk_filter = compact)
    logger.info(f"  {len(st)} stop times")

    # the stops: only those served by a trip (and inside the area, if any), numbered again from 0
    served = np.unique(st["stop"].to_numpy())
    stops = all_stops.iloc[served]
    if area:
        inside = _inside(area, stops["stop_lon"].astype(float).to_numpy(), stops["stop_lat"].astype(float).to_numpy(), area_margin_m)
        logger.info(f"  {(~inside).sum()} of {len(stops)} stops outside {os.path.basename(area)}")
        served, stops = served[inside], stops[inside]
    stops = stops.set_index("stop_id")
    renumber = np.full(len(all_stops), -1, np.int32)
    renumber[served] = np.arange(len(served))
    st_stop = renumber[st["stop"].to_numpy()]
    keep = st_stop >= 0
    trip, seq = st["trip"].to_numpy()[keep], st["seq"].to_numpy()[keep]
    t_stop, t_arr, t_dep = st_stop[keep], st["arr"].to_numpy()[keep], st["dep"].to_numpy()[keep]
    del st
    order = np.lexsort((seq, trip))   # the stop times by trip, then in the order of the trip
    trip, t_stop, t_arr, t_dep = trip[order], t_stop[order], t_arr[order], t_dep[order]
    stop_index = pd.Series(np.arange(len(stops)), index = stops.index)
    lat, lon = stops["stop_lat"].astype(float).to_numpy(), stops["stop_lon"].astype(float).to_numpy()

    # the trips: those with stops left, numbered again from 0, with where their stops begin (t_start) and their service
    trip_order, t = np.unique(trip, return_inverse = True)
    t = t.astype(np.int32)
    t_start = np.concatenate([[0], np.cumsum(np.bincount(t))]).astype(np.int64)
    trips = trips.iloc[trip_order]
    trip_service = services.get_indexer(trips["service_id"])
    trip_service[trip_service < 0] = len(services)   # a service missing from the calendar never runs: an extra row of False
    service_days = np.vstack([service_days, np.zeros((1, service_days.shape[1]), bool)])

    # the routes: the name shown for the line (short name, else long name, else id), the mode and the agency
    routes = tables.read("routes", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
    agency = tables.read("agency", ["agency_id", "agency_name"]) if tables.has("agency") else None
    if agency is not None and "agency_id" in routes.columns:
        routes = routes.merge(agency.drop_duplicates("agency_id"), on = "agency_id", how = "left")
    routes["name"] = routes["route_short_name"].fillna(routes.get("route_long_name")).fillna(routes["route_id"])
    routes["mode"] = routes["route_type"].fillna("3").map(mode_of)
    routes["agency"] = routes.get("agency_name", pd.Series("", index = routes.index)).fillna("")
    routes = routes[["route_id", "name", "mode", "agency"]].drop_duplicates("route_id").reset_index(drop = True)
    trip_route = pd.Index(routes["route_id"]).get_indexer(trips["route_id"]).astype(np.int32)

    # the connections: from each stop of a trip to its next one (not from the last stop), sorted by departure time
    pos = np.arange(len(t)) - t_start[t]
    nxt = np.arange(len(t)) + 1
    valid = np.ones(len(t), bool)
    valid[t_start[1:] - 1] = False  # the last stop of each trip
    i = np.flatnonzero(valid)
    c_dep, c_arr = t_dep[i], t_arr[nxt[i]]
    order = np.lexsort((c_arr, c_dep))
    i = i[order]

    # the walks: between the stops within walk_radius_m (found with a k-d tree on flat coordinates in metres), the time
    # of transfers.txt when there is a rule, else the straight-line distance × detour at walking speed. A transfers.txt
    # rule from a stop to itself is its change time; rules between stops further apart are added as walks too
    k = 111_320
    lat0 = np.radians(lat.mean())
    xy = np.column_stack([lon * k * np.cos(lat0), lat * k])
    pairs = np.array(sorted(cKDTree(xy).query_pairs(walk_radius_m)), dtype = np.int64).reshape(-1, 2)
    pairs = np.vstack([pairs, pairs[:, ::-1]])   # both ways
    dur = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) * detour / walk_speed_m_s
    change = np.full(len(stops), default_change_s, np.int32)
    if tables.has("transfers"):
        tr = tables.read("transfers", ["from_stop_id", "to_stop_id", "min_transfer_time"])
        tr = tr[tr["from_stop_id"].isin(stop_index.index) & tr["to_stop_id"].isin(stop_index.index)]
        tr["min_transfer_time"] = pd.to_numeric(tr["min_transfer_time"], errors = "coerce").fillna(0)
        a, b = stop_index.loc[tr["from_stop_id"]].to_numpy(), stop_index.loc[tr["to_stop_id"]].to_numpy()
        same = a == b
        change[a[same]] = tr["min_transfer_time"].to_numpy()[same]
        rules = pd.Series(tr["min_transfer_time"].to_numpy()[~same], index = pd.MultiIndex.from_arrays([a[~same], b[~same]])).groupby(level = [0, 1]).min()
        known = pd.MultiIndex.from_arrays([pairs[:, 0], pairs[:, 1]])
        hit = known.isin(rules.index)
        dur[hit] = rules.reindex(known[hit]).to_numpy()
        extra = rules.index.difference(known)   # rules between stops further apart than the radius
        if len(extra):
            pairs = np.vstack([pairs, np.array(list(extra))])
            dur = np.concatenate([dur, rules.reindex(extra).to_numpy()])
        logger.info(f"  {len(tr)} transfers.txt rules ({same.sum()} at the same stop)")
    o = np.argsort(pairs[:, 0], kind = "stable")   # the walks grouped by the stop they leave (f_start)
    pairs, dur = pairs[o], dur[o]
    f_start = np.searchsorted(pairs[:, 0], np.arange(len(stops) + 1)).astype(np.int64)

    network = Network(
        date = None, stop_ids = stops.index.to_numpy(str), stop_names = stops["stop_name"].fillna("").to_numpy(str),
        stop_lat = lat, stop_lon = lon, change_s = change,
        c_dep_stop = t_stop[i], c_arr_stop = t_stop[nxt[i]], c_dep = c_dep[order], c_arr = c_arr[order],
        c_trip = t[i], c_pos = pos[i].astype(np.int32),
        t_start = t_start, t_stop = t_stop, t_arr = t_arr, t_dep = t_dep, trip_ids = trips["trip_id"].to_numpy(str),
        trip_route = trip_route, routes = routes,
        f_start = f_start, f_to = pairs[:, 1].astype(np.int32), f_dur = np.ceil(dur).astype(np.int32),
        first_day = first_day, trip_service = trip_service.astype(np.int32), service_days = service_days,
    )
    first, last = network.dates
    logger.info(f"  {len(stops)} stops, {len(trips)} trips, {len(i)} connections ({first} to {last}), {len(pairs)} walking links")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok = True)
        with open(cache, "wb") as f:
            pickle.dump(network, f, protocol = pickle.HIGHEST_PROTOCOL)
    return network
