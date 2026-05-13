from pathlib import Path
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd


DIST_BINS          = [300, 500, 750, 1000]
RAIL_TYPES         = {100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117}
TRAM_TYPES         = {0, 900, 1, 400, 401, 402, 3, 700,702,704,705,710,712,715,716,717}
OTHER_TYPES        = {4, 5, 6, 1000, 1300, 1303, 1400, 1500}
NB_LINES_THRESHOLD = 7


def read_gtfs(gtfs_path):
    def read_files(open_fn, namelist_fn):
        stops      = pd.read_csv(open_fn("stops.txt"))
        routes     = pd.read_csv(open_fn("routes.txt"))
        trips      = pd.read_csv(open_fn("trips.txt"))
        stop_times = pd.read_csv(open_fn("stop_times.txt"))

        calendar       = pd.read_csv(open_fn("calendar.txt"))       if "calendar.txt"       in namelist_fn() else None
        calendar_dates = pd.read_csv(open_fn("calendar_dates.txt")) if "calendar_dates.txt" in namelist_fn() else None

        return stops, routes, trips, stop_times, calendar, calendar_dates

    gtfs_path = Path(gtfs_path)

    if gtfs_path.is_dir():
        def open_fn(filename):
            return open(gtfs_path / filename, "rb")
        def namelist_fn():
            return [f.name for f in gtfs_path.iterdir()]
        return read_files(open_fn, namelist_fn)

    elif zipfile.is_zipfile(gtfs_path):
        with zipfile.ZipFile(gtfs_path) as zf:
            def open_fn(filename):
                return zf.open(filename)
            def namelist_fn():
                return zf.namelist()
            return read_files(open_fn, namelist_fn)

    else:
        raise ValueError(f"gtfs_path must be a directory or a .zip file, got: {gtfs_path}")


def service_ids_for_date(calendar, calendar_dates, target_date = None):
    d = pd.to_datetime(target_date) if target_date else None

    if d is not None:
        weekday = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"][d.dayofweek]
    else:
        weekday = "wednesday"

    # Base set from calendar.txt
    if calendar is not None:
        active = set(calendar.loc[calendar[weekday] == 1, "service_id"].astype(str))
    else:
        active = set()

    # Apply exceptions from calendar_dates.txt
    if calendar_dates is not None and d is not None:
        ds = int(d.strftime("%Y%m%d"))
        day_exceptions = calendar_dates[calendar_dates["date"] == ds]

        # exception_type 1 = added, 2 = removed
        adds = day_exceptions.loc[day_exceptions["exception_type"] == 1, "service_id"].astype(str)
        rems = day_exceptions.loc[day_exceptions["exception_type"] == 2, "service_id"].astype(str)

        active.update(adds)
        active.difference_update(rems)

    return sorted(active)


def classify_routes(routes):
    r    = routes.copy()
    mode = r["route_type"].fillna(-1).astype(int)

    r["mode_group"] = "C"
    r.loc[mode.isin(RAIL_TYPES), "mode_group"] = "A2"
    r.loc[mode.isin(TRAM_TYPES), "mode_group"] = "B"

    return r[["route_id", "mode_group"]]


def parse_time_to_min(s):
    parts = s.str.split(":", expand=True).astype(int)
    return parts[0] * 60 + parts[1] + parts[2] / 60


def interval_and_mode_to_stopcat(stop_cat_mode, interval_min):
    if pd.isna(interval_min):
        return None
    
    THRESHOLDS = [(5, "I", "I", "II", "V"),
                  (10, "I", "II", "III", "V"),
                  (20, "II", "III", "IV", "V"),
                  (40, "III", "IV", "V", "V"),
                  (60, "IV", "V", "V", "V"),
                  (float("inf"), "X", "X", "X", "X")]
    
    for upper, rail_1, rail_2, b, c in THRESHOLDS:
        if interval_min < upper:
            if stop_cat_mode == "A1":
                return rail_1
            elif stop_cat_mode == "A2":
                return rail_2
            elif stop_cat_mode == "B":
                return b
            elif stop_cat_mode == "C":
                return c
            else:
                return c
            
    return "X"


def compute_stop_category(gtfs_path, date):
    stops, routes, trips, stop_times, calendar, calendar_dates = read_gtfs(gtfs_path)

    routes2 = classify_routes(routes) # Find route mode
    trips2  = trips.merge(routes2, on = "route_id", how = "left")

    if calendar is not None and "service_id" in trips2.columns:
        service_ids = service_ids_for_date(calendar, calendar_dates, date)
        if service_ids is not None:
            trips2 = trips2[trips2["service_id"].astype(str).isin(service_ids)]

    stop_times2    = stop_times.merge(trips2[["trip_id", "route_id", "mode_group", "direction_id"]], on="trip_id", how="inner")
    stop_to_parent = stops.set_index("stop_id").apply(lambda r: r["parent_station"] if pd.notna(r.get("parent_station")) and r.get("parent_station") != "" else r.name, axis=1)
    
    stop_times2["station_id"] = stop_times2["stop_id"].map(stop_to_parent).fillna(stop_times2["stop_id"])

    route_names = routes[["route_id", "route_short_name"]].copy()
    route_names["route_key"] = route_names["route_short_name"].fillna(route_names["route_id"]).astype(str)

    rail_st = stop_times2[stop_times2["mode_group"] == "A2"].merge(route_names[["route_id", "route_key"]], on="route_id", how="left")

    rail_lines_per_station = rail_st.groupby("station_id")["route_key"].nunique().reset_index().rename(columns={"route_key": "num_rail_lines"})

    st_timed = stop_times2.copy()
    st_timed["dep_min"] = parse_time_to_min(stop_times2["departure_time"].astype(str))
    st_timed = st_timed[(st_timed["dep_min"] >= 360) & (st_timed["dep_min"] < 1200)]  # 06:00–20:00

    # Highest-ranking mode per station
    mode_rank = {"A2": 0, "B": 1, "C": 2}
    station_mode = (
        st_timed.groupby("station_id")["mode_group"]
        .apply(lambda x: min(x.unique(), key=lambda m: mode_rank.get(m, 99)))
        .reset_index()
        .rename(columns={"mode_group": "best_mode"})
    )

    # Departures per station, divided by 2 to account for both directions
    dep_counts = (
        st_timed.groupby("station_id")
        .size()
        .reset_index(name="raw_departures")
    )
    dep_counts["departures"] = dep_counts["raw_departures"] / 2.0
    
    # Average interval in minutes over 14h window
    dep_counts["interval_min"] = (14 * 60) / dep_counts["departures"]

    # Assemble final stop category
    result = stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()

    # Attach parent station
    result["station_id"] = result["stop_id"].map(stop_to_parent).fillna(result["stop_id"])

    result = result.merge(station_mode,   on="station_id", how="left")
    result = result.merge(dep_counts[["station_id", "interval_min"]], on="station_id", how="left")
    result = result.merge(rail_lines_per_station, on="station_id", how="left")
    result["num_rail_lines"] = result["num_rail_lines"].fillna(0).astype(int)

    # Classify: A1 if rail + more than one distinct line, else A2, B, C
    def assign_category(row):
        if row["best_mode"] == "A2":
            return "A1" if row["num_rail_lines"] > NB_LINES_THRESHOLD else "A2"
        return row.get("best_mode", "C")

    result["stop_category"] = result.apply(assign_category, axis=1)
    result["stop_category"] = result["stop_category"].fillna("C")

    stops = result[["stop_id", "station_id", "stop_name", "stop_lat", "stop_lon",
                   "best_mode", "num_rail_lines", "interval_min", "stop_category"]].copy()
    
    stops = stops.rename(columns = {"stop_category": "stop_cat_mode"})

    stops["stop_cat"] = stops.apply(lambda r : interval_and_mode_to_stopcat(r["stop_cat_mode"], r["interval_min"]), axis = 1)
    

    return stops


def compute_ovgk_areas(stops):
    gdf_stops = gpd.GeoDataFrame(stops, geometry = gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs = "EPSG:4326")
    gdf_stops = gdf_stops.to_crs("EPSG:2056")

    rings_param = {
        "I":   [(0, 299, "A"), (300, 500, "A"), (501, 750, "B"), (751, 1000, "C")],
        "II":  [(0, 299, "A"), (300, 500, "B"), (501, 750, "C"), (751, 1000, "D")],
        "III": [(0, 299, "B"), (300, 500, "C"), (501, 750, "D")],
        "IV":  [(0, 299, "C"), (300, 500, "D")],
        "V":   [(0, 299, "D")]
    }

    class_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "Z": 4}

    rings = []

    for _, row in gdf_stops.iterrows():
        stop_cat = row["stop_cat"]
        if stop_cat not in rings_param:
            continue

        ptstop = row["geometry"]
        for dmin, dmax, theovgk in rings_param[stop_cat]:
            outer = ptstop.buffer(dmax)
            inner = ptstop.buffer(dmin) if dmin > 0 else None

            ring = outer
            if inner is None:
                ring = outer
            else:
                ring = outer.difference(inner)

            rings.append({"stop_id": row["stop_id"],
                          "stop_cat": stop_cat,
                          "ovgk_class": theovgk,
                          "class_rank": class_rank[theovgk],
                          "geometry": ring})
            
    gdf_rings = gpd.GeoDataFrame(rings, crs = gdf_stops.crs) 

    gdf_a = gdf_rings[gdf_rings["ovgk_class"] == "A"].copy()
    gdf_b = gdf_rings[gdf_rings["ovgk_class"] == "B"].copy()
    gdf_c = gdf_rings[gdf_rings["ovgk_class"] == "C"].copy()
    gdf_d = gdf_rings[gdf_rings["ovgk_class"] == "D"].copy()

    a_merged = gdf_a.dissolve(by = None).geometry.iloc[0]  
    b_merged = gdf_b.dissolve(by = None).geometry.iloc[0]  
    c_merged = gdf_c.dissolve(by = None).geometry.iloc[0] 
    d_merged = gdf_d.dissolve(by = None).geometry.iloc[0] 

    b_final = b_merged.difference(a_merged)

    a_union_b = a_merged.union(b_final)
    c_final   = c_merged.difference(a_union_b)

    a_union_b_union_c = a_union_b.union(c_final)
    d_final           = d_merged.difference(a_union_b_union_c)

    gdf_output = gpd.GeoDataFrame({"ovgk_class": [], "geometry": []}, crs = gdf_rings.crs)

    dic_gdf = {"A": a_merged, "B": b_final, "C": c_final, "D": d_final}

    for ovgk, shape in dic_gdf.items():
        thegdf     = gpd.GeoDataFrame({"ovgk_class": [ovgk], "geometry": shape}, crs = gdf_rings.crs)
        gdf_output = pd.concat([gdf_output, thegdf], ignore_index = True) 

    return gdf_output


def configure(context):
    context.config("data_path")
    context.config("threads")

    context.config("compute_ovgk_from_gtfs", default = False)
    if context.config("compute_ovgk_from_gtfs"):
        context.stage("data.gtfs.cleaned")


def execute(context):

    if not context.config("compute_ovgk_from_gtfs"):

        input_path = "%s/spatial/ov_guteklasse/OeV_Gueteklassen_ARE.gpkg" % context.config("data_path")
        df = gpd.read_file(input_path)
        df.crs = "epsg:2056"
        df = df[["KLASSE", "geometry"]].rename({"KLASSE": "ovgk"}, axis=1)
        return df
    
    else:
        gtfs_path = "%s/output" % context.path("data.gtfs.cleaned")
        date      = "2024/03/20"

        stops = compute_stop_category(gtfs_path, date)
        ovgk  = compute_ovgk_areas(stops).rename(columns = {"ovgk_class": "ovgk"})

        ovgk.to_file(f"{context.path()}/rings.gpkg", driver = "GPKG")

        return ovgk


def impute(context, df_ovgk, df, on, point_type="", chunk_size=100):
    indices = np.array_split(np.arange(len(df)), chunk_size)
    df_join = []

    print(f"Imputing ÖV Güteklasse for {len(df)} {point_type} coordinates...")
    for chunk in context.progress(indices, total=len(indices), label="Imputing ÖV Güteklasse..."):
        df_join.append(gpd.sjoin(df.iloc[chunk], df_ovgk, predicate="within")[on + ["ovgk"]])

    df_join = pd.concat(df_join)
    df_join = pd.merge(df, df_join, on=on, how="left")
    df_join.loc[df_join["ovgk"].isna(), "ovgk"] = "None"
    df_join["ovgk"] = df_join["ovgk"].astype("category")

    return df_join[on + ["ovgk"]]
