import glob
import os
import pandas as pd
import comparison
import gtfs_utils
import interactive_map
import lemanis_2022
import lemanis_2024
import sbb
import plotting
import tpg_data


def _resolve_tpg_stats_path(processed_counts_dir, year):
    pattern    = os.path.join(processed_counts_dir, f"tpg{year}_agg_workday_*.csv")
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No processed TPG stats found for {year} (looked for {pattern}). "
            f"Run `python tpg_raw_stats.py --out ...` (2024) or "
            f"`python tpg_raw_stats_2025.py --out ...` (2025) first."
        )

    return candidates[-1]  


def _load_gtfs_and_matched_counts(data):
    gtfs_stops  = data.gtfs_stops
    stops_in_ge = data.stops_in_perimeter
    tpg_stops   = data.tpg_stops

    counts_ge = data.counts_in(stops_in_ge)

    line_directions, counts_ge = tpg_data.match_line_directions(data.cfg.tpg_data_path, tpg_stops, counts_ge)

    return gtfs_stops, tpg_stops, line_directions, counts_ge, stops_in_ge


def _load_lemanis_weekday_counts(cfg):
    """Raw weekday Lemanis counts of cfg.lemanis_year (columns of the 2022 csv: line, stop, time, boardings, alightings)."""
    if cfg.lemanis_year == 2022:
        return lemanis_2022.load_weekday_counts(os.path.join(cfg.lemanis_path, "LEMANIS_Passengers counts_2022.csv"))

    raise NotImplementedError(
        f"The Lemanis {cfg.lemanis_year} counts have no boardings/alightings by period: see run_lemanis_border_crossing"
    )


def _load_lemanis_period_stats(cfg):
    return lemanis_2022.to_period_shape(_load_lemanis_weekday_counts(cfg))


def _build_lemanis_period_mofr(data):
    cfg        = data.cfg
    gtfs_stops = data.gtfs_stops

    lemanis_stats = _load_lemanis_period_stats(cfg)
    lemanis_stats["gtfs_code"] = lemanis_stats["gtfs_code"].astype(str)
    lex_lines = sorted(lemanis_stats["line_direction"].unique())

    lemanis_codes = set(lemanis_stats["gtfs_code"])
    all_stop_ids  = gtfs_stops["stop_id"].astype(str)
    lemanis_stop_ids = all_stop_ids[all_stop_ids.str.split(":").str[0].isin(lemanis_codes)].values

    matsim_counts = data.counts_in(lemanis_stop_ids)

    matsim_lines = set(matsim_counts["line_name"])
    missing_lines = [line for line in lex_lines if line not in matsim_lines]
    if missing_lines:
        print(f"WARNING: Lemanis line(s) with no MATSim counterpart (compared against 0): {missing_lines}")
    print(f"Lemanis lines matched to MATSim line names: {sorted(set(lex_lines) - set(missing_lines))}")

    matsim_counts = matsim_counts[matsim_counts["line_name"].isin(lex_lines)].copy()
    matsim_counts["stop_id_gtfs_base"] = matsim_counts["stop_id_gtfs"].str.split(":").str[0]
    matsim_counts["boardings"]  = matsim_counts["boardings"]  / cfg.input_downsampling
    matsim_counts["alightings"] = matsim_counts["alightings"] / cfg.input_downsampling

    matsim_counts["period"] = matsim_counts["hour"].map(lemanis_2022.HOUR_TO_PERIOD)
    matsim_counts = matsim_counts[matsim_counts["period"].notna()]

    matsim_side = matsim_counts.groupby(
        ["stop_id_gtfs_base", "line_name", "period"], as_index = False
    )[["boardings", "alightings"]].sum()

    lemanis_mofr = lemanis_stats.merge(
        matsim_side.rename(columns = {"boardings": "boardings_matsim", "alightings": "alightings_matsim"}),
        left_on  = ["gtfs_code", "line_direction", "period"],
        right_on = ["stop_id_gtfs_base", "line_name", "period"], how = "left",
    )
    lemanis_mofr["boardings_matsim"]  = lemanis_mofr["boardings_matsim"].fillna(0)
    lemanis_mofr["alightings_matsim"] = lemanis_mofr["alightings_matsim"].fillna(0)

    lemanis_mofr = lemanis_mofr.merge(
        gtfs_stops[["stop_id", "stop_name"]], left_on = "gtfs_code", right_on = "stop_id", how = "left"
    )

    unmatched = lemanis_mofr["stop_name"].isna().sum()
    if unmatched:
        print(f"Dropping {unmatched}/{len(lemanis_mofr)} lemanis_mofr rows whose gtfs_code isn't in the current GTFS feed")
        lemanis_mofr = lemanis_mofr[lemanis_mofr["stop_name"].notna()]

    return lemanis_mofr[[c for c in lemanis_mofr.columns if c not in ("stop_id_gtfs_base", "line_name")]]


def _build_tpg_mofr(data, year):
    cfg = data.cfg
    gtfs_stops, tpg_stops, _, counts_ge, stops_in_ge = _load_gtfs_and_matched_counts(data)

    counts_ge = counts_ge.copy()
    counts_ge["boardings"]  = counts_ge["boardings"]  / cfg.input_downsampling
    counts_ge["alightings"] = counts_ge["alightings"] / cfg.input_downsampling

    tpg_stats_path = _resolve_tpg_stats_path(cfg.tpg_processed_counts_path, year)
    print(f"Loading {year} TPG stats from {tpg_stats_path}")
    tpg_mofr = pd.read_csv(tpg_stats_path)
    tpg_mofr["gtfs_code"] = tpg_mofr["gtfs_code"].astype(str)

    counts_ge["stop_id_gtfs_base"] = counts_ge["stop_id_gtfs_base"].astype(str)

    has_direction = tpg_mofr["line_direction"].astype(str).str.endswith(("_H", "_R")).all()

    if has_direction:
        matsim_side = counts_ge[["stop_id_gtfs_base", "line_direction", "hour", "boardings", "alightings"]]
        merge_right = ["stop_id_gtfs_base", "line_direction", "hour"]

    else:
        print("TPG stats have no direction field - aggregating MATSim boardings/alightings across both directions per line")
        matsim_side = counts_ge.groupby(["stop_id_gtfs_base", "line_alone", "hour"], as_index = False)[["boardings", "alightings"]].sum()
        merge_right = ["stop_id_gtfs_base", "line_alone", "hour"]

    tpg_mofr = tpg_mofr.merge(
        matsim_side.rename(columns = {"boardings": "boardings_matsim", "alightings": "alightings_matsim"}),
        right_on = merge_right,
        left_on  = ["gtfs_code", "line_direction", "hour"], how = "left",
    )

    tpg_mofr["boardings_matsim"]  = tpg_mofr["boardings_matsim"].fillna(0)
    tpg_mofr["alightings_matsim"] = tpg_mofr["alightings_matsim"].fillna(0)
    tpg_mofr["gtfs_code"]         = tpg_mofr["gtfs_code"].astype(str)

    tpg_mofr = tpg_mofr.merge(gtfs_stops[["stop_id", "stop_name"]], left_on = "gtfs_code", right_on = "stop_id", how = "left")

    unmatched = tpg_mofr["stop_name"].isna().sum()
    if unmatched:
        print(f"Dropping {unmatched}/{len(tpg_mofr)} tpg_mofr rows whose gtfs_code isn't in the current GTFS feed")
        tpg_mofr = tpg_mofr[tpg_mofr["stop_name"].notna()]

    lemanis_period_mofr = None
    if cfg.include_lemanis:
        lemanis_period_mofr = _build_lemanis_period_mofr(data)
        print(
            f"Built a period-level comparison for {lemanis_period_mofr['gtfs_code'].nunique()} "
            f"Leman Express stop(s) ({len(lemanis_period_mofr)} rows)"
        )

    n_stops_before = tpg_mofr["gtfs_code"].nunique()
    tpg_mofr = comparison.filter_active_stops(
        tpg_mofr, min_avg_events = cfg.min_stop_avg_events, min_active_hours = cfg.min_stop_active_hours,
    )
    n_stops_after = tpg_mofr["gtfs_code"].nunique()
    print(
        f"Restricting to active stops (>{cfg.min_stop_avg_events} avg passenger movements "
        f"during >={cfg.min_stop_active_hours} hours/day): {n_stops_after}/{n_stops_before} stops kept"
    )

    return gtfs_stops, tpg_stops, tpg_mofr, stops_in_ge, lemanis_period_mofr


def run_stop_line_comparison(data, output_dir, year, stop = "Genève, gare Cornavin", line = "1_H"):
    _, _, tpg_mofr, _, _ = _build_tpg_mofr(data, year)

    os.makedirs(output_dir, exist_ok = True)

    plotting.plot_comparison_for_stop_and_line(
        tpg_mofr, 
        option = "boardings", line = line, stop = stop,
        output_path = f"{output_dir}/comparison_{stop}_{line}.pdf",
    )

    plotting.plot_heatmap_for_line(
        tpg_mofr, 
        option = "boardings", line = line,
        output_path = f"{output_dir}/heatmap_{line}.pdf",
    )

    return tpg_mofr


def run_stop_all_lines(data, output_dir, stop = "Annemasse"):
    cfg        = data.cfg
    gtfs_stops = data.gtfs_stops

    query      = gtfs_utils._normalize_stop_name(stop)
    names      = gtfs_stops["stop_name"].astype(str)
    normalized = names.map(gtfs_utils._normalize_stop_name)

    selected = gtfs_stops[normalized == query]
    if selected.empty:
        selected = gtfs_stops[normalized.str.contains(query, regex = False)]
    if selected.empty:
        raise ValueError(f"No stop whose name matches {stop!r} in the GTFS / simulated schedule")

    stop_names = sorted(selected["stop_name"].astype(str).unique())
    print(f"Stop {stop!r} matched {len(selected)} stop id(s) with name(s): {stop_names}")

    counts = data.counts_in(selected["stop_id"].astype(str).values)
    if counts.empty:
        raise ValueError(f"No MATSim passenger counts at stop {stop!r}")

    counts = counts.groupby(["line_name", "hour"], as_index = False)[["boardings", "alightings"]].sum()
    counts["boardings"]  = counts["boardings"]  / cfg.input_downsampling
    counts["alightings"] = counts["alightings"] / cfg.input_downsampling

    os.makedirs(output_dir, exist_ok = True)
    slug = gtfs_utils._normalize_stop_name(stop)

    counts.to_csv(f"{output_dir}/stop_all_lines_{slug}.csv", index = False)
    lemanis_df = None
    if cfg.include_lemanis:
        stop_bases = set(selected["stop_id"].astype(str).str.split(":").str[0])
        lemanis_stats = _load_lemanis_period_stats(cfg)
        lemanis_stats = lemanis_stats[lemanis_stats["gtfs_code"].astype(str).isin(stop_bases)]

        if lemanis_stats.empty:
            print(f"No Lemanis data at stop {stop!r}")
        else:
            lemanis_df = lemanis_stats.rename(columns = {"line_direction": "line_name",
                                                        "boardings_raw_mean": "boardings",
                                                        "alightings_raw_mean": "alightings"})
            lemanis_df = lemanis_df[["line_name", "period", "boardings", "alightings"]]
            lemanis_df.to_csv(f"{output_dir}/stop_all_lines_{slug}_lemanis.csv", index = False)
            print(f"Lemanis comparison available for line(s): {sorted(lemanis_df['line_name'].unique())}")

    plotting.plot_stop_all_lines(counts, ", ".join(stop_names), f"{output_dir}/stop_all_lines_{slug}.png",
                                 lemanis_df = lemanis_df)

    return counts


def run_sbb_stations(data, output_dir, year = 2024, day = "workday"):
    """MATSim passenger movements (boardings + alightings, all lines) of the train stations of the
    perimeter against the SBB passenger frequency of the stations."""
    import geopandas as gpd

    cfg = data.cfg

    stations = sbb.load_station_counts(cfg.sbb_path, year, day)

    stations_gdf = gpd.GeoDataFrame(stations, geometry = gpd.points_from_xy(stations["lon"], stations["lat"]), crs = "EPSG:4326")
    perimeter    = gpd.read_file(cfg.perimeter_shapefile).to_crs("EPSG:4326")
    stations     = stations[stations_gdf.within(perimeter.geometry.union_all() if hasattr(perimeter.geometry, "union_all") else perimeter.geometry.unary_union).values].reset_index(drop = True)

    if stations.empty:
        print(f"WARNING: no SBB station of {cfg.sbb_path} lies inside the perimeter {cfg.perimeter_shapefile}: nothing to map")

    counts    = data.matsim_counts
    movements = counts.groupby(counts["stop_id_gtfs"].str.split(":").str[0])[["boardings", "alightings"]].sum() / cfg.input_downsampling
    movements = movements["boardings"] + movements["alightings"]

    stations["matsim_total"]   = stations["uic"].map(movements).fillna(0.0)
    stations["pct_of_sbb"]     = 100 * stations["matsim_total"] / stations["sbb_total"].clip(lower = 1.0)
    stations["sbb_total"]      = stations["sbb_total"].astype(float)

    print(f"SBB station counts ({day}, {year}): {len(stations)} station(s) in the perimeter, "
          f"{(stations['matsim_total'] > 0).sum()} with MATSim passengers")

    os.makedirs(output_dir, exist_ok = True)
    stations.to_csv(f"{output_dir}/sbb_station_counts.csv", index = False)

    return stations


def run_lemanis_border_crossing(data, output_dir, day = "workday"):
    """Load of the Leman Express trains when they cross the border, per line and direction: MATSim
    against the Lemanis 2024 occupancies (mean over the days of `day`)."""
    cfg = data.cfg

    matsim   = lemanis_2024.matsim_border_loads(data.matsim_counts, data.route_stops, data.stop_alias, cfg.input_downsampling)
    observed = lemanis_2024.load_border_loads(cfg.lemanis_path, day)

    result = observed.merge(matsim, on = ["line", "direction"], how = "outer")
    result["matsim_load"] = result["matsim_load"].fillna(0.0)
    result["pct_of_lemanis"] = 100 * result["matsim_load"] / result["lemanis_load"].clip(lower = 1.0)
    result = result.sort_values(["line", "direction"]).reset_index(drop = True)

    result = lemanis_2024.border_marker_positions(result, data.stop_coordinates)

    print(f"Lemanis {cfg.lemanis_year} border crossing loads ({day}):")
    print(result.to_string(index = False))

    os.makedirs(output_dir, exist_ok = True)
    result.to_csv(f"{output_dir}/lemanis_{cfg.lemanis_year}_border_crossing_load.csv", index = False)
    plotting.plot_border_crossing_loads(
        result, f"Leman Express load at the border, {day} (Lemanis {cfg.lemanis_year} vs MATSim)",
        f"{output_dir}/lemanis_{cfg.lemanis_year}_border_crossing_load.png",
    )

    return result


def run_global_comparison(data, output_dir, year, border_crossing_df = None, sbb_station_df = None):
    cfg = data.cfg
    gtfs_stops, tpg_stops, tpg_mofr, stops_in_ge, lemanis_period_mofr = _build_tpg_mofr(data, year)

    os.makedirs(output_dir, exist_ok = True)

    stop_hour_df     = comparison.build_stop_hour_table(tpg_mofr, gtfs_stops, stops_in_ge)
    full_day_df      = comparison.build_full_day_table(stop_hour_df)
    global_hourly_df = comparison.build_global_hourly_table(stop_hour_df)

    plotting.plot_global_hourly_comparison(global_hourly_df, f"{output_dir}/global_hourly_comparison.png")

    hour_range = (cfg.map_hour_start, cfg.map_hour_end)
    windowed_stop_hour_df = stop_hour_df[stop_hour_df["hour"].between(*hour_range)]
    windowed_full_day_df  = comparison.build_full_day_table(windowed_stop_hour_df)

    lemanis_stop_period_df   = None
    lemanis_stop_full_day_df = None

    if lemanis_period_mofr is not None and not lemanis_period_mofr.empty:
        lemanis_stop_period_df   = comparison.build_lemanis_stop_period_table(lemanis_period_mofr, gtfs_stops)
        lemanis_stop_full_day_df = comparison.build_lemanis_stop_full_day_table(lemanis_stop_period_df)

    interactive_map.build_hourly_map(
        stop_hour_df, f"{output_dir}/pt_stop_map_by_hour.html",
        perimeter_shapefile = cfg.perimeter_shapefile, hour_range = hour_range,
    )
    interactive_map.build_full_day_map(
        windowed_stop_hour_df, windowed_full_day_df, f"{output_dir}/pt_stop_map_6am_10pm.html",
        perimeter_shapefile      = cfg.perimeter_shapefile,
        lemanis_stop_period_df   = lemanis_stop_period_df,
        lemanis_stop_full_day_df = lemanis_stop_full_day_df,
        border_crossing_df = border_crossing_df, sbb_station_df = sbb_station_df,
    )

    line_hour_df          = comparison.build_line_hour_table(tpg_mofr)
    line_hour_df          = comparison.filter_line_map_lines(line_hour_df)
    windowed_line_hour_df = line_hour_df[line_hour_df["hour"].between(*hour_range)]
    line_full_day_df      = comparison.build_line_full_day_table(windowed_line_hour_df)
    line_base_df          = comparison.build_line_base_table(line_full_day_df)

    lemanis_line_period_df = None
    if lemanis_period_mofr is not None and not lemanis_period_mofr.empty:
        lemanis_line_period_df   = comparison.build_lemanis_line_period_table(lemanis_period_mofr)
        lemanis_line_full_day_df = comparison.build_lemanis_line_full_day_table(lemanis_line_period_df)
        lemanis_line_base_df     = comparison.build_line_base_table(lemanis_line_full_day_df)
        line_base_df             = pd.concat([line_base_df, lemanis_line_base_df], ignore_index = True)

    line_geometries_df = tpg_data.build_line_route_geometries(cfg.tpg_data_path, tpg_stops, gtfs_stops)
    if cfg.include_lemanis:
        lemanis_geometries_df = lemanis_2022.build_line_route_geometries(gtfs_stops, None, weekday_df = _load_lemanis_weekday_counts(cfg))
        line_geometries_df    = pd.concat([line_geometries_df, lemanis_geometries_df], ignore_index = True)

    interactive_map.build_line_map(
        windowed_line_hour_df, line_base_df, line_geometries_df, f"{output_dir}/pt_line_map_6am_10pm.html",
        perimeter_shapefile = cfg.perimeter_shapefile,
        lemanis_line_period_df = lemanis_line_period_df,
    )

    return stop_hour_df, full_day_df, global_hourly_df
