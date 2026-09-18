import glob
import os
import pandas as pd
import comparison
import gtfs_utils
import interactive_map
import lemanis
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


def _load_gtfs_and_matched_counts(cfg):
    gtfs_stops = gtfs_utils.read_gtfs(cfg.gtfs_zip)
    gtfs_stops = gtfs_utils.add_missing_base_stops(gtfs_stops)

    gtfs_stops_wgs = gtfs_stops.to_crs("EPSG:4326")
    filtered_stops = gtfs_utils.filter_stops_in_shapefile(gtfs_stops_wgs, cfg.perimeter_shapefile)
    stops_in_ge    = filtered_stops["stop_id"].values

    counts_ge = tpg_data.load_matsim_counts(cfg.matsim_output_folder, stops_in_ge)

    tpg_stops = tpg_data.load_tpg_stops(cfg.tpg_data_path, gtfs_stops)

    line_directions, counts_ge = tpg_data.match_line_directions(cfg.tpg_data_path, tpg_stops, counts_ge)

    return gtfs_stops, tpg_stops, line_directions, counts_ge, stops_in_ge


def _build_lemanis_mofr(cfg, gtfs_stops, stops_in_ge):
    lemanis_stats = lemanis.to_tpg_shape(lemanis.expand_to_hourly(lemanis.load_weekday_counts(cfg.lemanis_csv_path)))
    lemanis_stats["gtfs_code"] = lemanis_stats["gtfs_code"].astype(str)
    lex_lines = sorted(lemanis_stats["line_direction"].unique())

    matsim_counts = tpg_data.load_matsim_counts(cfg.matsim_output_folder, stops_in_ge)
    matsim_counts = matsim_counts[matsim_counts["line_name"].isin(lex_lines)].copy()
    matsim_counts["stop_id_gtfs_base"] = matsim_counts["stop_id_gtfs"].str.split(":").str[0]
    matsim_counts["boardings"]  = matsim_counts["boardings"]  / cfg.input_downsampling
    matsim_counts["alightings"] = matsim_counts["alightings"] / cfg.input_downsampling

    matsim_side = matsim_counts.groupby(
        ["stop_id_gtfs_base", "line_name", "hour"], as_index = False
    )[["boardings", "alightings"]].sum()

    lemanis_mofr = lemanis_stats.merge(
        matsim_side.rename(columns = {"boardings": "boardings_matsim", "alightings": "alightings_matsim"}),
        left_on  = ["gtfs_code", "line_direction", "hour"],
        right_on = ["stop_id_gtfs_base", "line_name", "hour"], how = "left",
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


def _build_tpg_mofr(cfg, year):
    gtfs_stops, tpg_stops, _, counts_ge, stops_in_ge = _load_gtfs_and_matched_counts(cfg)

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

    if cfg.include_lemanis:
        lemanis_mofr = _build_lemanis_mofr(cfg, gtfs_stops, stops_in_ge)
        print(f"Adding {lemanis_mofr['gtfs_code'].nunique()} Leman Express stop(s) ({len(lemanis_mofr)} rows) to the comparison")
        tpg_mofr = pd.concat([tpg_mofr, lemanis_mofr], ignore_index = True)

    n_stops_before = tpg_mofr["gtfs_code"].nunique()
    tpg_mofr = comparison.filter_active_stops(
        tpg_mofr, min_avg_events = cfg.min_stop_avg_events, min_active_hours = cfg.min_stop_active_hours,
    )
    n_stops_after = tpg_mofr["gtfs_code"].nunique()
    print(
        f"Restricting to active stops (>{cfg.min_stop_avg_events} avg passenger movements "
        f"during >={cfg.min_stop_active_hours} hours/day): {n_stops_after}/{n_stops_before} stops kept"
    )

    return gtfs_stops, tpg_stops, tpg_mofr, stops_in_ge


def run_stop_line_comparison(cfg, output_dir, year, stop = "Genève, gare Cornavin", line = "1_H"):
    _, _, tpg_mofr, _ = _build_tpg_mofr(cfg, year)

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


def run_global_comparison(cfg, output_dir, year):
    gtfs_stops, tpg_stops, tpg_mofr, stops_in_ge = _build_tpg_mofr(cfg, year)

    os.makedirs(output_dir, exist_ok = True)

    stop_hour_df     = comparison.build_stop_hour_table(tpg_mofr, gtfs_stops, stops_in_ge)
    full_day_df      = comparison.build_full_day_table(stop_hour_df)
    global_hourly_df = comparison.build_global_hourly_table(stop_hour_df)

    plotting.plot_global_hourly_comparison(global_hourly_df, f"{output_dir}/global_hourly_comparison.png")

    hour_range = (cfg.map_hour_start, cfg.map_hour_end)
    windowed_stop_hour_df = stop_hour_df[stop_hour_df["hour"].between(*hour_range)]
    windowed_full_day_df  = comparison.build_full_day_table(windowed_stop_hour_df)

    interactive_map.build_hourly_map(
        stop_hour_df, f"{output_dir}/pt_stop_map_by_hour.html",
        perimeter_shapefile = cfg.perimeter_shapefile, hour_range = hour_range,
    )
    interactive_map.build_full_day_map(
        windowed_stop_hour_df, windowed_full_day_df, f"{output_dir}/pt_stop_map_6am_10pm.html",
        perimeter_shapefile = cfg.perimeter_shapefile,
    )

    line_hour_df = comparison.build_line_hour_table(tpg_mofr)
    line_hour_df = comparison.filter_line_map_lines(line_hour_df)
    windowed_line_hour_df = line_hour_df[line_hour_df["hour"].between(*hour_range)]
    line_full_day_df = comparison.build_line_full_day_table(windowed_line_hour_df)
    line_base_df = comparison.build_line_base_table(line_full_day_df)
    line_geometries_df = tpg_data.build_line_route_geometries(cfg.tpg_data_path, tpg_stops, gtfs_stops)
    if cfg.include_lemanis:
        lemanis_geometries_df = lemanis.build_line_route_geometries(gtfs_stops, cfg.lemanis_csv_path)
        line_geometries_df = pd.concat([line_geometries_df, lemanis_geometries_df], ignore_index = True)

    interactive_map.build_line_map(
        windowed_line_hour_df, line_base_df, line_geometries_df, f"{output_dir}/pt_line_map_6am_10pm.html",
        perimeter_shapefile = cfg.perimeter_shapefile,
    )

    return stop_hour_df, full_day_df, global_hourly_df
