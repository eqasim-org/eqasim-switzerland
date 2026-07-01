import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from shapely import Point

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


REQUIRED_SLOTS = [
    "agency", "stops", "routes", "trips", "stop_times"
]


OPTIONAL_SLOTS = [
    "calendar", "calendar_dates", "fare_attributes", "fare_rules",
    "shapes", "frequencies", "transfers", "pathways", "levels",
    "feed_info", "translations", "attributions"
]


def read_gtfs(context):
    gtfs_path = context.config("data_path") + "/gtfs/gtfs_fp2024_2024-11-11.zip"

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

        if not "parent_station" in df_stops:
            print("Missing parent_station in stops, setting to NaN")
            df_stops["parent_station"] = np.nan

        df_stops["location_type"]  = df_stops["location_type"].fillna(0).astype(int)
        df_stops["parent_station"] = df_stops["parent_station"].fillna("").astype(str)

        gtfs_geometry = [Point(xy) for xy in zip(df_stops["stop_lon"], df_stops["stop_lat"])]
        gdf = gpd.GeoDataFrame(df_stops, geometry=gtfs_geometry, crs="EPSG:4326")
        gdf = gdf.to_crs("EPSG:2056")
        
        return gdf


def filter_stops_in_shapefile(stops_gdf, shapefile_path):
    polygon_gdf = gpd.read_file(shapefile_path)

    # Ensure same CRS
    if stops_gdf.crs != polygon_gdf.crs:
        stops_gdf = stops_gdf.to_crs(polygon_gdf.crs)

    filtered = gpd.sjoin(stops_gdf, polygon_gdf, predicate="within", how="inner")
    filtered = filtered[["stop_id", "geometry"]]

    return filtered


def plot_annual_counts_by_stop(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    valid_stops = df[df["n_obs"] >= 1000]["gtfs_code"].unique()
    print(f"Plotting annual counts for {len(valid_stops)} stops with n_obs >= 1000")

    for gtfs_code in valid_stops:
        stop_df   = df[df["gtfs_code"] == gtfs_code].sort_values("bin_idx")
        stop_name = stop_df["stop_name"].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(stop_name)

        for ax, median_col, matsim_col, var_col, label in [
            (axes[0], "Montees_median",   "Montees_matsim",   "Montees_var",   "Boardings"),
            (axes[1], "Descentes_median", "Descentes_matsim", "Descentes_var", "Alightings"),
        ]:
            std = np.sqrt(stop_df[var_col].clip(lower=0))
            lo  = stop_df[median_col] - 1.96 * std    # Compute min and max of the 95%-confidence interval assuming normal distribution
            hi  = stop_df[median_col] + 1.96 * std

            ax.vlines(stop_df["bin_idx"], lo.clip(lower = 0), hi, color="lightgray", linewidth=8, zorder=1)
            ax.scatter(stop_df["bin_idx"], stop_df[median_col], color="gray", marker="_", s=60, linewidths=1.5, zorder=2, label="TPG median")
            ax.scatter(stop_df["bin_idx"], stop_df[matsim_col], color="steelblue", s=30, zorder=3, label="MATSim")

            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Count")
            ax.set_title(label)
            ax.legend()

        plt.tight_layout()
        safe_name = str(stop_name).replace("/", "-").replace(" ", "_")
        fig.savefig(f"{output_folder}/annual_{safe_name}.pdf")
        plt.close(fig)


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")

    context.config("analysis.pt.matsim_output_folder_path")
    context.config("analysis.pt.perimeter", default = "spatial/MMT/CMDP_Limites_WG84.shp")
    context.config("analysis.pt.tpg_data", default = "TPG_passenger_counts")


def execute(context):
    data_path = context.config("data_path")

    # GTFS stops -> gdf
    gtfs_stops = read_gtfs(context)

    # Filter stops in perimeter
    shapefile_path = data_path + "/" + context.config("analysis.pt.perimeter")
    gtfs_stops_wsg = gtfs_stops.to_crs("EPSG:4326")
    filtered_stops = filter_stops_in_shapefile(gtfs_stops_wsg, shapefile_path)
    stops_in_GE    = filtered_stops["stop_id"].values

    # Read MATSim counts
    matsim_output_path   = context.config("analysis.pt.matsim_output_folder_path")
    matsim_pxcounts_path = f"{matsim_output_path}/pt_passenger_counts.csv.gz"

    counts                = pd.read_csv(matsim_pxcounts_path, sep = ";")
    counts.loc[:, "hour"] = counts["time_bin"].str.extract(r"^(\d{1,2})").astype(int)

    counts     = counts[counts["hour"]<25]
    counts     = counts.groupby(["stop_id", "line_id", "line_name", "route_id", "vehicle_id", "hour", "route_direction", "line_main_direction"])[["boardings", "alightings"]].agg("sum").reset_index()
    counts_agg = counts.copy()

    counts_agg["stop_id_gtfs"] = counts_agg["stop_id"].str.split(".").str[0].astype(str)

    # Filter out counts on stops outside the perimeter
    counts_ge = counts_agg[counts_agg["stop_id_gtfs"].isin(stops_in_GE)]

    # Read TPG lines and stop name descriptions
    tpg_path  = data_path + "/" + context.config("analysis.pt.tpg_data")
    tpg_lines = pd.read_csv(f"{tpg_path}/counts2024/tpg_Lignes-arrêts_2024.csv")
    tpg_stops = pd.read_csv(f"{tpg_path}/counts2024/tpg_Arrets.csv", encoding = "latin1", sep = ";")

    tpg_stops.columns = ["stop_code", "lon", "lat", "country", "name", "municipality", "gtfs_code", "date1", "date2"]
    tpg_stops = tpg_stops[["stop_code", "gtfs_code", "name"]]
    tpg_stops = tpg_stops[tpg_stops["gtfs_code"].notna()]
    tpg_stops["gtfs_code"] = tpg_stops["gtfs_code"].astype(int).astype(str)
    tpg_stops = tpg_stops.merge(gtfs_stops[["stop_id", "stop_name", "geometry"]], left_on = "gtfs_code", right_on = "stop_id", how = "left")
    tpg_stops = tpg_stops[tpg_stops["stop_name"].notna()]

    tpg_lines = tpg_lines.merge(tpg_stops[["stop_code", "gtfs_code", "stop_name"]], left_on = "Code", right_on = "stop_code", how = "left")
    tpg_lines = tpg_lines[tpg_lines["stop_code"].notna()]

    tpg_directions = tpg_lines.groupby(["Ligne", "Sens"]).agg(first_stop = ("stop_name", "first"), last_stop = ("stop_name", "last")).reset_index()

    tpg_directions.columns = ["line", "direction", "first_stop", "last_stop"]
    tpg_directions["direction_extended"] = tpg_directions["first_stop"] + "->" + tpg_directions["last_stop"]
    tpg_directions["direction"]          = tpg_directions["direction"].map({"Aller": "H", "Retour": "R"})

    # Compute the matching between MATSim line directions and TPG directions
    matsim_directions = counts_ge[["line_name", "route_direction", "line_main_direction"]].drop_duplicates()

    TPG_lines = list(set(tpg_directions["line"].values.tolist()).union(set(matsim_directions["line_name"].values.tolist())))

    records = []

    for line in TPG_lines:
        for direction in ["H", "R"]:
            row = {
                "line": line,
                "direction": direction,
            }

            tpg_match = tpg_directions[
                (tpg_directions["line"] == line) &
                (tpg_directions["direction"] == direction)
            ]
            
            if not tpg_match.empty:
                row["TPG_direction"] = tpg_match["direction_extended"].values[0]
            else:
                row["TPG_direction"] = "missing"


            matsim_match = matsim_directions[
                (matsim_directions["line_name"] == line) &
                (matsim_directions["line_main_direction"] == row["TPG_direction"])
            ]
            
            if not matsim_match.empty:
                row["MATSim_direction"] = matsim_match["line_main_direction"].values[0]
            else:
                matsim_candidates = matsim_directions[matsim_directions["line_name"] == line]

                if not matsim_candidates.empty and row["TPG_direction"] != "missing":
                    tpg_parts = row["TPG_direction"].split("->")
                    tpg_first = tpg_parts[0].strip() if len(tpg_parts) > 0 else ""
                    tpg_last  = tpg_parts[1].strip() if len(tpg_parts) > 1 else ""

                    fuzzy_match = None
                    for _, candidate in matsim_candidates.iterrows():
                        matsim_parts = str(candidate["line_main_direction"]).split("->")
                        matsim_first = matsim_parts[0].strip() if len(matsim_parts) > 0 else ""
                        matsim_last  = matsim_parts[1].strip() if len(matsim_parts) > 1 else ""

                        first_matches = tpg_first and (tpg_first in matsim_first or matsim_first in tpg_first)
                        last_matches  = tpg_last  and (tpg_last  in matsim_last  or matsim_last  in tpg_last)

                        if first_matches or last_matches:
                            fuzzy_match = candidate["line_main_direction"]
                            break

                    row["MATSim_direction"] = fuzzy_match if fuzzy_match is not None else "missing"
                else:
                    row["MATSim_direction"] = "missing"

            records.append(row)

    direction_comparison_df          = pd.DataFrame(records)
    direction_comparison_df["match"] = direction_comparison_df["TPG_direction"] == direction_comparison_df["MATSim_direction"]

    line_directions                   = direction_comparison_df[(direction_comparison_df["TPG_direction"]!="missing") & (direction_comparison_df["MATSim_direction"]!="missing")].copy()
    line_directions["line_direction"] = line_directions["line"] + "_" + line_directions["direction"]
    line_directions_names             = line_directions["MATSim_direction"].values

    # Select counts where the comparison is possible. 1. MATSim
    counts_ge = counts_ge[counts_ge["line_main_direction"].isin(line_directions_names)]
    counts_ge = counts_ge[["stop_id_gtfs", "line_name", "line_main_direction", "hour", "boardings", "alightings"]]

    counts_ge.loc[:, "stop_id_gtfs_base"] = counts_ge["stop_id_gtfs"].str.split(":").str[0]
    counts_ge = counts_ge.groupby(["stop_id_gtfs_base", "line_name", "line_main_direction", "hour"])[["boardings", "alightings"]].sum().reset_index()
    counts_ge = counts_ge.merge(line_directions.rename(columns = {"line": "line_name", "direction": "direction_letter"}), right_on = ["line_name", "MATSim_direction"], left_on = ["line_name", "line_main_direction"], how = "left")

    #counts_ge["boardings"]  = counts_ge["boardings"]  / context.config("input_downsampling")
    #counts_ge["alightings"] = counts_ge["alightings"] / context.config("input_downsampling")

    # Read the processed TPG counts
    tpg_annual_combined = pd.read_csv(f"{tpg_path}/counts2025/annual_combined.csv")
    tpg_annual_by_line  = pd.read_csv(f"{tpg_path}/counts2025/annual_byline.csv")

    tpg_monthly_combined = pd.read_csv(f"{tpg_path}/counts2025/monthly_combined.csv")
    tpg_monthly_by_line  = pd.read_csv(f"{tpg_path}/counts2025/monthly_byline.csv")

    valid_codes = tpg_stops["stop_code"]

    for name, df in [
        ("tpg_annual_combined",  tpg_annual_combined),
        ("tpg_annual_by_line",   tpg_annual_by_line),
        ("tpg_monthly_combined", tpg_monthly_combined),
        ("tpg_monthly_by_line",  tpg_monthly_by_line),
    ]:
        removed = (~df["CodeLong"].isin(valid_codes)).sum()
        pct     = 100 * removed / len(df) if len(df) > 0 else 0
        print(f"[{name}] Removed {removed}/{len(df)} rows ({pct:.1f}%) where CodeLong not in tpg_stops")

    tpg_annual_combined  = tpg_annual_combined[tpg_annual_combined["CodeLong"].isin(valid_codes)]
    tpg_annual_by_line   = tpg_annual_by_line[tpg_annual_by_line["CodeLong"].isin(valid_codes)]
    tpg_monthly_combined = tpg_monthly_combined[tpg_monthly_combined["CodeLong"].isin(valid_codes)]
    tpg_monthly_by_line  = tpg_monthly_by_line[tpg_monthly_by_line["CodeLong"].isin(valid_codes)]

    stop_info = tpg_stops[["stop_code", "gtfs_code", "stop_name"]]
    
    tpg_annual_combined  = tpg_annual_combined.merge(stop_info,  left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")
    tpg_annual_by_line   = tpg_annual_by_line.merge(stop_info,   left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")
    tpg_monthly_combined = tpg_monthly_combined.merge(stop_info, left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")
    tpg_monthly_by_line  = tpg_monthly_by_line.merge(stop_info,  left_on="CodeLong", right_on="stop_code", how="left").drop(columns="stop_code")

    counts_ge["stop_id_gtfs_base"]    = counts_ge["stop_id_gtfs_base"].astype(str)
    tpg_annual_combined["gtfs_code"]  = tpg_annual_combined["gtfs_code"].astype(str)
    tpg_annual_by_line["gtfs_code"]   = tpg_annual_by_line["gtfs_code"].astype(str)
    tpg_monthly_combined["gtfs_code"] = tpg_monthly_combined["gtfs_code"].astype(str)
    tpg_monthly_by_line["gtfs_code"]  = tpg_monthly_by_line["gtfs_code"].astype(str)

    # Multiple CodeLong can share the same gtfs_code: aggregate to gtfs_code level.
    # Sum medians (mean of sum = sum of means) and sum variances (var of sum = sum of vars, assuming independence).
    tpg_annual_combined = tpg_annual_combined.groupby(["gtfs_code", "bin_idx"]).agg(
        stop_name        = ("stop_name",        "first"),
        Montees_median   = ("Montees_median",   "sum"),
        Montees_var      = ("Montees_var",       "sum"),
        Descentes_median = ("Descentes_median", "sum"),
        Descentes_var    = ("Descentes_var",    "sum"),
        n_obs            = ("n_obs",            "min"),
    ).reset_index()

    counts_ge_agg_by_stop = counts_ge[
        ["stop_id_gtfs_base", "line_direction", "hour", "boardings", "alightings"]
        ].rename(columns={"boardings": "boardings_matsim", "alightings": "alightings_matsim"}).groupby(
            ["stop_id_gtfs_base", "hour"]).agg(
                Montees_matsim = ("boardings_matsim", "sum"),
                Descentes_matsim = ("alightings_matsim", "sum")
            )

    tpg_annual_combined = tpg_annual_combined.merge(counts_ge_agg_by_stop,
                            right_on = ["stop_id_gtfs_base", "hour"], 
                            left_on  = ["gtfs_code", "bin_idx"], how = "left")
    
    tpg_annual_combined["Montees_matsim"]   = tpg_annual_combined["Montees_matsim"].fillna(0)
    tpg_annual_combined["Descentes_matsim"] = tpg_annual_combined["Descentes_matsim"].fillna(0)

    tpg_annual_combined = tpg_annual_combined[["gtfs_code", "stop_name", "bin_idx",
                                               "Montees_median", "Montees_var", "Montees_matsim",
                                               "Descentes_median", "Descentes_var", "Descentes_matsim",
                                               "n_obs"]]
    
    output_path   = context.config("output_path")
    output_folder = os.path.join(output_path, "pt_comparison_tpg_2025")
    os.makedirs(output_folder, exist_ok=True)

    plot_annual_counts_by_stop(tpg_annual_combined, os.path.join(output_folder, "annual_by_stop"))