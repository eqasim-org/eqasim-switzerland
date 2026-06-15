import pandas as pd
import geopandas as gpd
import os
from zipfile import ZipFile
from shapely import Point

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from plot_definition import *

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


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")

    context.config("analysis.pt.matsim_output_folder_path")
    context.config("analysis.pt.perimeter", default = "spatial/MMT/CMDP_Limites_WG84.shp")
    context.config("analysis.pt.tpg_data", default = "TPG_passenger_counts_2024")


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
    tpg_lines = pd.read_csv(f"{tpg_path}/tpg_Lignes-arrêts_2024.csv")
    tpg_stops = pd.read_csv(f"{tpg_path}/tpg_Arrets.csv", encoding = "latin1", sep = ";")

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

    line_directions                   = direction_comparison_df[(direction_comparison_df["TPG_direction"]!="missing") & (direction_comparison_df["MATSim_direction"]!="missing")]
    line_directions["line_direction"] = line_directions["line"] + "_" + line_directions["direction"]
    line_directions_names             = line_directions["MATSim_direction"].values

    # Select counts where the comparison is possible. 1. MATSim
    counts_ge = counts_ge[counts_ge["line_main_direction"].isin(line_directions_names)]
    counts_ge = counts_ge[["stop_id_gtfs", "line_name", "line_main_direction", "hour", "boardings", "alightings"]]

    counts_ge.loc[:, "stop_id_gtfs_base"] = counts_ge["stop_id_gtfs"].str.split(":").str[0]
    counts_ge = counts_ge.groupby(["stop_id_gtfs_base", "line_name", "line_main_direction", "hour"])[["boardings", "alightings"]].sum().reset_index()
    counts_ge = counts_ge.merge(line_directions.rename(columns = {"line": "line_name", "direction": "direction_letter"}), right_on = ["line_name", "MATSim_direction"], left_on = ["line_name", "line_main_direction"], how = "left")

    counts_ge["boardings"]  = counts_ge["boardings"]  / context.config("input_downsampling")
    counts_ge["alightings"] = counts_ge["alightings"] / context.config("input_downsampling")

    # Read the processed TPG counts
    tpg_mofr = pd.read_csv(f"{tpg_path}/tpg_counts_agg_workdays.csv")

    # Add MATSim counts to TPG counts
    tpg_mofr["gtfs_code"]          = tpg_mofr["gtfs_code"].astype(str)
    counts_ge["stop_id_gtfs_base"] = counts_ge["stop_id_gtfs_base"].astype(str)

    tpg_mofr = tpg_mofr.merge(counts_ge[["stop_id_gtfs_base", "line_direction", "hour", "boardings", "alightings", "MATSim_direction", "TPG_direction"]].rename(columns={"boardings": "boardings_matsim", "alightings": "alightings_matsim"}), 
                            right_on = ["stop_id_gtfs_base", "line_direction", "hour"], 
                            left_on  = ["gtfs_code", "line_direction", "hour"], how = "left")
    
    tpg_mofr["boardings_matsim"]  = tpg_mofr["boardings_matsim"].fillna(0)
    tpg_mofr["alightings_matsim"] = tpg_mofr["alightings_matsim"].fillna(0)
    tpg_mofr["gtfs_code"]         = tpg_mofr["gtfs_code"].astype(str)

    tpg_mofr = tpg_mofr.merge(gtfs_stops[["stop_id", "stop_name"]], left_on = "gtfs_code", right_on = "stop_id", how="left")

    output_path   = context.config("output_path")
    output_folder = output_path + "/" + "pt_comparison_tpg"

    # Create a folder where the results will be saved
    os.makedirs(output_folder, exist_ok=True)

    # Plot by stop-line
    stop = "Genève, gare Cornavin"
    line = "1_H"

    plot_comparison_for_stop_and_line(tpg_mofr, option = "boardings", line = line, stop = stop, output_path = output_folder + f"/comparison_{stop}_{line}.pdf")

    # Plot heatmap
    plot_heatmap_for_line(tpg_mofr, option = "boardings", line = line, output_path = output_folder + f"/heatmap_{line}.pdf")











