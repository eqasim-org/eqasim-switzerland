import pandas as pd


def load_tpg_stops(tpg_data_path, gtfs_stops):
    tpg_stops = pd.read_csv(f"{tpg_data_path}/TPG_stops_info/tpg_Arrets.csv", encoding = "latin1", sep = ";")

    tpg_stops.columns = ["stop_code", "lon", "lat", "country", "name", "municipality", "gtfs_code", "date1", "date2"]
    tpg_stops = tpg_stops[["stop_code", "gtfs_code", "name"]]
    tpg_stops = tpg_stops[tpg_stops["gtfs_code"].notna()]
    tpg_stops["gtfs_code"] = tpg_stops["gtfs_code"].astype(int).astype(str)
    tpg_stops = tpg_stops.merge(gtfs_stops[["stop_id", "stop_name", "geometry"]], left_on = "gtfs_code", right_on = "stop_id", how = "left")
    tpg_stops = tpg_stops[tpg_stops["stop_name"].notna()]

    return tpg_stops


def load_matsim_counts(matsim_output_folder, stops_in_ge):
    matsim_pxcounts_path = f"{matsim_output_folder}/pt_passenger_counts.csv.gz"

    counts = pd.read_csv(matsim_pxcounts_path, sep = ";")
    counts.loc[:, "hour"] = counts["time_bin"].str.extract(r"^(\d{1,2})").astype(int)

    counts = counts[counts["hour"] < 25]
    counts = counts.groupby(
        ["stop_id", "line_id", "line_name", "route_id", "vehicle_id", "hour", "route_direction", "line_main_direction"]
    )[["boardings", "alightings"]].agg("sum").reset_index()

    counts["stop_id_gtfs"] = counts["stop_id"].str.split(".").str[0].astype(str)

    counts_ge = counts[counts["stop_id_gtfs"].isin(stops_in_ge)]

    return counts_ge


def match_line_directions(tpg_data_path, tpg_stops, counts_ge):
    tpg_lines = pd.read_csv(f"{tpg_data_path}/TPG_stops_info/tpg_Lignes-arrêts_2024.csv")
    tpg_lines = tpg_lines.merge(tpg_stops[["stop_code", "gtfs_code", "stop_name"]], left_on = "Code", right_on = "stop_code", how = "left")
    tpg_lines = tpg_lines[tpg_lines["stop_code"].notna()]

    tpg_directions = tpg_lines.groupby(["Ligne", "Sens"]).agg(
        first_stop = ("stop_name", "first"), last_stop = ("stop_name", "last")
    ).reset_index()

    tpg_directions.columns = ["line", "direction", "first_stop", "last_stop"]
    tpg_directions["direction_extended"] = tpg_directions["first_stop"] + "->" + tpg_directions["last_stop"]
    tpg_directions["direction"] = tpg_directions["direction"].map({"Aller": "H", "Retour": "R"})

    matsim_directions = counts_ge[["line_name", "route_direction", "line_main_direction"]].drop_duplicates()

    tpg_line_names = list(set(tpg_directions["line"].values.tolist()).union(set(matsim_directions["line_name"].values.tolist())))

    records = []

    for line in tpg_line_names:
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
                    tpg_first  = tpg_parts[0].strip() if len(tpg_parts) > 0 else ""
                    tpg_last   = tpg_parts[1].strip() if len(tpg_parts) > 1 else ""

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

    direction_comparison_df = pd.DataFrame(records)
    direction_comparison_df["match"] = direction_comparison_df["TPG_direction"] == direction_comparison_df["MATSim_direction"]

    line_directions = direction_comparison_df[
        (direction_comparison_df["TPG_direction"] != "missing") &
        (direction_comparison_df["MATSim_direction"] != "missing")
    ].copy()
    line_directions["line_direction"] = line_directions["line"] + "_" + line_directions["direction"]
    line_directions_names = line_directions["MATSim_direction"].values

    counts_ge = counts_ge[counts_ge["line_main_direction"].isin(line_directions_names)]
    counts_ge = counts_ge[["stop_id_gtfs", "line_name", "line_main_direction", "hour", "boardings", "alightings"]]

    counts_ge = counts_ge.copy()
    counts_ge["stop_id_gtfs_base"] = counts_ge["stop_id_gtfs"].str.split(":").str[0]
    counts_ge = counts_ge.groupby(["stop_id_gtfs_base", "line_name", "line_main_direction", "hour"])[["boardings", "alightings"]].sum().reset_index()
    counts_ge = counts_ge.merge(
        line_directions.rename(columns = {"line": "line_name", "direction": "direction_letter"}),
        right_on = ["line_name", "MATSim_direction"], left_on = ["line_name", "line_main_direction"], how = "left"
    )

    counts_ge["line_alone"] = counts_ge["line_name"]

    return line_directions, counts_ge


_DIRECTION_LETTER = {"Aller": "H", "Retour": "R"}


def build_line_route_geometries(tpg_data_path, tpg_stops, gtfs_stops):

    tpg_lines = pd.read_csv(f"{tpg_data_path}/TPG_stops_info/tpg_Lignes-arrêts_2024.csv")
    tpg_lines = tpg_lines[tpg_lines["Occasionnel"] == 0]
    tpg_lines = tpg_lines.reset_index(names = "_row_order")

    tpg_lines = tpg_lines.merge(tpg_stops[["stop_code", "gtfs_code"]], left_on = "Code", right_on = "stop_code", how = "inner")
    tpg_lines = tpg_lines.merge(gtfs_stops[["stop_id", "stop_lat", "stop_lon"]], left_on = "gtfs_code", right_on = "stop_id", how = "inner")
    tpg_lines = tpg_lines.sort_values("_row_order")

    tpg_lines["line"] = tpg_lines["Ligne"].astype(str)
    tpg_lines["direction_letter"] = tpg_lines["Sens"].map(_DIRECTION_LETTER)
    tpg_lines = tpg_lines[tpg_lines["direction_letter"].notna()]

    records = []
    for (line, direction_letter), group in tpg_lines.groupby(["line", "direction_letter"], sort = False):
        coords = list(zip(group["stop_lat"], group["stop_lon"]))
        if len(coords) < 2:
            continue
        records.append({
            "line": line,
            "direction_letter": direction_letter,
            "first_stop": group["Arrêt"].iloc[0],
            "last_stop": group["Arrêt"].iloc[-1],
            "coords": coords,
        })

    return pd.DataFrame(records)
