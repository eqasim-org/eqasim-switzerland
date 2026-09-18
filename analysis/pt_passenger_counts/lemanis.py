import numpy as np
import pandas as pd


_PERIOD_IS_TOTAL = True
_OFF_PEAK_HOURS  = [9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 5]
_OPERATING_HOURS = {
    "6h-9h": [6, 7, 8],
    "16h-19h": [16, 17, 18],
    "9h-16h and 19h-6h": _OFF_PEAK_HOURS,
    "All day": sorted({6, 7, 8, 16, 17, 18, *_OFF_PEAK_HOURS}),
}


STOP_NAME_TO_GTFS_CODE = {
    "ANNEMASSE": "8774549",
    "CHENE BOURG": "8516274",
    "LES TUILERIES": "8501011",
    "VERNIER": "8501007",  
    "COPPET": "8501023",
    "CREUX DE GENTHOD": "8501012",
    "GENEVE": "8501008", 
    "GENEVE CHAMPEL": "8516272",
    "GENEVE EAUX VIVES": "8516273",
    "GENEVE SECHERON": "8516283",
    "GENTHOD BELLEVUE": "8501021",
    "LANCY BACHET": "8517142",
    "LANCY PONT ROUGE": "8516155",
    "MIES": "8501014",
    "PONT CEARD": "8501013",
    "TANNAY": "8501015",
    "VERSOIX": "8501022",
    "LA PLAINE": "8501001",
    "MEYRIN": "8501006",
    "RUSSIN": "8501002",
    "SATIGNY": "8501003",
    "ZIMEYSA": "8501000",
    "POUGNY CHANCY": "8774538",
    "BONS EN CHABLAIS": "8774559",
    "EVIAN": "8774567",
    "MACHILLY": "8774558",
    "PERRIGNIER": "8774562",
    "THONON": "8774564",
    "ANNECY": "8774600",
    "GROISY THORENS LA CAILLE": "8774624",
    "LA ROCHE SUR FORON": "8774630",
    "PRINGY": "8774620",
    "REIGNIER": "8774651",
    "BONNEVILLE": "8774633",
    "CLUSES": "8774637",
    "MAGLAND": "8774641",
    "MARIGNIER": "8774634",
    "SAINT GERVAIS LES BAINS LE FAYET": "8774647",
    "SAINT PIERRE EN FAUCIGNY": "8774631",
    "SALLANCHES COMBLOUX MEGEVE": "8774643",
    "BELLEGARDE": "8774500",
}


def load_weekday_counts(path):
    df = pd.read_csv(path)
    df = df[df["day"] == "Lundi - vendredi"]

    return df.reset_index(drop = True)


def expand_to_hourly(weekday_df):
    records = []

    for _, row in weekday_df.iterrows():
        hours   = _OPERATING_HOURS[row["time"]]
        n_hours = len(hours)

        boardings_total  = row["boardings"]  if _PERIOD_IS_TOTAL else row["boardings"]  * n_hours
        alightings_total = row["alightings"] if _PERIOD_IS_TOTAL else row["alightings"] * n_hours

        for hour in hours:
            records.append({
                "operator": row["operator"], "line": row["line"], "sens": row["sens"],
                "stop": row["stop"], "hour": hour,
                "boardings": boardings_total / n_hours,
                "alightings": alightings_total / n_hours,
            })

    return pd.DataFrame.from_records(records)


def to_tpg_shape(hourly_df):
    df = hourly_df.copy()
    df["gtfs_code"] = df["stop"].map(STOP_NAME_TO_GTFS_CODE)

    unmatched = sorted(df.loc[df["gtfs_code"].isna(), "stop"].unique())

    if unmatched:
        print(f"lemanis.to_tpg_shape: dropping {len(unmatched)} unmatched stop(s): {unmatched}")

    df = df[df["gtfs_code"].notna()]

    df["line_direction"] = df["line"].astype(str)

    grouped = df.groupby(["gtfs_code", "line_direction", "hour"], as_index = False).agg(
        boardings = ("boardings", "sum"), alightings = ("alightings", "sum"),
    )

    out = grouped[["gtfs_code", "line_direction", "hour"]].copy()
    out.insert(0, "day_type", "Weekday")

    for prefix in ["boardings", "alightings"]:
        value = grouped[prefix]
        for stat in ["min", "max", "mean", "q10", "q20", "q50", "q80", "q90"]:
            out[f"{prefix}_raw_{stat}"] = value
            out[f"{prefix}_{stat}"] = value
        out[f"{prefix}_raw_std"] = 0.0
        out[f"{prefix}_std"] = 0.0

    return out


def build_line_route_geometries(gtfs_stops, path, weekday_df = None):
    if weekday_df is None:
        weekday_df = load_weekday_counts(path)

    df = weekday_df.copy()
    df["gtfs_code"] = df["stop"].map(STOP_NAME_TO_GTFS_CODE)
    df = df[df["gtfs_code"].notna()]

    coords = gtfs_stops[["stop_id", "stop_lat", "stop_lon", "geometry"]].drop_duplicates("stop_id").set_index("stop_id")
    all_codes = [c for c in df["gtfs_code"].unique() if c in coords.index]

    points = np.array([[coords.loc[c].geometry.x, coords.loc[c].geometry.y] for c in all_codes])
    _, _, vt = np.linalg.svd(points - points.mean(axis = 0))
    corridor_axis = vt[0]
    projection = {
        c: float(np.dot([coords.loc[c].geometry.x, coords.loc[c].geometry.y] - points.mean(axis = 0), corridor_axis))
        for c in all_codes
    }

    records = []
    for line, group in df.groupby("line"):
        line_codes = sorted({c for c in group["gtfs_code"] if c in projection}, key = lambda c: projection[c])
        line_coords = [(coords.loc[c].stop_lat, coords.loc[c].stop_lon) for c in line_codes]
        if len(line_coords) < 2:
            continue
        records.append({"line": line, "direction_letter": None, "coords": line_coords})

    return pd.DataFrame(records)
