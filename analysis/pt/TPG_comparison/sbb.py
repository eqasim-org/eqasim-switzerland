import pandas as pd


# SBB passenger frequency of the train stations (passagierfrequenz.csv): average number of passengers
# entering or leaving the station per day, over all days (DTV), working days (DWV, "TMJO") and
# non-working days (DNWV, "TMJNO").
WORKDAY_COLUMN_KEY    = "TMJO"
NON_WORKDAY_COLUMN_KEY = "TMJNO"

_WEEKEND = ["weekend", "Saturday", "Sunday"]


def _day_column(df, day):
    key = NON_WORKDAY_COLUMN_KEY if day in _WEEKEND else WORKDAY_COLUMN_KEY
    columns = [c for c in df.columns if key in c]

    if len(columns) != 1:
        raise RuntimeError(f"Expected one SBB column containing '{key}', found {columns}")

    return columns[0]


def load_station_counts(sbb_csv_path, year = 2024, day = "workday"):
    """One row per station: uic (gtfs code), station, canton, sbb_year, sbb_total (passenger movements
    of the day type of `day`), lat, lon. A station that has no count for `year` takes the one of its
    closest year (an earlier one first)."""
    df = pd.read_csv(sbb_csv_path, sep = ";", encoding = "utf-8-sig")
    value_column = _day_column(df, day)

    df = df[df["UIC"].notna() & df[value_column].notna() & df["Geopos"].notna()].copy()
    df["uic"] = df["UIC"].astype(int).astype(str)

    coordinates = df["Geopos"].str.split(",", expand = True).astype(float)
    df["lat"], df["lon"] = coordinates[0], coordinates[1]

    df["sbb_year"] = df["Jahr_Annee_Anno"].astype(int)
    # closest year, an earlier one first in case of a tie
    df["_distance"] = (df["sbb_year"] - year).abs() * 2 + (df["sbb_year"] > year)
    df = df.sort_values(["uic", "_distance"]).drop_duplicates("uic")

    df = df.rename(columns = {"Bahnhof_Gare_Stazione": "station", "Kt_Ct_Cantone": "canton", value_column: "sbb_total"})

    return df[["uic", "station", "canton", "sbb_year", "sbb_total", "lat", "lon"]].reset_index(drop = True)
