import math
import os

import pandas as pd


# Lemanis 2024: average occupancy (people on board) of each Leman Express train at the station
# where it crosses the border, per weekday and month. It is a load on the border-crossing
# segment, not boardings/alightings at stops: the comparison with MATSim is therefore done per
# line and direction on the load of the trains when they cross the border.
FILE_NAME = "tpg_heures_passage_frontières_trains_LemanExpress_GTFS2024.xlsx"

# Months kept: the other ones are not representative of the simulated (school-term) weekday.
MONTHS = ["novembre"]

# "F": Switzerland -> France, "CH": France -> Switzerland
DIRECTION_SHEETS = {"F": "Sens Suisse->FR", "CH": "Sens France->CH"}

# Last Swiss station / first French station of each line (gtfs codes): the border is crossed
# between them. Lemanis measures the load when the train leaves the first one of the direction.
BORDER_STATIONS = {
    "L1": ("8516274", "8774549"),  # Chene-Bourg / Annemasse
    "L2": ("8516274", "8774549"),
    "L3": ("8516274", "8774549"),
    "L4": ("8516274", "8774549"),
    "L6": ("8501001", "8774538"),  # La Plaine / Pougny-Chancy
}

# Station (upper case) where the load is read in the data: stations before the border, e.g.
# Russin on the L6, are ignored
_SWISS_BORDER_STOPS = {"CHENE-BOURG", "LA PLAINE"}

_DAY_LABELS = {
    "Monday": "1 - Lundi", "Tuesday": "2 - Mardi", "Wednesday": "3 - Mercredi", "Thursday": "4 - Jeudi",
    "Friday": "5 - Vendredi", "Saturday": "6 - Samedi", "Sunday": "7 - Dimanche",
}
_WORKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_WEEKEND  = ["Saturday", "Sunday"]


def resolve_days(day):
    """specific_day_scenario ("workday", "weekend" or a day name) -> the day labels of the data."""
    if day == "workday":
        names = _WORKDAYS
    elif day == "weekend":
        names = _WEEKEND
    elif day in _DAY_LABELS:
        names = [day]
    else:
        raise ValueError(f"Unsupported specific_day_scenario: '{day}'")

    return ["2 - Mardi"]#[_DAY_LABELS[name] for name in names]


def load_border_loads(lemanis_path, day = "workday"):
    """Daily load of the trains at the border, per line and direction, as the mean over the days of
    `day` of the sum over the trains of their occupancy (averaged over MONTHS).

    Returns a data frame with the columns line, direction, lemanis_load, lemanis_trains."""
    path = os.path.join(lemanis_path, FILE_NAME)
    days = resolve_days(day)

    records = []

    for direction, sheet in DIRECTION_SHEETS.items():
        df = pd.read_excel(path, sheet_name = sheet)
        df.columns = ["year", "day", "line", "direction", "train", "stop", "month", "occupancy"]

        df = df[df["day"].isin(days) & df["month"].isin(MONTHS)]

        if direction == "F":
            df = df[df["stop"].str.upper().isin(_SWISS_BORDER_STOPS)]

        # mean over the months for each train and day, then sum over the trains of a day
        per_train = df.groupby(["day", "line", "train"], as_index = False)["occupancy"].mean()
        per_day   = per_train.groupby(["day", "line"]).agg(load = ("occupancy", "sum"), trains = ("train", "nunique"))

        full_index = pd.MultiIndex.from_product([days, sorted(per_train["line"].unique())], names = ["day", "line"])
        per_day    = per_day.reindex(full_index, fill_value = 0)

        mean = per_day.groupby("line").mean().reset_index()

        for row in mean.itertuples():
            records.append({"line": row.line, "direction": direction,
                            "lemanis_load": row.load, "lemanis_trains": row.trains})

    return pd.DataFrame(records)


def matsim_border_loads(counts, route_stops, stop_alias, input_downsampling):
    """Daily load of the simulated trains at the border, per line and direction.

    counts: MATSim passenger counts (see tpg_data.load_matsim_counts) with line_id, route_id,
    line_name, stop_id_gtfs, boardings and alightings. route_stops: {(line_id, route_id): stop ids in
    the order of the route} of the transit schedule.

    The number of passengers on board when the trains of a route leave a stop is, summed over the
    day, the cumulative boardings minus alightings of the stops up to it. So no vehicle level
    information is needed."""
    counts = counts[counts["line_name"].isin(BORDER_STATIONS)].copy()
    counts["stop_base"] = counts["stop_id_gtfs"].str.split(":").str[0]

    records = []
    unknown_routes = set()

    for (line_id, route_id, line_name), group in counts.groupby(["line_id", "route_id", "line_name"]):
        stops = route_stops.get((line_id, route_id))
        if stops is None:
            unknown_routes.add((line_id, route_id))
            continue

        sequence = [stop_alias.get(stop, stop).split(":")[0] for stop in (s.split(".")[0] for s in stops)]
        swiss, french = BORDER_STATIONS[line_name]

        if swiss not in sequence or french not in sequence:
            continue

        # load when leaving the station before the border, in the direction of the route
        direction = "F" if sequence.index(swiss) < sequence.index(french) else "CH"
        last      = sequence.index(swiss if direction == "F" else french)

        net  = group.groupby("stop_base")[["boardings", "alightings"]].sum()
        net  = net["boardings"] - net["alightings"]
        load = sum(net.get(stop, 0.0) for stop in dict.fromkeys(sequence[:last + 1]))

        records.append({"line": line_name, "direction": direction, "matsim_load": load / input_downsampling, "routes": 1})

    if unknown_routes:
        print(f"WARNING: {len(unknown_routes)} Lemanis route(s) of the counts are not in the transit schedule "
              f"and are ignored: {sorted(unknown_routes)[:5]}")

    if not records:
        return pd.DataFrame(columns = ["line", "direction", "matsim_load", "routes"])

    return pd.DataFrame(records).groupby(["line", "direction"], as_index = False).sum()


def border_marker_positions(border_df, stop_coords, fraction = 0.25, spacing_m = 150.0):
    """(lat, lon) of the map marker of each (line, direction) of border_df: on the segment between the
    last stop of the country of origin and the first one of the destination country, at `fraction` of
    the way from the origin. The lines of the same segment and direction are spread perpendicular to
    it, spacing_m apart. stop_coords: {gtfs code: (lat, lon)}."""
    df = border_df.copy()
    df["marker_lat"], df["marker_lon"] = float("nan"), float("nan")

    for (swiss, french, direction), group in df.assign(
        swiss = df["line"].map(lambda line: BORDER_STATIONS[line][0]),
        french = df["line"].map(lambda line: BORDER_STATIONS[line][1]),
    ).groupby(["swiss", "french", "direction"]):
        origin, destination = (swiss, french) if direction == "F" else (french, swiss)

        if origin not in stop_coords or destination not in stop_coords:
            print(f"WARNING: border stop(s) {origin}/{destination} not found, no map marker for {sorted(group['line'])}")
            continue

        lat0, lon0 = stop_coords[origin]
        lat1, lon1 = stop_coords[destination]
        cos_lat    = math.cos(math.radians(lat0))

        # segment in metres (east, north), and its unit normal
        east, north = (lon1 - lon0) * 111_320 * cos_lat, (lat1 - lat0) * 111_320
        length      = math.hypot(east, north) or 1.0
        normal      = (-north / length, east / length)

        lines = sorted(group["line"])
        for index, line in enumerate(lines):
            shift = (index - (len(lines) - 1) / 2) * spacing_m
            e = fraction * east  + shift * normal[0]
            n = fraction * north + shift * normal[1]

            row = group.index[group["line"] == line]
            df.loc[row, "marker_lat"] = lat0 + n / 111_320
            df.loc[row, "marker_lon"] = lon0 + e / (111_320 * cos_lat)

    return df
