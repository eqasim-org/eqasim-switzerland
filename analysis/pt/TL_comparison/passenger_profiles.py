"""Who travels on the lines of the line comparison maps (line_comparison): the distributions of the attributes of the
simulated passengers (age, sex, trip purpose, employment, travelcard, car availability, income, home municipality), as
in GTFS_router (gtfs_router/profiles.py).

Each public transport leg of output_legs.csv.gz on a compared line is split into the connections of its route between
its boarding and alighting stops; its person (output_persons.csv.gz) and the purpose of its trip (the activity at the end
of output_trips.csv.gz) count on each connection (all day), and once for the line in its period of the day (morning peak,
off-peak, evening peak, by the boarding time). The reference is the same distribution over all the public transport legs
of the simulation. The counts are simulated people (not scaled by the sample): the page shows a profile from MIN_PEOPLE."""
import os

import numpy as np
import pandas as pd
from analysis.output_files import find_simulation_output

MIN_PEOPLE = 30
PERIODS = ["Morning peak (6-9 h)", "Off-peak", "Evening peak (16-19 h)"]

DIMENSIONS = [
    ("Age", ["0–15", "16–24", "25–44", "45–64", "65+"]),
    ("Sex", ["Women", "Men"]),
    ("Trip purpose", ["Work", "Education", "Shopping", "Leisure", "Going home", "Other"]),
    ("Employment", ["Employed", "Not employed"]),
    ("Travelcard", ["GA", "Half-fare", "Regional pass", "Regional + half-fare", "None"]),
    ("Car available", ["Always", "Never"]),
    ("Income per person (CHF/month)", ["< 3000", "3000–4999", "5000–6999", "7000+"]),
    ("Home", ["Urban core", "Urban", "Suburban", "Rural", "Abroad"]),
]
OFFSETS = np.cumsum([0] + [len(labels) for _, labels in DIMENSIONS])
PURPOSES = {"work": 0, "work_secondary": 0, "education": 1, "education_secondary": 1, "shop": 2, "leisure": 3,
            "home": 4, "home_secondary": 4}
PERSON_COLUMNS = ["person", "age", "sex", "employed", "subscriptions", "carAvail", "incomePerCapita", "municipalityType"]


def period_of(seconds):
    """0 morning peak (06:00-09:00), 2 evening peak (16:00-19:00), 1 off-peak."""
    hour = int(seconds) // 3600 % 24
    return 0 if 6 <= hour < 9 else 2 if 16 <= hour < 19 else 1


def _seconds(hms):
    h, m, s = (int(x) for x in str(hms).split(":"))
    return h * 3600 + m * 60 + s


def _codes(persons, purpose):
    """The category of each dimension (-1: unknown), columns in the order of DIMENSIONS."""
    # Optional person attributes must stay unknown rather than failing the report.
    age = pd.to_numeric(persons.get("age", pd.Series(index=persons.index, dtype=float)), errors = "coerce")
    income = pd.to_numeric(persons.get("incomePerCapita", pd.Series(index=persons.index, dtype=float)), errors = "coerce")
    pick = lambda column, mapping: persons[column].map(mapping).fillna(-1).astype(int).to_numpy() if column in persons \
        else np.full(len(persons), -1)
    return np.column_stack([
        np.where(age.isna(), -1, np.digitize(age.fillna(0), [16, 25, 45, 65])),
        pick("sex", {"f": 0, "m": 1}),
        purpose.map(PURPOSES).fillna(5).astype(int).to_numpy(),
        pick("employed", {"true": 0, "false": 1}),
        pick("subscriptions", {"GA": 0, "HT": 1, "VA": 2, "VA+HT": 3, "none": 4}),
        pick("carAvail", {"always": 0, "never": 1}),
        np.where(income.isna(), -1, np.digitize(income.fillna(0), [3000, 5000, 7000])),
        pick("municipalityType", {"urbancore": 0, "urban": 1, "suburban": 2, "rural": 3, "crossborder": 4}),
    ])


def _count(codes):
    """Number of people per category (a list of OFFSETS[-1] integers)."""
    counts = np.zeros(OFFSETS[-1], dtype = int)
    for d in range(len(DIMENSIONS)):
        c = codes[:, d]
        np.add.at(counts, OFFSETS[d] + c[c >= 0], 1)
    return counts.tolist()


def build(simulation_path, routes, line_ids, code_of):
    """{"dimensions", "periods", "minPeople", "overall": counts per period (all the PT legs),
    "lines": {line_id: {"legs": legs per period, "line": counts per period, "sections": {"<a>><b>": [people, counts]}}}}
    for the transit lines line_ids; routes: the routes of the transit schedule (visualize.load_transit_schedule);
    code_of: stop id -> the stop code of the line comparison. None without the legs, trips and persons."""
    try:
        paths = [find_simulation_output(simulation_path, f)
                 for f in ("output_legs.csv.gz", "output_trips.csv.gz", "output_persons.csv.gz")]
    except FileNotFoundError:
        print("No output_legs/trips/persons.csv.gz in the simulation output: no passenger profiles")
        return None
    legs_path, trips_path, persons_path = paths

    print("Reading the public transport legs for the passenger profiles ...")
    legs = pd.read_csv(legs_path, sep = ";", dtype = str, usecols = [
        "person", "trip_id", "dep_time", "wait_time", "access_stop_id", "egress_stop_id", "transit_line", "transit_route"])
    legs = legs[legs["transit_route"].notna()].reset_index(drop = True)
    header = pd.read_csv(persons_path, sep = ";", nrows = 0).columns
    persons = pd.read_csv(persons_path, sep = ";", dtype = str, usecols = [c for c in PERSON_COLUMNS if c in header])
    persons = persons.drop_duplicates("person").set_index("person").reindex(legs["person"])
    trips = pd.read_csv(trips_path, sep = ";", dtype = str, usecols = ["trip_id", "end_activity_type"]).set_index("trip_id")
    codes = _codes(persons, trips["end_activity_type"].reindex(legs["trip_id"]).reset_index(drop = True))
    period = np.array([period_of(_seconds(d) + _seconds(w)) for d, w in zip(legs["dep_time"], legs["wait_time"].fillna("00:00:00"))])

    overall = [_count(codes[period == p]) for p in range(len(PERIODS))]

    stops_of_route = {route["route_id"]: [code_of(s) for s in route["stop_ids"]] for route in routes if route["line_id"] in line_ids}
    lines = {}
    on_lines = np.flatnonzero(legs["transit_line"].isin(line_ids).to_numpy())
    print(f"  {len(legs)} public transport legs, {len(on_lines)} on the compared lines")

    sections = {}  # (line, a, b) -> leg indices
    route_of, line_of = legs["transit_route"].to_numpy(), legs["transit_line"].to_numpy()
    access, egress = legs["access_stop_id"].to_numpy(), legs["egress_stop_id"].to_numpy()
    for i in on_lines:
        stops = stops_of_route.get(route_of[i])
        if stops is None:
            continue
        board, alight = code_of(access[i]), code_of(egress[i])
        k0 = stops.index(board) if board in stops else None
        k1 = next((k for k in range(k0 + 1, len(stops)) if stops[k] == alight), None) if k0 is not None else None
        if k1 is None:
            continue
        line = line_of[i]
        for a, b in zip(stops[k0:k1], stops[k0 + 1:k1 + 1]):
            if a != b:
                sections.setdefault((line, a, b), []).append(i)

    for line in line_ids:
        rows = on_lines[line_of[on_lines] == line]
        lines[line] = {
            "legs": [int((period[rows] == p).sum()) for p in range(len(PERIODS))],
            "line": [_count(codes[rows[period[rows] == p]]) for p in range(len(PERIODS))],
            "sections": {},
        }
    for (line, a, b), rows in sections.items():
        rows = np.array(rows)
        lines[line]["sections"][f"{a}>{b}"] = [len(rows), _count(codes[rows])]

    return {"dimensions": [[name, labels] for name, labels in DIMENSIONS], "periods": PERIODS, "minPeople": MIN_PEOPLE,
            "overall": overall, "lines": lines}
