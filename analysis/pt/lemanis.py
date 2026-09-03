"""
Leman Express (LEX) 2022 boarding/alighting counts, reshaped into an
hourly, TPG-shaped estimate so they can be treated as another "TPG-like"
data source. to_tpg_shape()'s output is what stages._build_lemanis_mofr
merges against MATSim and folds into the regular TPG comparison tables -
see that function's docstring for how the MATSim join works (direction-
blind, both LEX sens summed together, for the reasons in to_tpg_shape's
docstring).

The source CSV path is not hardcoded here - it comes from
config.Config.lemanis_csv_path (only required when include_lemanis is
true, see comparison_passenger_counts_geneva.py's analysis.pt.lemanis_csv_path).

SOURCE DATA: one row per (line, sens, stop, day-type, time-bin). day-types
are "Lundi - vendredi" (weekday), "Samedi - dimanche" and "Samedi" - per
instruction, only "Lundi - vendredi" rows are used here; weekends are
dropped entirely, not just skipped in some output.

WHAT THE RAW boardings/alightings VALUE MEANS - CHECKED, NOT AN HOURLY
RATE: the three weekday time bins are "6h-9h" (3h), "16h-19h" (3h) and
"9h-16h and 19h-6h" (a combined, 18-elapsed-hour off-peak bin). If the raw
values were hourly rates needing multiplication, GENEVE/L4/sens1's
off-peak value (1221) would imply an off-peak TOTAL of 1221*18 =~ 22k -
about 90x the evening peak total (244), which doesn't make sense for a
commuter line (peak periods carry the most riders, not the least). Read as
period TOTALS instead, the implied hourly rates line up across bins
instead (that same off-peak total / 18h =~ 68/h vs the evening peak's
244/3h =~ 81/h - comparable, as you'd expect). So this module treats the
raw values as already being period TOTALS, not hourly rates to multiply
out. If you have source documentation saying otherwise, flip
_PERIOD_IS_TOTAL below - everything downstream keys off it.

SPREADING EACH BIN'S TOTAL ACROSS HOURS: LEX has no service in the small
hours - checked directly against the L1-L6 (agency_id 87_LEX) trips in the
GTFS feed's stop_times.txt: scheduled stops start at 5h and stop after
23h, none in 0h-4h. So within the "9h-16h and 19h-6h" bin, the elapsed
19h-6h span (11h) is split into its actually-served hours (19h-23h, plus
5h) and the 4h of true closure (0h-4h), which get none of that bin's
total, rather than spreading it across all 18 nominal hours as if the
network ran all night - see _OPERATING_HOURS. Each bin's total is then
split EVENLY across its operating hours: the source has nothing finer than
these 3 coarse bins to weight an uneven split by.
"""

import numpy as np
import pandas as pd

# See module docstring. Flip to False only if you have documentation
# saying the source values are per-hour rates that should be multiplied by
# a bin's operating-hour count before being spread across hours.
_PERIOD_IS_TOTAL = True

# Hour-of-day each time bin actually has LEX service (see module
# docstring) - closed hours (0h-4h, inside the nominal "19h-6h" half of
# the off-peak bin) get none of that bin's total. A handful of
# (line, sens, stop) combos only have a single "All day" row instead of
# the usual 3 bins (e.g. L3/sens1/GENTHOD BELLEVUE) - spread across every
# operating hour of the day.
_OFF_PEAK_HOURS = [9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 5]
_OPERATING_HOURS = {
    "6h-9h": [6, 7, 8],
    "16h-19h": [16, 17, 18],
    "9h-16h and 19h-6h": _OFF_PEAK_HOURS,
    "All day": sorted({6, 7, 8, 16, 17, 18, *_OFF_PEAK_HOURS}),
}

# Station name (as given in the CSV) -> GTFS stop_id, matched against
# pt_passenger_counts.csv.gz ITSELF (not just stops.txt by name): the
# official GTFS feed has TWO separate id families for the same physical
# Geneva-area stations - an "858xxxx"/"859xxxx" family (used by TPG
# bus/tram stops elsewhere in this codebase) and an "8501xxx"/"8516xxx"/
# "8517xxx" family, and MATSim's LEX simulation turns out to call at the
# SECOND family, not the first. An earlier version of this dict used
# name-matched "858x"/"859x" ids (e.g. "Genève, gare Cornavin" for GENEVE)
# that looked right by name but silently produced all-zero MATSim
# comparisons for every Geneva-area stop - see git history/conversation
# for how that was caught (Annemasse's boardings_matsim being 0
# everywhere). Every id below for the Geneva-area cluster was confirmed
# present in pt_passenger_counts.csv.gz's own line_name=="L1".."L6" rows.
# The far-French ids (Annecy, Bonneville, ... Bellegarde) are still only
# name-matched against stops.txt, since MATSim's simulation area doesn't
# reach that far - it can't confirm or deny those (they're expected to
# show 0 boardings_matsim, which isn't a crosswalk error out there).
STOP_NAME_TO_GTFS_CODE = {
    "ANNEMASSE": "8774549",
    "CHENE BOURG": "8516274",
    "LES TUILERIES": "8501011",
    "VERNIER": "8501007",  # confirmed via MATSim - neither "gare/Croisette" nor "gare/Renfile" was right
    "CHAMBESY": "8501020",
    "COPPET": "8501023",
    "CREUX DE GENTHOD": "8501012",
    "GENEVE": "8501008",  # bare "Genève", not "Genève, gare Cornavin" (8587057) - that id is unused by MATSim's LEX rows
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
    # Not present in MATSim's simulated LEX network (outside its coverage
    # area) - name-matched against stops.txt only, unconfirmed:
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
    """Weekday-only ("Lundi - vendredi") rows, one per (line, sens, stop, time bin)."""

    df = pd.read_csv(path)
    df = df[df["day"] == "Lundi - vendredi"]

    return df.reset_index(drop = True)


def expand_to_hourly(weekday_df):
    """
    One row per (line, sens, stop, hour), built by splitting each time
    bin's total (see _PERIOD_IS_TOTAL) evenly across the hours it's
    actually in service for (see _OPERATING_HOURS).
    """

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
    """
    Reshapes expand_to_hourly's output into the same columns
    stages._build_tpg_mofr / comparison.py expect from a TPG stats file:
    day_type, gtfs_code, line_direction, hour, then boardings_mean /
    alightings_mean (+ their _raw_ duplicates, std, and quantiles). This
    is a single point estimate from one year of aggregate counts, not
    day-by-day raw data like TPG_processed_counts' tpg{year}_agg_workday
    files, so there's no real spread to report: std is 0 and every
    quantile equals the mean (same "duplicate the one number you have"
    approach tpg_raw_stats_2025.py uses for its raw/adjusted columns - see
    that module's docstring point 4).

    Both LEX directions (sens 1/2) are summed together here, so
    line_direction is just the bare line ("L1", not "L1_1"/"L1_2") - same
    convention as 2025's direction-less TPG stats (see
    tpg_raw_stats_2025.py's docstring point 3). This is deliberate, not a
    limitation of the source data (unlike 2025's): LEX's MATSim direction
    labels are free-text endpoints ("Coppet->Annemasse") with no clean way
    to line up against LEX's own numeric sens 1/2, and stages.py's
    integration is direction-blind on both sides for exactly that reason -
    see stages._build_lemanis_mofr.

    Only stops in STOP_NAME_TO_GTFS_CODE are kept; unmatched stop names
    are dropped with a printed count (see that dict's docstring for which
    ones and why).
    """

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
    """
    One route polyline per LEX line (L1-L6), both directions combined into
    a single line - same convention as to_tpg_shape (see its docstring for
    why: MATSim's LEX direction labels don't line up with anything usable
    here).

    Unlike tpg_data.build_line_route_geometries (which reads real stop
    *order* straight out of tpg_Lignes-arrêts_2024.csv), the Lemanis CSV's
    row order is alphabetical, not route order - checked directly, e.g.
    L1's stops list "ANNEMASSE, BONS EN CHABLAIS, CHAMBESY, ..." A-Z, not
    geographically. GTFS trip stop_times can't fill that gap either: L4
    and L5 (both real, high-ridership MATSim lines) have no route_id at
    all under route_short_name "L4"/"L5" in this feed, and reconstructing
    them from trip patterns for the other four lines turned up ambiguous/
    inconsistent GTFS route metadata (multiple candidate route_ids per
    line mixing full-length and rail-replacement-bus patterns). So instead
    each line's stop set (which stops it actually serves, per the CSV) is
    ordered geographically: every LEX stop's coordinates are projected
    onto the first principal axis of the whole LEX stop cloud (a
    reasonable stand-in for "the corridor's own direction", since LEX is
    close to a single roughly-straight Coppet/La Plaine - Geneve -
    Annemasse/Evian corridor) and sorted along it. Checked against the
    real route (Coppet -> ... -> Geneve -> Chene-Bourg -> Annemasse -> ...
    -> Evian/Thonon): this reproduces the correct order. Good enough to
    draw a recognizable line on a map; not a substitute for a real
    shapes.txt-derived polyline if you need exact curvature later.

    Returns a DataFrame: line, direction_letter (always None - see
    to_tpg_shape's docstring), coords (list of (lat, lon) tuples).
    """

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lemanis-csv-path", required = True, help = "Path to the Leman Express raw counts CSV")
    args = parser.parse_args()

    weekday_df = load_weekday_counts(args.lemanis_csv_path)
    hourly_df = expand_to_hourly(weekday_df)
    tpg_shaped = to_tpg_shape(hourly_df)

    print(tpg_shaped.shape)
    print(tpg_shaped.head())
