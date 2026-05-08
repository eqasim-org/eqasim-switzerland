"""Phase 2 loaders: persons, households, activities, trips.

Each loader writes one table and is a pure function of its inputs (no side
effects beyond the DuckDB connection). The loaders are tolerant of missing
optional inputs — they leave NULLs rather than refusing to run.

Coord system: all spatial columns are EPSG:2056 (LV95). Activities/trips
without coords are dropped (not inserted) and counted as warnings.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from .hilbert import CH_BBOX_LV95, hilbert_2d

log = logging.getLogger(__name__)


_CAR_AVAIL_MAP = {0.0: "always", 1.0: "sometimes", 2.0: "never"}

_PURPOSE_BUCKETS = {"home", "work", "education", "shop", "leisure"}

_MODE_BUCKETS = {"car", "pt", "walk", "bike", "car_passenger"}


def load_persons_synthetic(
    db: duckdb.DuckDBPyConnection,
    persons_parquet: Path,
    statpop_persons_pickle: Optional[Path],
    activities_for_home_pt: Optional[pd.DataFrame] = None,
    activities_for_n_acts: Optional[pd.DataFrame] = None,
) -> int:
    """Load synthetic persons. Returns row count."""
    df = pd.read_parquet(persons_parquet)
    df = df.rename(columns={"person_id": "person_id"}).copy()

    df["car_availability"] = df["car_availability"].map(_CAR_AVAIL_MAP)
    for col in ("has_driving_license", "employed",
                "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
                "subscriptions_strecke", "subscriptions_gleis7",
                "subscriptions_junior", "subscriptions_other"):
        if col in df.columns:
            df[col] = df[col].astype(bool)


    home_x = pd.Series(np.nan, index=df.index, dtype="float64")
    home_y = pd.Series(np.nan, index=df.index, dtype="float64")
    if activities_for_home_pt is not None and not activities_for_home_pt.empty:
        homes = (activities_for_home_pt[activities_for_home_pt["purpose"] == "home"]
                 .sort_values(["person_id", "activity_index"])
                 .drop_duplicates("person_id", keep="first")
                 .set_index("person_id")[["x", "y"]])
        idx = df.set_index("person_id").index
        home_x = pd.Series(homes["x"].reindex(idx).to_numpy(), index=df.index)
        home_y = pd.Series(homes["y"].reindex(idx).to_numpy(), index=df.index)

    needs_statpop = home_x.isna()
    if needs_statpop.any() and statpop_persons_pickle is not None:
        with open(statpop_persons_pickle, "rb") as f:
            sp = pickle.load(f)
        sp = sp[["person_id", "home_x", "home_y"]].rename(columns={"person_id": "statpop_person_id"})
        merged = df[["statpop_person_id"]].merge(sp, on="statpop_person_id", how="left")

        home_x = home_x.where(~needs_statpop, merged["home_x"].to_numpy())
        home_y = home_y.where(~needs_statpop, merged["home_y"].to_numpy())


    bad = (home_x < 2_000_000) | (home_y < 1_000_000)
    home_x = home_x.where(~bad, other=np.nan)
    home_y = home_y.where(~bad, other=np.nan)
    df["home_x"] = home_x
    df["home_y"] = home_y


    if activities_for_n_acts is not None and not activities_for_n_acts.empty:
        counts = activities_for_n_acts.groupby("person_id").size().rename("n_activities")
        df = df.merge(counts, left_on="person_id", right_index=True, how="left")
    else:
        df["n_activities"] = pd.NA
    df["n_activities"] = df["n_activities"].astype("Int32")


    has_xy = df["home_x"].notna() & df["home_y"].notna()
    hidx = np.zeros(len(df), dtype=np.uint64)
    if has_xy.any():
        hidx[has_xy.to_numpy()] = hilbert_2d(
            df.loc[has_xy, "home_x"].to_numpy(),
            df.loc[has_xy, "home_y"].to_numpy(),
            CH_BBOX_LV95,
        )
    df["hilbert_idx"] = hidx
    df = df.sort_values("hilbert_idx", kind="mergesort").reset_index(drop=True)

    _insert_persons(db, df)
    log.info("Inserted %d persons (with home_pt: %d)", len(df), int(has_xy.sum()))
    return len(df)


def load_persons_microcensus(
    db: duckdb.DuckDBPyConnection,
    household_persons_pickle: Path,
    households_pickle: Optional[Path] = None,
) -> int:
    """Microcensus persons: full 163k roster from household_persons[0],
    enriched with home_x/home_y/canton_id by joining the households cache.

    Schema-contract fixes vs. the v1.0 first-iteration build:
      • ``sex`` is normalised from MZ's {1=male, 2=female, -98=unknown}
        encoding to the schema-contract {0=male, 1=female, NULL=unknown}.
      • ``canton_id`` is taken straight from the households cache (already
        pre-computed by the upstream MZ-spatial stage) — no spatial join here.
      • ``home_x`` / ``home_y`` (LV95) are looked up via household_id, so all
        163k persons get a ``home_pt`` rather than 0.
      • ``hilbert_idx`` is computed from the home point (matches synthetic
        ordering, so demo_grid joins behave identically).
    """
    with open(household_persons_pickle, "rb") as f:
        triple = pickle.load(f)
    df = triple[0].copy()


    df["household_id"] = df["household_id"].astype("int64")
    df["person_id"] = df["household_id"] * 100 + df["hhpers_id"].astype("int64")


    sex_raw = pd.to_numeric(df["sex"], errors="coerce").astype("Int64")
    sex_norm = pd.Series(pd.NA, index=df.index, dtype="Int32")
    sex_norm[sex_raw == 1] = 0
    sex_norm[sex_raw == 2] = 1

    df["sex"] = sex_norm
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int32")


    home_x = pd.Series(np.nan, index=df.index, dtype="float64")
    home_y = pd.Series(np.nan, index=df.index, dtype="float64")
    canton = pd.Series(pd.NA, index=df.index, dtype="Int32")
    if households_pickle is not None and households_pickle.exists():
        with open(households_pickle, "rb") as f:
            hh = pickle.load(f)


        hh = (hh[["person_id", "home_x", "home_y", "canton_id"]]
              .rename(columns={"person_id": "household_id"})
              .copy())
        hh["household_id"] = hh["household_id"].astype("int64")
        merged = df[["household_id"]].merge(hh, on="household_id", how="left")
        home_x = merged["home_x"].astype("float64")
        home_y = merged["home_y"].astype("float64")
        canton = pd.to_numeric(merged["canton_id"], errors="coerce").astype("Int32")
    else:
        log.warning("microcensus households cache missing — persons will have "
                    "NULL home_pt and NULL canton_id")
    df["home_x"] = home_x.to_numpy()
    df["home_y"] = home_y.to_numpy()
    df["canton_id"] = canton.to_numpy()


    bad = (df["home_x"] < 2_000_000) | (df["home_y"] < 1_000_000) | df["home_x"].isna() | df["home_y"].isna()
    df.loc[bad, "home_x"] = np.nan
    df.loc[bad, "home_y"] = np.nan


    for col in ("car_availability", "has_driving_license", "employed",
                "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
                "subscriptions_strecke", "subscriptions_gleis7",
                "subscriptions_junior", "subscriptions_other",
                "n_activities"):
        if col not in df.columns:
            df[col] = pd.NA

    has_xy = df["home_x"].notna() & df["home_y"].notna()
    hidx = np.zeros(len(df), dtype=np.uint64)
    if has_xy.any():
        hidx[has_xy.to_numpy()] = hilbert_2d(
            df.loc[has_xy, "home_x"].to_numpy(),
            df.loc[has_xy, "home_y"].to_numpy(),
            CH_BBOX_LV95,
        )
    df["hilbert_idx"] = hidx
    df = df.sort_values("hilbert_idx", kind="mergesort").reset_index(drop=True)

    _insert_persons(db, df)
    log.info("Inserted %d microcensus persons (with home_pt: %d, with canton_id: %d)",
             len(df), int(has_xy.sum()), int(df["canton_id"].notna().sum()))
    return len(df)


def _insert_persons(db: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Insert into the `persons` table. df may carry extra cols — they're ignored."""
    cols = ["person_id", "household_id", "age", "sex", "car_availability",
            "has_driving_license", "employed",
            "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
            "subscriptions_strecke", "subscriptions_gleis7",
            "subscriptions_junior", "subscriptions_other",
            "canton_id", "n_activities", "hilbert_idx"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    payload = df[cols + ["home_x", "home_y"]].copy()
    db.register("_tmp_persons", payload)
    db.execute(f"""
        INSERT INTO persons ({", ".join(cols + ["home_pt"])})
        SELECT {", ".join(cols)},
               CASE WHEN home_x IS NULL OR home_y IS NULL THEN NULL
                    ELSE ST_Point(home_x, home_y) END AS home_pt
        FROM _tmp_persons
    """)
    db.unregister("_tmp_persons")


def load_households_synthetic(
    db: duckdb.DuckDBPyConnection,
    enriched_pickle: Optional[Path],
    households_pickle: Optional[Path] = None,
) -> int:
    """Insert one row per household with attribute fields.

    Source of truth: ``synthesis.population.enriched`` cache, which has the
    per-person income/cars/bikes/ovgk columns (constant within a household).
    The legacy ``data.statpop.households`` cache is just a household_id list
    and is kept here only as a last-resort fallback (rows go in with NULL
    attributes, matching old behaviour).
    """
    if enriched_pickle is not None and enriched_pickle.exists():
        with open(enriched_pickle, "rb") as f:
            enriched = pickle.load(f)


        df = (enriched[["household_id", "income_class",
                        "number_of_cars_class", "number_of_bikes_class", "ovgk"]]
              .drop_duplicates("household_id")
              .copy())
        df = df.rename(columns={
            "number_of_cars_class": "n_cars_class",
            "number_of_bikes_class": "n_bikes_class",
        })
        df["household_id"] = df["household_id"].astype("int64")


        for col in ("income_class", "n_cars_class", "n_bikes_class", "ovgk"):
            df[col] = df[col].astype("string")
        db.register("_tmp_hh", df[["household_id", "income_class",
                                    "n_cars_class", "n_bikes_class", "ovgk"]])
        db.execute("""
            INSERT INTO households (household_id, income_class, n_cars_class, n_bikes_class, ovgk)
            SELECT household_id, income_class, n_cars_class, n_bikes_class, ovgk FROM _tmp_hh
        """)
        db.unregister("_tmp_hh")
        log.info("Inserted %d synthetic households (with attributes from enriched)", len(df))
        return len(df)

    if households_pickle is None or not households_pickle.exists():
        log.warning("synthetic households: no enriched cache and no households cache — table left empty")
        return 0
    log.warning("synthetic households: enriched cache missing — falling back to %s "
                "(household_id only, attributes will be NULL)", households_pickle)
    with open(households_pickle, "rb") as f:
        df = pickle.load(f)
    if "household_id" not in df.columns:
        return 0
    df = df.copy()
    for col in ("income_class", "n_cars_class", "n_bikes_class", "ovgk"):
        df[col] = pd.NA
    df["household_id"] = df["household_id"].astype("int64")
    db.register("_tmp_hh", df[["household_id", "income_class", "n_cars_class", "n_bikes_class", "ovgk"]])
    db.execute("""
        INSERT INTO households (household_id, income_class, n_cars_class, n_bikes_class, ovgk)
        SELECT household_id, income_class, n_cars_class, n_bikes_class, ovgk FROM _tmp_hh
    """)
    db.unregister("_tmp_hh")
    return len(df)


def load_households_microcensus(
    db: duckdb.DuckDBPyConnection,
    households_pickle: Path,
) -> int:
    """Insert microcensus households from the per-household cache.

    The cache (``data.microcensus.households``) has one row per household.
    Its key column is misleadingly called ``person_id`` but in this dataset
    that ID == household_id (the household's reference person). It carries
    income_class / number_of_cars_class / number_of_bikes_class / ovgk
    directly, so we just rename and insert.
    """
    with open(households_pickle, "rb") as f:
        df = pickle.load(f)
    if "person_id" not in df.columns:
        log.warning("microcensus households cache has no person_id key — skipping")
        return 0
    df = df.copy()
    df = df.rename(columns={
        "person_id": "household_id",
        "number_of_cars_class": "n_cars_class",
        "number_of_bikes_class": "n_bikes_class",
    })
    df["household_id"] = df["household_id"].astype("int64")
    for col in ("income_class", "n_cars_class", "n_bikes_class", "ovgk"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype("string")
    db.register("_tmp_hh_mc", df[["household_id", "income_class",
                                    "n_cars_class", "n_bikes_class", "ovgk"]])
    db.execute("""
        INSERT INTO households (household_id, income_class, n_cars_class, n_bikes_class, ovgk)
        SELECT household_id, income_class, n_cars_class, n_bikes_class, ovgk FROM _tmp_hh_mc
    """)
    db.unregister("_tmp_hh_mc")
    log.info("Inserted %d microcensus households", len(df))
    return len(df)


ACTIVITY_BATCH = 500_000


def parse_activities_csv(path: Path) -> pd.DataFrame:
    """Read eqasim output_activities.csv[.gz] into a DataFrame.

    Columns produced by eqasim ActivityWriter (verified against the Java source):
      person_id, activity_index, purpose, start_time, end_time, x, y,
      facility_id, link_id  (delimiter ;)

    eqasim emits literal ``-Infinity`` / ``Infinity`` for first/last activity
    boundaries — convert those to NaN so downstream SQL sees NULL. Freight
    person IDs (``freight_*``) are non-numeric and dropped — they don't count
    as travellers in webmap analyses.
    """
    df = pd.read_csv(path, sep=";", dtype={"person_id": str, "activity_index": "int32"},
                     na_values=["", "NA", "NaN", "-Infinity", "Infinity"])
    df = _filter_numeric_person_id(df, "activities")
    for tcol in ("start_time", "end_time"):
        if tcol in df.columns:
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
            df.loc[~np.isfinite(df[tcol]), tcol] = np.nan
    df["purpose"] = df["purpose"].apply(_canonical_purpose)
    return df


def load_activities(
    db: duckdb.DuckDBPyConnection,
    activities_csv: Optional[Path],
) -> int:
    """Wrapper: parse + insert. Used when the caller has no DataFrame."""
    if activities_csv is None or not activities_csv.exists():
        log.warning("activities CSV not found — table left empty")
        return 0
    return insert_activities_df(db, parse_activities_csv(activities_csv))


def insert_activities_df(db: duckdb.DuckDBPyConnection, df: Optional[pd.DataFrame]) -> int:
    """Insert an already-parsed activities DataFrame."""
    if df is None or df.empty:
        return 0
    df = df.copy()

    grp = df.groupby("person_id", sort=False)["activity_index"]
    df["is_first"] = (df["activity_index"] == grp.transform("min")).astype(bool)
    df["is_last"] = (df["activity_index"] == grp.transform("max")).astype(bool)

    df.loc[df["is_first"], "start_time"] = pd.NA
    df.loc[df["is_last"], "end_time"] = pd.NA

    valid = df["x"].notna() & df["y"].notna()
    dropped = (~valid).sum()
    if dropped:
        log.warning("Dropping %d activities with missing coords", dropped)
    df = df[valid].copy()

    n_total = 0
    for start in range(0, len(df), ACTIVITY_BATCH):
        chunk = df.iloc[start:start + ACTIVITY_BATCH]
        db.register("_tmp_acts", chunk[["person_id", "activity_index", "purpose",
                                        "start_time", "end_time", "is_first", "is_last",
                                        "x", "y"]])
        db.execute("""
            INSERT INTO activities
                (person_id, activity_index, purpose, start_time, end_time,
                 is_first, is_last, location_pt)
            SELECT person_id, activity_index, purpose, start_time, end_time,
                   is_first, is_last, ST_Point(x, y) FROM _tmp_acts
        """)
        db.unregister("_tmp_acts")
        n_total += len(chunk)
    log.info("Inserted %d activities", n_total)
    return n_total


TRIP_BATCH = 500_000


def parse_trips_csv(path: Path) -> pd.DataFrame:
    """eqasim output_trips.csv[.gz] schema (verified against TripWriter.java).

    Drops freight rows (non-numeric person_id) — same rationale as activities.
    """
    df = pd.read_csv(path, sep=";",
                     dtype={"person_id": str, "person_trip_id": "int32"},
                     na_values=["", "NA", "NaN"])
    df = _filter_numeric_person_id(df, "trips")
    df = df.rename(columns={
        "person_trip_id": "trip_index",
        "mode": "main_mode",
        "routed_distance": "network_distance",
        "euclidean_distance": "crowfly_distance",
        "origin_x": "origin_x", "origin_y": "origin_y",
        "destination_x": "dest_x", "destination_y": "dest_y",
    })
    df["main_mode"] = df["main_mode"].apply(_canonical_mode)
    df["preceding_purpose"] = df["preceding_purpose"].apply(_canonical_purpose)
    df["following_purpose"] = df["following_purpose"].apply(_canonical_purpose)
    return df


def load_trips_synthetic(
    db: duckdb.DuckDBPyConnection,
    trips_csv: Optional[Path],
) -> int:
    if trips_csv is None or not trips_csv.exists():
        log.warning("synthetic trips CSV not found — table left empty")
        return 0
    df = parse_trips_csv(trips_csv)
    return _insert_trips(db, df)


def load_trips_microcensus(
    db: duckdb.DuckDBPyConnection,
    trips_pickle: Path,
) -> int:
    """Load microcensus trips. Returns row count.

    The trips cache keys each trip on the diary respondent's
    ``person_id`` — which in MZ equals their ``household_id`` (the
    respondent is treated as a single representative per household).
    To make trips joinable against ``persons`` (where each row carries
    the synthetic key ``household_id * 100 + hhpers_id``) we map the
    diary respondent to ``household_id * 100 + 1``, i.e. we assume the
    diary respondent is always hhpers_id=1 (head of household).
    Documented assumption — if upstream changes that convention, fix
    here AND in :func:`derive_activities_microcensus` together.
    """
    with open(trips_pickle, "rb") as f:
        tup = pickle.load(f)
    df = tup[0].copy() if isinstance(tup, tuple) else tup.copy()
    df = df.rename(columns={
        "trip_id": "trip_index",
        "mode": "main_mode",
        "destination_x": "dest_x",
        "destination_y": "dest_y",
        "purpose": "following_purpose",
    })
    df["preceding_purpose"] = pd.NA
    df["travel_time"] = (df.get("arrival_time") - df.get("departure_time")) if "arrival_time" in df.columns else pd.NA
    df["main_mode"] = df["main_mode"].apply(_canonical_mode)
    df["following_purpose"] = df["following_purpose"].apply(_canonical_purpose)

    df["person_id"] = df["person_id"].astype("int64") * 100 + 1
    df["trip_index"] = df["trip_index"].astype("int32")
    return _insert_trips(db, df)


def derive_activities_microcensus(
    db: duckdb.DuckDBPyConnection,
    trips_pickle: Path,
) -> int:
    """Reconstruct an activity sequence per diary respondent from MZ trips.

    MZ provides only trips, but each trip carries `purpose` (= the purpose
    *at the destination*, i.e. the start of the next activity), plus
    `departure_time`, `arrival_time` and `origin`/`destination` coords. So
    for every respondent with N trips we materialise N+1 activities:

      Activity 0 (home start):  location = trip[0].origin,
                                purpose  = "home",
                                start_time = NULL, end_time = trip[0].departure_time
      Activity i (1 ≤ i < N):   location = trip[i-1].destination,
                                purpose  = trip[i-1].purpose,
                                start_time = trip[i-1].arrival_time,
                                end_time   = trip[i].departure_time
      Activity N (terminal):    location = trip[N-1].destination,
                                purpose  = trip[N-1].purpose,
                                start_time = trip[N-1].arrival_time,
                                end_time   = NULL

    person_id is synthesised the same way as in :func:`load_trips_microcensus`
    (raw_pid * 100 + 1) so activities can be joined back to persons.
    """
    with open(trips_pickle, "rb") as f:
        tup = pickle.load(f)
    trips = tup[0].copy() if isinstance(tup, tuple) else tup.copy()
    if trips.empty:
        log.warning("microcensus trips empty — no activities derived")
        return 0

    trips["person_id"] = trips["person_id"].astype("int64") * 100 + 1
    trips["trip_id"] = trips["trip_id"].astype("int32")
    trips = trips.sort_values(["person_id", "departure_time", "trip_id"], kind="mergesort").reset_index(drop=True)


    trips["trip_ord"] = trips.groupby("person_id", sort=False).cumcount()


    next_dep = (trips.groupby("person_id", sort=False)["departure_time"]
                       .shift(-1).rename("next_departure_time"))
    arrival_acts = pd.DataFrame({
        "person_id":      trips["person_id"].values,
        "activity_index": (trips["trip_ord"] + 1).astype("int32").values,
        "purpose":        trips["purpose"].apply(_canonical_purpose).values,
        "start_time":     trips["arrival_time"].values,
        "end_time":       next_dep.values,
        "x":              trips["destination_x"].values,
        "y":              trips["destination_y"].values,
    })


    first_trip = trips.drop_duplicates("person_id", keep="first")
    home_acts = pd.DataFrame({
        "person_id":      first_trip["person_id"].values,
        "activity_index": np.zeros(len(first_trip), dtype=np.int32),
        "purpose":        ["home"] * len(first_trip),
        "start_time":     [pd.NA] * len(first_trip),
        "end_time":       first_trip["departure_time"].values,
        "x":              first_trip["origin_x"].values,
        "y":              first_trip["origin_y"].values,
    })

    acts = pd.concat([home_acts, arrival_acts], ignore_index=True)
    acts = acts.sort_values(["person_id", "activity_index"], kind="mergesort").reset_index(drop=True)
    log.info("Derived %d microcensus activities from %d trips (%d respondents)",
             len(acts), len(trips), trips["person_id"].nunique())

    return insert_activities_df(db, acts)


def _insert_trips(db: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    xmin, ymin, xmax, ymax = CH_BBOX_LV95
    has_coords = (df["origin_x"].notna() & df["origin_y"].notna()
                  & df["dest_x"].notna() & df["dest_y"].notna())
    in_bbox = (df["origin_x"].between(xmin, xmax) & df["origin_y"].between(ymin, ymax)
               & df["dest_x"].between(xmin, xmax) & df["dest_y"].between(ymin, ymax))
    valid = has_coords & in_bbox
    dropped = (~valid).sum()
    if dropped:
        log.warning("Dropping %d trips (missing coords or outside CH bbox)", dropped)
    df = df[valid].copy()
    if df.empty:
        return 0
    for col in ("travel_time", "network_distance", "crowfly_distance",
                "preceding_purpose", "following_purpose"):
        if col not in df.columns:
            df[col] = pd.NA

    df["hilbert_origin"] = hilbert_2d(df["origin_x"].to_numpy(),
                                      df["origin_y"].to_numpy(),
                                      CH_BBOX_LV95)
    df = df.sort_values("hilbert_origin", kind="mergesort").reset_index(drop=True)

    n_total = 0
    cols_payload = ["person_id", "trip_index", "departure_time", "travel_time",
                    "main_mode", "preceding_purpose", "following_purpose",
                    "network_distance", "crowfly_distance",
                    "origin_x", "origin_y", "dest_x", "dest_y", "hilbert_origin"]
    for start in range(0, len(df), TRIP_BATCH):
        chunk = df.iloc[start:start + TRIP_BATCH][cols_payload]
        db.register("_tmp_trips", chunk)
        db.execute("""
            INSERT INTO trips
                (person_id, trip_index, departure_time, travel_time,
                 main_mode, preceding_purpose, following_purpose,
                 network_distance, crowfly_distance,
                 origin_pt, dest_pt, hilbert_origin)
            SELECT person_id, trip_index, departure_time, travel_time,
                   main_mode, preceding_purpose, following_purpose,
                   network_distance, crowfly_distance,
                   ST_Point(origin_x, origin_y),
                   ST_Point(dest_x, dest_y),
                   hilbert_origin FROM _tmp_trips
        """)
        db.unregister("_tmp_trips")
        n_total += len(chunk)
    log.info("Inserted %d trips", n_total)
    return n_total


def _filter_numeric_person_id(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Coerce person_id to int64. Drop rows with non-numeric (freight) IDs."""
    coerced = pd.to_numeric(df["person_id"], errors="coerce")
    keep = coerced.notna()
    n_dropped = (~keep).sum()
    if n_dropped:
        log.warning("Dropping %d %s rows with non-numeric person_id (e.g. freight)", n_dropped, label)
    df = df.loc[keep].copy()
    df["person_id"] = coerced.loc[keep].astype("int64")
    return df


def _canonical_purpose(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()

    s = s.split("_")[0] if "_" in s else s
    return s if s in _PURPOSE_BUCKETS else "other"


def _canonical_mode(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    return s if s in _MODE_BUCKETS else "walk"
