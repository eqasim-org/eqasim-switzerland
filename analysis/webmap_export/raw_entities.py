"""Phase 2 loaders: persons, households, activities, trips.
All spatial columns are EPSG:2056 (LV95); rows without coords are dropped.
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


# Legacy synthetic persons schema (the former webmap_data/synthetic/
# switzerland_persons.parquet). Both synthetic.duckdb and the download bundle's
# persons.parquet keep this layout; build_synthetic_persons_frame produces it.
LEGACY_PERSON_COLUMNS = [
    "person_id", "household_id", "car_availability", "age", "employed", "sex",
    "has_driving_license",
    "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
    "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior",
    "subscriptions_other",
    "subscriptions_ga_class", "subscriptions_verbund_class", "subscriptions_strecke_class",
    "statpop_person_id", "mz_person_id", "mz_head_id", "canton_id",
]

# synthesis.population.models.subscriptions draws one category per person:
# 0 none, 1 GA (incl. junior GA < 16), 2 Verbund (incl. Strecke), 3 Halbtax,
# 4 Verbund + Halbtax.
_PT_SUBSCRIPTION_FLAGS = {
    "subscriptions_ga": (1,),
    "subscriptions_halbtax": (3, 4),
    "subscriptions_verbund": (2, 4),
}
# No longer modelled separately - always NULL.
_UNMODELLED_SUBSCRIPTIONS = (
    "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior",
    "subscriptions_other",
    "subscriptions_ga_class", "subscriptions_verbund_class", "subscriptions_strecke_class",
)

_EMPLOYED = 1  # data.constants: 1 employed, 2 unemployed, 3 inactive


def build_synthetic_persons_frame(persons_csv: Path, enriched_pickle: Optional[Path]) -> pd.DataFrame:
    """synthesis.output's switzerland_persons.csv mapped onto LEGACY_PERSON_COLUMNS.

    car_availability and statpop_person_id are not exported by synthesis.output
    and come from the synthesis.population.enriched cache (same person_ids).
    """
    df = pd.read_csv(persons_csv, sep=";")
    log.info("Reading %d synthetic persons from %s", len(df), persons_csv)

    out = pd.DataFrame({
        "person_id": df["person_id"].astype("int64"),
        "household_id": df["household_id"].astype("int64"),
        "age": df["age"].astype("int64"),
        "employed": df["employed"] == _EMPLOYED,
        "sex": df["sex"].astype("int64"),
        "has_driving_license": df["has_driving_license"].astype(bool),
        "mz_person_id": df["mz_person_id"].astype("int64"),
        "mz_head_id": pd.array([pd.NA] * len(df), dtype="Int64"),
        "canton_id": df["canton_id"].astype("int64"),
    })

    # Unmatched persons (mz_person_id == -1, the age 0-5 children) have no survey
    # subscriptions; keep them NULL rather than the model's default "none".
    unmatched = (df["mz_person_id"] == -1).to_numpy()
    sub = df["pt_subscription"].astype("int64")
    for col, codes in _PT_SUBSCRIPTION_FLAGS.items():
        flag = sub.isin(codes).astype(object)
        flag[unmatched] = None
        out[col] = flag
    for col in _UNMODELLED_SUBSCRIPTIONS:
        out[col] = pd.Series([None] * len(df), dtype=object)

    out["car_availability"] = np.nan
    out["statpop_person_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    if enriched_pickle is not None and enriched_pickle.exists():
        with open(enriched_pickle, "rb") as f:
            enriched = pickle.load(f)
        enriched = enriched.drop_duplicates("person_id").set_index("person_id")
        idx = out["person_id"]
        # car_availability is binary now (1 available, 0 not); legacy float
        # codes are 0 always / 1 sometimes / 2 never.
        car = pd.to_numeric(enriched["car_availability"], errors="coerce").reindex(idx)
        out["car_availability"] = car.map({1: 0.0, 0: 2.0}).to_numpy()
        sp_ids = pd.array(enriched["statpop_person_id"].reindex(idx).to_numpy(), dtype="Int64")
        out["statpop_person_id"] = sp_ids.astype("int64") if not sp_ids.isna().any() else sp_ids
        n_miss = int(out["car_availability"].isna().sum())
        if n_miss:
            log.warning("persons: %d not in enriched cache %s - car_availability NULL",
                        n_miss, enriched_pickle)
    else:
        log.warning("persons: enriched cache missing - car_availability/statpop_person_id NULL")

    return out[LEGACY_PERSON_COLUMNS]


def simulated_person_ids(output_persons_csv: Optional[Path]) -> Optional[pd.Index]:
    """Numeric agent ids of the run (output_persons.csv[.gz]); None if unavailable."""
    if output_persons_csv is None or not output_persons_csv.exists():
        return None
    ids = pd.read_csv(output_persons_csv, sep=";", usecols=["person"], dtype=str)["person"]
    return pd.Index(pd.to_numeric(ids, errors="coerce").dropna().astype("int64").unique())


def check_persons_match_run(persons: pd.DataFrame, run_ids: Optional[pd.Index],
                            min_share: float = 0.95) -> None:
    """Fail when the persons file belongs to another population than the run.

    A stale persons file once shared only ~2% of its ids with the simulated
    agents and still produced a plausible-looking webmap. Checked persons ->
    agents: the run also has cross-border agents that are not in the file."""
    if run_ids is None:
        log.warning("persons: no output_persons for the run - population match not checked")
        return
    share = float(persons["person_id"].isin(run_ids).mean())
    log.info("persons: %.1f%% of persons file found among simulated agents", 100 * share)
    if share < min_share:
        raise ValueError(
            f"only {share:.1%} of the persons file are simulated agents - it is "
            "from a different synthesis run than the MATSim output")


def load_persons_synthetic(
    db: duckdb.DuckDBPyConnection,
    persons: pd.DataFrame,
    statpop_persons_pickle: Optional[Path],
    activities_for_home_pt: Optional[pd.DataFrame] = None,
    activities_for_n_acts: Optional[pd.DataFrame] = None,
) -> int:
    """Load synthetic persons (build_synthetic_persons_frame output). Returns row count."""
    df = persons.copy()

    df["car_availability"] = df["car_availability"].map(_CAR_AVAIL_MAP)
    for col in ("has_driving_license", "employed",
                "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
                "subscriptions_strecke", "subscriptions_gleis7",
                "subscriptions_junior", "subscriptions_other"):
        if col in df.columns:
            # nullable dtype: persons unmatched to the microcensus (age 0-5) carry
            # None here, and plain astype(bool) would silently record them as False
            df[col] = df[col].astype("boolean")

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
    respondents_pickle: Optional[Path] = None,
) -> int:
    """Load microcensus persons (full roster, enriched from the households cache)."""
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
        log.warning("microcensus households cache missing - persons will have "
                    "NULL home_pt and NULL canton_id")
    df["home_x"] = home_x.to_numpy()
    df["home_y"] = home_y.to_numpy()
    df["canton_id"] = canton.to_numpy()

    bad = (df["home_x"] < 2_000_000) | (df["home_y"] < 1_000_000) | df["home_x"].isna() | df["home_y"].isna()
    df.loc[bad, "home_x"] = np.nan
    df.loc[bad, "home_y"] = np.nan

    # Survey attributes exist only for diary respondents (hhpers_id=1, so
    # person_id = respondent_id*100+1); other household members stay NULL.
    if respondents_pickle is not None and respondents_pickle.exists():
        with open(respondents_pickle, "rb") as f:
            resp = pickle.load(f).copy()
        resp["person_id"] = resp["person_id"].astype("int64") * 100 + 1
        bool_cols = ["driving_license", "employed",
                     "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
                     "subscriptions_strecke", "subscriptions_gleis7",
                     "subscriptions_junior", "subscriptions_other"]
        keep = (["person_id", "car_availability"]
                + [c for c in bool_cols if c in resp.columns]
                + (["person_weight"] if "person_weight" in resp.columns else []))
        resp = resp[keep].rename(columns={"driving_license": "has_driving_license"})
        resp["car_availability"] = (
            pd.to_numeric(resp["car_availability"], errors="coerce").map(_CAR_AVAIL_MAP)
        )
        df = df.merge(resp, on="person_id", how="left")
        n_enriched = int(df["car_availability"].notna().sum())
        if n_enriched == 0:
            log.error("microcensus: respondents pickle %s found but 0 of %d rows "
                      "joined on person_id = raw_pid*100+1 - id convention broken? "
                      "(sample respondent ids: %s, sample roster ids: %s)",
                      respondents_pickle, len(resp),
                      resp["person_id"].head(3).tolist(),
                      df["person_id"].head(3).tolist())
        else:
            log.info("microcensus: enriched %d respondents with survey attributes "
                     "(car_availability/licence/employed/subscriptions) from %s",
                     n_enriched, respondents_pickle)
    else:
        log.error("microcensus respondents cache missing (looked for %s) - "
                  "car_availability/subscriptions/employed stay NULL and the webmap "
                  "loses its car-availability and PT-subscription panels",
                  respondents_pickle)

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


def backfill_n_activities(db: duckdb.DuckDBPyConnection) -> int:
    """Fill persons.n_activities from the activities table (microcensus)."""
    db.execute("""
        UPDATE persons SET n_activities = a.cnt
        FROM (SELECT person_id, COUNT(*)::INTEGER AS cnt FROM activities GROUP BY person_id) a
        WHERE persons.person_id = a.person_id
    """)
    n = db.execute("SELECT COUNT(*) FROM persons WHERE n_activities IS NOT NULL").fetchone()[0]
    log.info("backfilled n_activities for %d persons", n)
    return n


def backfill_preceding_purpose(db: duckdb.DuckDBPyConnection) -> int:
    """Fill trips.preceding_purpose from the preceding activity's purpose (microcensus)."""
    db.execute("""
        UPDATE trips SET preceding_purpose = a.purpose
        FROM activities a
        WHERE a.person_id = trips.person_id
          AND a.activity_index = trips.trip_index - 1
    """)
    n = db.execute("SELECT COUNT(*) FROM trips WHERE preceding_purpose IS NOT NULL").fetchone()[0]
    log.info("backfilled preceding_purpose for %d trips", n)
    return n


def _insert_persons(db: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Insert into the persons table; extra df columns are ignored."""
    from .hex import (
        H3_RES_COARSE, H3_RES_FINE, H3_RES_MID,
        hex_for_xy_lv95, hex_parent_int,
    )

    cols = ["person_id", "household_id", "age", "sex", "person_weight",
            "car_availability", "has_driving_license", "employed",
            "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
            "subscriptions_strecke", "subscriptions_gleis7",
            "subscriptions_junior", "subscriptions_other",
            "canton_id", "n_activities", "hilbert_idx"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    has_xy = df["home_x"].notna() & df["home_y"].notna()
    h12 = np.full(len(df), -1, dtype="int64")
    h9 = np.full(len(df), -1, dtype="int64")
    h6 = np.full(len(df), -1, dtype="int64")
    if has_xy.any():
        idx = has_xy.to_numpy()
        h12_subset = hex_for_xy_lv95(
            df.loc[has_xy, "home_x"].to_numpy(),
            df.loc[has_xy, "home_y"].to_numpy(),
            H3_RES_FINE,
        )
        h12[idx] = h12_subset
        h9[idx] = hex_parent_int(h12_subset, H3_RES_MID)
        h6[idx] = hex_parent_int(h12_subset, H3_RES_COARSE)
    df = df.copy()
    df["home_h3_res12"] = h12
    df["home_h3_res9"] = h9
    df["home_h3_res6"] = h6

    h3_cols = [f"home_h3_res{H3_RES_FINE}",
               f"home_h3_res{H3_RES_MID}",
               f"home_h3_res{H3_RES_COARSE}"]
    payload = df[cols + ["home_x", "home_y"] + h3_cols].copy()
    db.register("_tmp_persons", payload)
    db.execute(f"""
        INSERT INTO persons ({", ".join(cols + ["home_pt"] + h3_cols)})
        SELECT {", ".join(cols)},
               CASE WHEN home_x IS NULL OR home_y IS NULL THEN NULL
                    ELSE ST_Point(home_x, home_y) END AS home_pt,
               CASE WHEN home_h3_res{H3_RES_FINE} < 0 THEN NULL ELSE home_h3_res{H3_RES_FINE} END,
               CASE WHEN home_h3_res{H3_RES_MID} < 0 THEN NULL ELSE home_h3_res{H3_RES_MID} END,
               CASE WHEN home_h3_res{H3_RES_COARSE} < 0 THEN NULL ELSE home_h3_res{H3_RES_COARSE} END
        FROM _tmp_persons
    """)
    db.unregister("_tmp_persons")


def build_synthetic_households_frame(enriched_pickle: Path) -> pd.DataFrame:
    """One row per household: household_id, income_class, n_cars_class, n_bikes_class, ovgk."""
    with open(enriched_pickle, "rb") as f:
        enriched = pickle.load(f)

    # statpop enrichment no longer attaches household bike classes, only
    # person-level bike_availability (0/1). Derive the class as the number of
    # members with bike access, capped like the MZ class (MAX_NUMBER_OF_BIKES_CLASS).
    if "number_of_bikes_class" not in enriched.columns and "bike_availability" in enriched.columns:
        n_bikes = (pd.to_numeric(enriched["bike_availability"], errors="coerce")
                   .fillna(0).groupby(enriched["household_id"]).transform("sum"))
        enriched = enriched.assign(number_of_bikes_class=n_bikes.clip(upper=3).astype("int64"))
        log.info("synthetic households: n_bikes_class derived from person bike_availability")

    # Tolerate any remaining missing attribute columns (left NULL).
    attr_cols = ["income_class", "number_of_cars_class", "number_of_bikes_class", "ovgk"]
    missing = [c for c in attr_cols if c not in enriched.columns]
    if missing:
        log.warning("synthetic households: enriched cache lacks %s - left NULL", missing)
    df = (enriched[["household_id"] + [c for c in attr_cols if c not in missing]]
          .drop_duplicates("household_id")
          .copy())
    for col in missing:
        df[col] = pd.NA
    df = df.rename(columns={
        "number_of_cars_class": "n_cars_class",
        "number_of_bikes_class": "n_bikes_class",
    })
    df["household_id"] = df["household_id"].astype("int64")
    for col in ("income_class", "n_cars_class", "n_bikes_class", "ovgk"):
        df[col] = df[col].astype("string")
    return df[["household_id", "income_class", "n_cars_class", "n_bikes_class", "ovgk"]]


def load_households_synthetic(
    db: duckdb.DuckDBPyConnection,
    enriched_pickle: Optional[Path],
    households_pickle: Optional[Path] = None,
) -> int:
    """Insert one row per household with attribute fields."""
    if enriched_pickle is not None and enriched_pickle.exists():
        df = build_synthetic_households_frame(enriched_pickle)
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
        log.warning("synthetic households: no enriched cache and no households cache - table left empty")
        return 0
    log.warning("synthetic households: enriched cache missing - falling back to %s "
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

    The cache's key column is called ``person_id`` but in this dataset it
    equals household_id (the household's reference person).
    """
    with open(households_pickle, "rb") as f:
        df = pickle.load(f)
    if "person_id" not in df.columns:
        log.warning("microcensus households cache has no person_id key - skipping")
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


_MATSIM_ACTIVITY_RENAMES = {
    "person": "person_id",
    "activity_number": "activity_index",
    "activity_type": "purpose",
    "coord_x": "x",
    "coord_y": "y",
}


def parse_activities_csv(path: Path) -> pd.DataFrame:
    """Read output_activities.csv[.gz] into a DataFrame.

    Handles both the eqasim layout (person_id/activity_index/purpose/x/y) and the
    MATSim-native one (person/activity_number/activity_type/coord_x/coord_y).
    eqasim emits literal -Infinity/Infinity for first/last activity boundaries;
    these are converted to NaN. Non-numeric (freight) person IDs are dropped.
    """
    df = pd.read_csv(path, sep=";", dtype={"person_id": str, "person": str},
                     na_values=["", "NA", "NaN", "-Infinity", "Infinity"])
    matsim_native = "person_id" not in df.columns and "person" in df.columns
    if matsim_native:
        df = df.rename(columns=_MATSIM_ACTIVITY_RENAMES)
    df = _filter_numeric_person_id(df, "activities")
    df["activity_index"] = pd.to_numeric(df["activity_index"], errors="coerce").astype("int32")
    if matsim_native:
        # MATSim numbers from 1, eqasim (and the webmap schema) from 0
        df["activity_index"] -= 1
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
    """Parse and insert activities from a CSV."""
    if activities_csv is None or not activities_csv.exists():
        log.warning("activities CSV not found - table left empty")
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


_MATSIM_TRIP_RENAMES = {
    "person": "person_id",
    "trip_number": "trip_index",
    "dep_time": "departure_time",
    "trav_time": "travel_time",
    "traveled_distance": "network_distance",
    "euclidean_distance": "crowfly_distance",
    "start_activity_type": "preceding_purpose",
    "end_activity_type": "following_purpose",
    "start_x": "origin_x", "start_y": "origin_y",
    "end_x": "dest_x", "end_y": "dest_y",
}


def _hhmmss_to_seconds(series: pd.Series) -> pd.Series:
    """MATSim-native CSV times ('HH:MM:SS', hours may exceed 24) to seconds."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    parts = series.astype("string").str.split(":", expand=True)
    if parts.shape[1] != 3:
        return pd.to_numeric(series, errors="coerce")
    return (pd.to_numeric(parts[0], errors="coerce") * 3600
            + pd.to_numeric(parts[1], errors="coerce") * 60
            + pd.to_numeric(parts[2], errors="coerce"))


def parse_trips_csv(path: Path) -> pd.DataFrame:
    """Read output_trips.csv[.gz] into a DataFrame.

    Handles both the eqasim layout (person_id/person_trip_id/mode/...) and the
    MATSim-native one (person/trip_number/main_mode with HH:MM:SS times)."""
    df = pd.read_csv(path, sep=";", dtype={"person_id": str, "person": str},
                     na_values=["", "NA", "NaN"])
    matsim_native = "person_id" not in df.columns and "person" in df.columns
    if matsim_native:
        df = df.rename(columns=_MATSIM_TRIP_RENAMES)
        for tcol in ("departure_time", "travel_time"):
            df[tcol] = _hhmmss_to_seconds(df[tcol])
    df = _filter_numeric_person_id(df, "trips")
    df = df.rename(columns={
        "person_trip_id": "trip_index",
        "mode": "main_mode",
        "routed_distance": "network_distance",
        "euclidean_distance": "crowfly_distance",
        "origin_x": "origin_x", "origin_y": "origin_y",
        "destination_x": "dest_x", "destination_y": "dest_y",
    })
    df["trip_index"] = pd.to_numeric(df["trip_index"], errors="coerce").astype("int32")
    if matsim_native:
        # 0-based like eqasim and spider_link_index (events trip counter)
        df["trip_index"] -= 1
    df["main_mode"] = df["main_mode"].apply(_canonical_mode)
    df["preceding_purpose"] = df["preceding_purpose"].apply(_canonical_purpose)
    df["following_purpose"] = df["following_purpose"].apply(_canonical_purpose)
    return df


# Columns of eqasim's trips/activities CSVs, the layout the webapp ingests
# (matsim/eqasim_{trips,activities}.csv in the download bundle).
_EQASIM_TRIP_COLUMNS = {
    "person": "person_id", "trip_number": "person_trip_id",
    "start_x": "origin_x", "start_y": "origin_y",
    "end_x": "destination_x", "end_y": "destination_y",
    "dep_time": "departure_time", "trav_time": "travel_time",
    "traveled_distance": "routed_distance", "euclidean_distance": "euclidean_distance",
    "main_mode": "mode",
    "start_activity_type": "preceding_purpose", "end_activity_type": "following_purpose",
}
_EQASIM_ACTIVITY_COLUMNS = {
    "person": "person_id", "activity_number": "activity_index",
    "activity_type": "purpose", "start_time": "start_time", "end_time": "end_time",
    "coord_x": "x", "coord_y": "y",
}


def write_eqasim_trips_csv(src: Path, dst: Path) -> Path:
    """MATSim-native output_trips.csv[.gz] -> eqasim trips layout (0-based
    person_trip_id, times in seconds, raw mode/purpose values)."""
    df = pd.read_csv(src, sep=";", dtype={"person": str, "person_id": str}, low_memory=False)
    if "person" not in df.columns:  # already eqasim layout
        df.to_csv(dst, sep=";", index=False)
        return dst
    df = df[list(_EQASIM_TRIP_COLUMNS)].rename(columns=_EQASIM_TRIP_COLUMNS)
    df["person_trip_id"] = df["person_trip_id"].astype("int64") - 1
    for tcol in ("departure_time", "travel_time"):
        df[tcol] = _hhmmss_to_seconds(df[tcol])
    df.to_csv(dst, sep=";", index=False)
    log.info("wrote %d trips in eqasim layout -> %s", len(df), dst)
    return dst


def write_eqasim_activities_csv(src: Path, dst: Path) -> Path:
    """MATSim-native output_activities.csv[.gz] -> eqasim activities layout
    (0-based activity_index, x/y)."""
    df = pd.read_csv(src, sep=";", dtype={"person": str, "person_id": str}, low_memory=False)
    if "person" not in df.columns:
        df.to_csv(dst, sep=";", index=False)
        return dst
    df = df[list(_EQASIM_ACTIVITY_COLUMNS)].rename(columns=_EQASIM_ACTIVITY_COLUMNS)
    df["activity_index"] = df["activity_index"].astype("int64") - 1
    df.to_csv(dst, sep=";", index=False)
    log.info("wrote %d activities in eqasim layout -> %s", len(df), dst)
    return dst


def load_trips_synthetic(
    db: duckdb.DuckDBPyConnection,
    trips_csv: Optional[Path],
) -> int:
    if trips_csv is None or not trips_csv.exists():
        log.warning("synthetic trips CSV not found - table left empty")
        return 0
    df = parse_trips_csv(trips_csv)
    return _insert_trips(db, df)


def load_trips_microcensus(
    db: duckdb.DuckDBPyConnection,
    trips_pickle: Path,
) -> int:
    """Load microcensus trips. Returns row count.

    person_id is mapped to household_id * 100 + 1, assuming the diary respondent
    is hhpers_id=1 (head of household); keep in sync with derive_activities_microcensus.
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
    """Reconstruct N+1 activities per diary respondent from N MZ trips.

    MZ trip purpose is the purpose at the destination (start of the next
    activity). person_id = raw_pid * 100 + 1, as in load_trips_microcensus.
    """
    with open(trips_pickle, "rb") as f:
        tup = pickle.load(f)
    trips = tup[0].copy() if isinstance(tup, tuple) else tup.copy()
    if trips.empty:
        log.warning("microcensus trips empty - no activities derived")
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

    from .hex import H3_RES_COARSE, H3_RES_MID, hex_for_xy_lv95, hex_parent_int

    df["hilbert_origin"] = hilbert_2d(df["origin_x"].to_numpy(),
                                      df["origin_y"].to_numpy(),
                                      CH_BBOX_LV95)
    origin_h3_9 = hex_for_xy_lv95(
        df["origin_x"].to_numpy(), df["origin_y"].to_numpy(), H3_RES_MID,
    )
    dest_h3_9 = hex_for_xy_lv95(
        df["dest_x"].to_numpy(), df["dest_y"].to_numpy(), H3_RES_MID,
    )
    df["origin_h3_res9"] = origin_h3_9
    df["dest_h3_res9"] = dest_h3_9
    df["origin_h3_res6"] = hex_parent_int(origin_h3_9, H3_RES_COARSE)
    df["dest_h3_res6"] = hex_parent_int(dest_h3_9, H3_RES_COARSE)
    df = df.sort_values("hilbert_origin", kind="mergesort").reset_index(drop=True)

    n_total = 0
    cols_payload = ["person_id", "trip_index", "departure_time", "travel_time",
                    "main_mode", "preceding_purpose", "following_purpose",
                    "network_distance", "crowfly_distance",
                    "origin_x", "origin_y", "dest_x", "dest_y", "hilbert_origin",
                    "origin_h3_res9", "dest_h3_res9",
                    "origin_h3_res6", "dest_h3_res6"]
    for start in range(0, len(df), TRIP_BATCH):
        chunk = df.iloc[start:start + TRIP_BATCH][cols_payload]
        db.register("_tmp_trips", chunk)
        db.execute("""
            INSERT INTO trips
                (person_id, trip_index, departure_time, travel_time,
                 main_mode, preceding_purpose, following_purpose,
                 network_distance, crowfly_distance,
                 origin_pt, dest_pt, hilbert_origin,
                 origin_h3_res9, dest_h3_res9,
                 origin_h3_res6, dest_h3_res6)
            SELECT person_id, trip_index, departure_time, travel_time,
                   main_mode, preceding_purpose, following_purpose,
                   network_distance, crowfly_distance,
                   ST_Point(origin_x, origin_y),
                   ST_Point(dest_x, dest_y),
                   hilbert_origin,
                   origin_h3_res9, dest_h3_res9,
                   origin_h3_res6, dest_h3_res6
            FROM _tmp_trips
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
    if s.endswith("_loop"):  # MATSim-native round-trip variants (car_loop, ...)
        s = s[:-len("_loop")]
    return s if s in _MODE_BUCKETS else "walk"
