"""Shared covariates, their provenance and lossless V2 feature preparation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.utils import coerce_boolean_series

CONTINUOUS = [
    "age", "income_class", "household_size", "N_children_under_3",
    "N_children_under_6", "N_children_under_12", "N_children_under_18", "N_adults",
    "population_density_log", "commute_distance_log", "dist_home_to_border_log",
]
CATEGORICAL_GROUPS = {
    "geo": ["canton_id", "sp_region", "municipality_type", "ovgk"],
    "cars": ["number_of_cars_class", "car_availability", "driving_license"],
    "emp": ["employed", "is_student", "employment_status", "work_location_type", "job_position"],
    "misc": ["sex", "marital_status", "is_swiss"],
}
CATEGORICAL = [column for group in CATEGORICAL_GROUPS.values() for column in group]
FEATURE_GROUPS = {
    "age": ["age"], "income": ["income_class"],
    "household_composition": ["household_size", "N_children_under_3", "N_children_under_6",
                              "N_children_under_12", "N_children_under_18", "N_adults"],
    "density": ["population_density_log"],
    "residential_context": CATEGORICAL_GROUPS["geo"],
    "mobility_resources": CATEGORICAL_GROUPS["cars"],
    "work_context": ["commute_distance_log", "work_location_type", "job_position"],
    "personal_context": CATEGORICAL_GROUPS["misc"],
    "border_proximity": ["dist_home_to_border_log"],
}


def numeric(frame, column):
    values = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def reconstruct_household_counts(population):
    """Recover counts only from complete households, never from presence flags.

    STATPOP's original stage exports child presence; the household sampler keeps
    whole households. Verify that assumption against reported household size.
    Incomplete households retain genuine reported counts, while ambiguous
    presence-only information is missing. All work is on the consumer's copy.
    """
    frame = population.copy()
    if "household_id" not in frame or "age" not in frame:
        return frame
    size = numeric(frame, "household_size")
    members = frame.groupby("household_id")["person_id"].transform("size")
    age = numeric(frame, "age")
    complete_rows = size.eq(members) & age.notna() & age.ge(0)
    complete = complete_rows.groupby(frame["household_id"]).transform("all") & frame["household_id"].notna()
    for threshold in (3, 6, 12, 18):
        column = f"N_children_under_{threshold}"
        reported = numeric(frame, column)
        is_presence = column in frame and (pd.api.types.is_bool_dtype(frame[column])
                                            or (reported.dropna().isin([0, 1]).all()))
        recovered = (age.lt(threshold) & age.ge(0)).groupby(frame["household_id"]).transform("sum")
        # In this pipeline a numeric 0/1 field can also be a collapsed presence.
        if is_presence:
            reported = reported.where(~reported.eq(1), np.nan)
        frame[column] = reported.where(~complete, recovered)
    adults = age.ge(18).groupby(frame["household_id"]).transform("sum")
    frame["N_adults"] = numeric(frame, "N_adults").where(~complete, adults)
    frame["household_counts_reconstructed"] = complete
    return frame


def prepare_features(source, population, const):
    source, population = source.copy(), reconstruct_household_counts(population)
    for is_source, frame in ((True, source), (False, population)):
        if "age" not in frame:
            # ARE has intervals rather than exact ages. Keep its uncertainty explicit.
            frame["age"] = numeric(frame, "age_class").map({0: 2.5, 1: 11.5, 2: 21, 3: 34.5,
                                                            4: 54.5, 5: 69.5, 6: 82})
            frame["age_is_interval_midpoint"] = True
        age = numeric(frame, "age")
        if not (age.notna() & age.between(0, 120)).all():
            raise ValueError("Matching needs valid exact ages or ARE age intervals")
        frame["age"] = age
        for column in ("sex", "employed", "employment_status"):
            if column not in frame:
                raise ValueError(f"Required matching feature is missing: {column}")
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if not frame["employment_status"].isin([0, 1, 2, 3]).all():
            raise ValueError("Unknown employment_status; cannot construct the mandatory student mask")
        if is_source:
            frame["employed"] = frame["employed"].eq(1).astype(int)
        else:
            frame["employed"] = (frame["employed"].isin([1, 3]) if const.census == "are_synpop"
                                   else frame["employed"].eq(const.EMPLOYED)).astype(int)
        if "is_student" in frame:
            supplied = frame["is_student"].where(frame["is_student"].notna(), frame["employment_status"].isin([2, 3]))
            frame["is_student"] = coerce_boolean_series(supplied, name="is_student").astype(int)
        else:
            # The pipeline contract defines STUDENT=2 and EMPLOYED_STUDENT=3.
            frame["is_student"] = frame["employment_status"].isin([2, 3]).astype(int)
        for column in ("income_class", "household_size", "N_children_under_3", "N_children_under_6",
                       "N_children_under_12", "N_children_under_18", "N_adults"):
            values = numeric(frame, column)
            frame[column] = values.where(values >= 0)
        for column in ("population_density", "commute_distance", "dist_home_to_border"):
            raw = numeric(frame, column)
            frame[column] = raw.where(raw >= 0)
            frame[column + "_log"] = np.log1p(frame[column])
        # Students without employment can still have a usual education commute.
        frame.loc[frame["employed"].ne(1) & frame["is_student"].ne(1),
                  ["commute_distance", "commute_distance_log"]] = np.nan
        for column in ("canton_id", "sp_region", "number_of_cars_class", "marital_status", "job_position"):
            frame[column] = numeric(frame, column).fillna(-1).astype(int)
        if is_source:
            frame["job_position"] = frame["job_position"].map(const.MAP_JOB_POSITIONS_MZ_TO_SURVEY).fillna(-1).astype(int)
        # Job status is undefined for non-workers on both sides. Predicted
        # population job codes must not become an artificial domain identifier.
        frame.loc[frame["employed"].ne(1), "job_position"] = -1
        raw_car = numeric(frame, "car_availability")
        frame["car_availability"] = (np.where(raw_car.notna(), raw_car.ne(const.CAR_AVAILABILITY_NEVER), -1)
                                       if is_source else raw_car.fillna(-1)).astype(int)
        for column in ("driving_license", "is_swiss"):
            values = frame.get(column, pd.Series(pd.NA, index=frame.index))
            known = values.notna()
            frame[column] = -1
            frame.loc[known, column] = coerce_boolean_series(values.loc[known], name=column).astype(int)
        work = frame.get("work_location_type", pd.Series("none", index=frame.index))
        frame["work_location_type"] = work.astype("string").fillna("none").str.lower()
        for column, default in (("municipality_type", "unknown"), ("ovgk", "none")):
            values = frame.get(column, pd.Series(default, index=frame.index))
            frame[column] = values.astype("string").fillna(default).str.lower()
    return source.reset_index(drop=True), population.reset_index(drop=True)


def feature_inventory(source, population):
    rows = []
    for column in CONTINUOUS + CATEGORICAL:
        for label, frame in (("survey", source), ("population", population)):
            values = frame[column]
            missing = values.isna() if column in CONTINUOUS else values.isna() | values.astype("string").eq("-1")
            rows.append({"feature": column, "dataset": label,
                         "role": "continuous" if column in CONTINUOUS else "categorical",
                         "missing_fraction": float(missing.mean()) if len(frame) else None,
                         "distinct_values": int(values.nunique()),
                         "inference_mask": column in ("employed", "is_student")})
    return pd.DataFrame(rows)


def source_token(*modules):
    """Synpp hashes stage modules only; include imported implementation helpers."""
    digest = hashlib.sha256()
    for module in modules:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def fit_feature_processors(source):
    """Fit on training donors only, reserving index zero for unseen categories."""
    raw = source[CONTINUOUS].to_numpy(dtype=float)
    medians = np.array([np.nanmedian(col) if np.isfinite(col).any() else 0.0 for col in raw.T])
    filled = np.where(np.isfinite(raw), raw, medians)
    scaler = StandardScaler().fit(filled, sample_weight=source["person_weight"].to_numpy())
    scaler.matching_imputation_ = medians
    mappings = {}
    for col in CATEGORICAL:
        values = sorted(source[col].astype("string").fillna("<na>").unique())
        mappings[col] = {value: i + 1 for i, value in enumerate(values)}
    return scaler, mappings


def transform_inputs(frame, scaler, mappings):
    raw = frame[CONTINUOUS].to_numpy(dtype=float)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, scaler.matching_imputation_, raw)
    scaled = scaler.transform(filled) if len(frame) else filled
    out = {"cont": np.column_stack([scaled, missing.astype(float)]).astype(np.float32)}
    for col, mapping in mappings.items():
        values = frame[col].astype("string").fillna("<na>")
        out[col] = values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
    out["cardinalities"] = {key: len(mapping) + 1 for key, mapping in mappings.items()}
    return out


class FeatureProcessor:
    """Training-fitted numeric imputation/scaling and categorical vocabularies."""

    def fit(self, frame):
        self.scaler_, self.mappings_ = fit_feature_processors(frame)
        return self

    def transform(self, frame):
        if not hasattr(self, "scaler_"):
            raise RuntimeError("FeatureProcessor.fit must be called before transform")
        return transform_inputs(frame, self.scaler_, self.mappings_)

    @classmethod
    def from_fitted(cls, scaler, mappings):
        processor = cls()
        processor.scaler_, processor.mappings_ = scaler, mappings
        return processor
