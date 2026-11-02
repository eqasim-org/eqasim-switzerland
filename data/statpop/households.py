import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def classify_non_private_residences(persons, residences, radius=250., maximum_age=39):
    """Describe possible dorm proximity without assigning student/employment status.

    The age upper bound is exclusive; only recovered synthetic households are
    eligible. Private households and younger unresolved residents keep ordinary
    modelling. Nearby younger residents also use ordinary models; this flag is
    evidence about accommodation, not proof of student status or affiliation.
    """
    if not np.isfinite(radius) or radius < 0 or maximum_age <= 18:
        raise ValueError("Non-private residence classification requires valid radius/age bounds")
    result = persons.copy()
    result["non_private_dorm_candidate_households"] = False
    result["non_private_home_only_households"] = False
    result["non_private_residence_distance"] = np.nan
    result["non_private_residence_egid"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["non_private_residence_evidence"] = "not_recovered"
    selected = result["is_synthetic_household"].fillna(False)
    if not selected.any():
        return result
    if residences.empty:
        raise ValueError("Recovered residence classification requires residence candidates")
    tree = cKDTree(residences[["GKODE", "GKODN"]].to_numpy(dtype=float))
    distances, indices = tree.query(result.loc[selected, ["home_x", "home_y"]].to_numpy(dtype=float))
    result.loc[selected, "non_private_residence_distance"] = distances
    result.loc[selected, "non_private_residence_egid"] = residences.iloc[indices].EGID.to_numpy()
    result.loc[selected, "non_private_residence_evidence"] = residences.iloc[indices].evidence.to_numpy()
    candidates = selected & result.age.ge(16) & result.age.lt(maximum_age) & result.non_private_residence_distance.le(radius)
    # The agreed home-only scenario applies to the older recovered remainder,
    # independently of distance. Younger unresolved residents remain ordinary.
    result.loc[selected & result.age.ge(maximum_age), "non_private_home_only_households"] = True
    result.loc[candidates, "non_private_dorm_candidate_households"] = True
    return result


def attach_persons(df_persons, df_households, retain_missing_households=True):
    """Attach private households and recover geocoded missing records as singletons."""
    df = df_persons.merge(df_households, on="household_id", how="left",
                          validate="many_to_one", indicator="_household_join")
    missing = df["_household_join"].eq("left_only")
    # Positive coordinates alone include administrative collection locations.
    # INDIC_EGID=1 and the existing centre-location flag jointly exclude these.
    coordinates = df[["home_x", "home_y"]].to_numpy(dtype=float)
    geocoded = (np.isfinite(coordinates).all(axis=1)
                & (coordinates > 0).all(axis=1)
                & df["statpop_geocoding_indicator"].eq(1)
                & ~df["collective_housing_resident"])
    recover = missing & geocoded & retain_missing_households
    df["is_synthetic_household"] = recover
    # This preserves the original institution/collection ID (including -9).
    # statpop_household_id remains a UNIQUE modelling key for the scaling stage.
    df["statpop_source_household_id"] = df["household_id"]
    if recover.any():
        # Reserve IDs above BOTH the private table and every source linked ID.
        # Sorting person IDs makes allocation independent of input row order.
        offset = max(0, int(df["household_id"].max()),
                     int(df_households["household_id"].max()) if len(df_households) else 0)
        count = int(recover.sum())
        if offset > np.iinfo(np.int64).max - count:
            raise ValueError("No int64 household ID range available for synthetic households")
        indices = df.loc[recover].sort_values("person_id").index
        df.loc[indices, "household_id"] = offset + np.arange(1, count + 1, dtype=np.int64)
    return df.loc[~missing | recover].drop(columns="_household_join").copy()


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")

    import lzma as xz
    import data.utils

    with open("%s/statpop/STATPOP_2023_HOUSEHOLD_CH_K.csv" % data_path, mode="rb") as f:
        fields = {
            "HOUSEHOLDID": int
            #"Plausibel": int
        }

        renames = {
            "HOUSEHOLDID": "household_id"
            #"Plausibel": "plausible"
        }

        return data.utils.read_csv(context, f, fields, renames, total=3964840)
