"""Shared education-origin policy for student stocks and domestic OD flows."""

import pandas as pd


def education_origins(survey, residence_fallback=True):
    """Prefer school departure; fall back only for unresolved, not foreign, origins.

    Keep registered residence untouched. The chosen origin is a modelling
    location for education, not a new registration or an observed dorm address.
    Use the finest available departure/residence zone, retaining municipality
    IDs so municipal summaries do not double-count overlapping quarter rows.
    """
    result = pd.DataFrame(index=survey.index)
    status = survey["education_departure_status"]
    departure = status.eq("domestic") & survey["education_departure_zone_id"].notna()
    fallback = (status.eq("unresolved") & survey["home_zone_id"].notna()
                & survey["home_zone_level"].isin(["municipality", "quarter"])) if residence_fallback else pd.Series(False, index=survey.index)
    for suffix in ["municipality_id", "quarter_id", "zone_id", "zone_level"]:
        result["education_origin_" + suffix] = survey["education_departure_" + suffix].where(departure)
        result.loc[fallback, "education_origin_" + suffix] = survey.loc[fallback, "home_" + suffix]
    result["education_origin_source"] = "unresolved"
    result.loc[status.eq("abroad"), "education_origin_source"] = "abroad"
    result.loc[departure, "education_origin_source"] = "school_departure"
    result.loc[fallback, "education_origin_source"] = "residence_fallback"
    result["education_origin_domestic"] = departure | fallback
    return result
