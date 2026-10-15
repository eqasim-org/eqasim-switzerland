"""Student-classifier geography: school departure, otherwise registered residence."""

import geopandas as gpd
import numpy as np
import pandas as pd

from data.structural_survey.education_origins import education_origins


def configure(context):
    for stage in ["data.structural_survey.structural_survey", "data.spatial.municipalities",
                  "data.spatial.districts", "data.spatial.cantons", "data.spatial.municipality_types"]:
        context.stage(stage)


def validate(context):
    # Origin parsing is shared with OD preparation, outside this stage's source.
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(__file__).with_name("education_origins.py").read_bytes()).hexdigest()


def municipality_features(municipalities, districts, cantons, types):
    """Map each municipality to consistent administrative and settlement features.

    Interior points avoid ambiguous polygon overlaps at municipal boundaries.
    This small lookup also covers municipalities with no residence respondents.
    """
    points = municipalities[["municipality_id", "geometry"]].to_crs(2056).copy()
    points.geometry = points.geometry.representative_point()
    result = points[["municipality_id"]].copy()
    for field, polygons in [("district_id", districts), ("canton_id", cantons)]:
        matched = gpd.sjoin(points, polygons[[field, "geometry"]].to_crs(2056),
                            how="left", predicate="intersects")
        if matched.index.duplicated().any():
            raise ValueError("Student geography requires an unambiguous %s per municipality" % field)
        result[field] = matched[field].reindex(points.index).to_numpy()
    result = result.merge(types[["municipality_id", "municipality_type"]],
                          on="municipality_id", how="left", validate="one_to_one")
    # Boundary files also contain lakes, Liechtenstein and foreign enclaves.
    # They are not Swiss residence municipalities. Validate coverage of actual
    # domestic departure observations below, rather than inventing Swiss IDs.
    return pd.DataFrame(result.dropna(subset=["district_id", "canton_id", "municipality_type"]))


def prepare_student_training(survey, geography):
    """Change classifier predictors on a copy, keeping labels, weights and all rows.

    Nonstudents retain residence geography: a usable education OD pair is never
    a condition for entering the binary classifier. Explicit foreign departures
    remain an outside-Switzerland category instead of a Swiss residence fallback.
    The source survey's home/work geography is never overwritten in its cache.
    """
    result = survey.copy().reset_index(drop=True)
    # Processed survey geography can be categorical. Convert before introducing
    # the explicit foreign category; pandas 3 rejects unregistered categories.
    result["municipality_type"] = result.municipality_type.astype("string")
    for field in ["district_id", "canton_id"]:
        result[field] = pd.to_numeric(result[field], errors="coerce").astype("Int64")
    origin = education_origins(result)
    student = result.is_student.eq(1)
    departure = student & origin.education_origin_source.eq("school_departure")
    abroad = student & origin.education_origin_source.eq("abroad")
    result["student_origin_source"] = "residence"
    result.loc[student & ~departure & ~abroad, "student_origin_source"] = "residence_fallback"
    result.loc[departure, "student_origin_source"] = "school_departure"
    result.loc[abroad, "student_origin_source"] = "abroad"
    result["student_origin_municipality_id"] = pd.to_numeric(result.home_municipality_id, errors="coerce").astype("Float64")
    result.loc[departure, "student_origin_municipality_id"] = origin.loc[departure, "education_origin_municipality_id"]
    result.loc[abroad, "student_origin_municipality_id"] = np.nan
    lookup = geography.set_index("municipality_id")
    if not lookup.index.is_unique:
        raise ValueError("Student geography lookup contains duplicate municipalities")
    for field in ["district_id", "canton_id", "municipality_type"]:
        values = result.student_origin_municipality_id.map(lookup[field])
        if values.loc[departure].isna().any():
            raise ValueError("School-departure municipality has unresolved student predictors")
        result.loc[departure, field] = values.loc[departure].to_numpy()
        result.loc[abroad, field] = "outside_switzerland" if field == "municipality_type" else -1
    # This field participates in the existing classifier's missing-data gate,
    # though municipality ID itself is not a fitted predictor.
    result["home_municipality_id"] = result.student_origin_municipality_id.where(~abroad, -1)
    return result


def execute(context):
    lookup = municipality_features(context.stage("data.spatial.municipalities")[0],
        context.stage("data.spatial.districts"), context.stage("data.spatial.cantons"),
        context.stage("data.spatial.municipality_types"))
    return prepare_student_training(context.stage("data.structural_survey.structural_survey"), lookup)
