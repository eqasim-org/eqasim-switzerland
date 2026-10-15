import numpy as np
import pandas as pd
import logging
from data.structural_survey.education_origins import education_origins

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.config("secondary_od_prior_strength", 100.)
    context.config("secondary_od_neighbors", 50)
    context.config("secondary_od_bandwidth", 20000.)
    context.config("secondary_od_distance_decay", .005)


def validate(context):
    # The shared origin and smoothing helpers are imported by both OD stages.
    import hashlib
    from pathlib import Path
    files = [Path(__file__).with_name("tertiary_education_smoothing.py"),
             Path(__file__).parents[1] / "structural_survey/education_origins.py"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def prepare_observations(df_survey, tertiary_only=False):
    """Select usable domestic OD pairs, pooling tertiary levels when requested."""
    df_od = df_survey[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "education_municipality_id", "education_quarter_id", "education_zone_id", "education_zone_level",
        "mode", "weight", "education_country_code", "current_education", "survey_year"
    ]].copy()

    # Two broad flow models. Student flags and positive weights are required;
    # programme reports from nonstudents do not define regular student commutes.
    codes = range(9, 14) if tertiary_only else range(2, 9)
    selected = (df_survey["is_student"].eq(1) & df_survey["age"].ge(15)
                & df_survey["current_education"].isin(codes)
                & np.isfinite(df_survey["weight"]) & df_survey["weight"].gt(0))
    df_od = df_od.loc[selected]

    # Education OD origins describe the actual start of the school commute,
    # including term-time residences. Preserve residence separately in the
    # survey stage; these home_* names are the existing OD consumer interface.
    # Both broad groups use departure first, with residence fallback explicitly
    # tagged as a modelling assumption. Foreign departures stay foreign.
    origins = education_origins(df_survey.loc[df_od.index])
    for suffix in ["municipality_id", "quarter_id", "zone_id", "zone_level"]:
        df_od["home_" + suffix] = origins["education_origin_" + suffix]
    df_od["education_origin_source"] = origins["education_origin_source"]
    domestic_departure = origins["education_origin_domestic"]
    logger.info("Removed %d observations without a usable domestic education origin", np.count_nonzero(~domestic_departure))
    df_od = df_od.loc[domestic_departure]

    # Exclude reported foreign study before estimating domestic education flows.
    # Negative country codes denote special statuses, not foreign destinations.
    foreign_study = df_od["education_country_code"].gt(0) & df_od["education_country_code"].ne(8100)
    logger.info("Removed %d observations with a reported foreign education country", np.count_nonzero(foreign_study))
    df_od = df_od[~foreign_study]

    # There are many people for which we don't have a valid OD pair
    before_count = len(df_od)
    df_od = df_od[df_od["home_zone_id"].notna()]
    df_od = df_od[df_od["education_zone_id"].notna()]

    unknown_count = before_count - len(df_od)
    logger.info("Removed %d (%.2f%%) observations from structural survey for which no education or home location is known" % (unknown_count, 100 * unknown_count / before_count if before_count else 0.0))

    # Filter out people who are not in a neighboring country (currently, don't exist any)
    before_count = len(df_od)
    df_od = df_od[~(df_od["education_zone_level"] == "country")]
    df_od = df_od[~(df_od["home_zone_level"] == "country")]

    outside_count = before_count - len(df_od)
    logger.info("Removed %d (%.2f%%) observations from structural survey which live or study abroad" % (outside_count, 100 * outside_count / before_count if before_count else 0.0))

    return df_od


def build_matrix(df_od, df_zones, fill_empty_origins=True):
    """Return PDF/CDF rows in spatial-zone order, including municipality and quarter.

    The legacy all-education matrix keeps its own-zone empty-origin fallback.
    Tertiary smoothing instead estimates explicit priors outside this helper.
    """

    #######################################################
    # Create the matrices
    zone_ids = list(df_zones["zone_id"])
    municipality_ids = list(df_zones[df_zones["zone_level"] == "municipality"]["zone_level_id"])
    quarter_ids = list(df_zones[df_zones["zone_level"] == "quarter"]["zone_level_id"])

    municipality_matrix = pd.crosstab(
        df_od["home_municipality_id"], df_od["education_zone_id"],
        df_od["weight"], aggfunc="sum").reindex(
            index=pd.Index(municipality_ids, name="municipality_id"), columns=pd.Index(zone_ids, name="destination_zone_id")
        ).fillna(0).reset_index()

    quarter_matrix = pd.crosstab(
        df_od["home_quarter_id"], df_od["education_zone_id"],
        df_od["weight"], aggfunc="sum").reindex(
            index=pd.Index(quarter_ids, name="quarter_id"), columns=pd.Index(zone_ids, name="destination_zone_id")
        ).fillna(0).reset_index()

    municipality_matrix = pd.merge(
        municipality_matrix, df_zones[df_zones["zone_level"] == "municipality"],
        left_on="municipality_id", right_on="zone_level_id"
    )
    del municipality_matrix["municipality_id"]

    quarter_matrix = pd.merge(
        quarter_matrix, df_zones[df_zones["zone_level"] == "quarter"],
        left_on="quarter_id", right_on="zone_level_id"
    )
    del quarter_matrix["quarter_id"]

    matrix = pd.concat((municipality_matrix, quarter_matrix))
    for column in ("zone_name", "zone_level", "zone_level_id"): del matrix[column]

    matrix = matrix.set_index("zone_id")
    matrix = matrix.reindex(index=pd.Index(zone_ids))
    matrix = matrix.values

    f_origin = df_zones["zone_level"].isin(("municipality", "quarter"))
    f_zero = np.sum(matrix, axis=1) == 0.0

    # Handle origins (quarters or municipalities) with zero observed trips:
    # - These are zones for which no survey respondents have education locations, resulting in zero rows in the matrix.
    # - To avoid issues with downstream processing, we assign all individuals from these zones to stay within their own zone.
    # - Note: This is a placeholder solution; we should be aware that these zones lack empirical data and results may be biased.
    if fill_empty_origins:
        logger.info("Fixing %d municipalities/quarters with zero education destinations", np.count_nonzero(f_origin & f_zero))
        for index in np.where(f_origin & f_zero)[0]:
            matrix[index,:] = 0.0
            matrix[index,index] = 1.0
    else:
        logger.info("Leaving %d unsupported tertiary origins unresolved", np.count_nonzero(f_origin & f_zero))

    totals = np.sum(matrix, axis=1)[:, np.newaxis]
    pdf_matrix = np.full(matrix.shape, np.nan, dtype=float)
    np.divide(matrix, totals, out=pdf_matrix, where=totals > 0)

    cdf_matrix = np.full(matrix.shape, np.nan, dtype=float)
    np.divide(np.cumsum(matrix, axis=1), totals, out=cdf_matrix, where=totals > 0)

    logger.info("Finished education matrix with %d unsupported origins", np.count_nonzero(f_origin & f_zero))

    return pdf_matrix, cdf_matrix


def execute(context):
    from data.od.tertiary_education_smoothing import build_smoothed_matrix
    df_od = prepare_observations(context.stage("data.structural_survey.structural_survey"))
    municipalities, mapping = context.stage("data.spatial.municipalities")
    # Reuse the existing geographic/parent-origin shrinkage. An unobserved origin
    # is never interpreted as proof that all of its students study in that zone.
    pdf, cdf, _ = build_smoothed_matrix(df_od, context.stage("data.spatial.zones"), municipalities,
        context.stage("data.spatial.quarters"), mapping,
        context.config("secondary_od_prior_strength"), context.config("secondary_od_neighbors"),
        context.config("secondary_od_bandwidth"), context.config("secondary_od_distance_decay"))
    return pdf, cdf
