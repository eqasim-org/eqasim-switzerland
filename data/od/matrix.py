import numpy as np
import pandas as pd
import logging
from data.structural_survey.structural_survey import get_filtered_data
logger = logging.getLogger("synpp")

AGE_BIN_EDGES = np.array([30, 45, 65], dtype=float)
SEX_VALUES = (0, 1)
AGE_BINS = (0, 1, 2, 3)
DEFAULT_SEGMENT_KEY = ("all", "all")

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")


def _build_matrix(df_od, df_zones, zone_ids, municipality_ids, quarter_ids):
    municipality_matrix = pd.crosstab(
        df_od["home_municipality_id"],
        df_od["work_zone_id"],
        df_od["weight"],
        aggfunc=sum,
    ).reindex(
        index=pd.Index(municipality_ids, name="municipality_id"),
        columns=pd.Index(zone_ids, name="destination_zone_id"),
    ).fillna(0).reset_index()

    quarter_matrix = pd.crosstab(
        df_od["home_quarter_id"],
        df_od["work_zone_id"],
        df_od["weight"],
        aggfunc=sum,
    ).reindex(
        index=pd.Index(quarter_ids, name="quarter_id"),
        columns=pd.Index(zone_ids, name="destination_zone_id"),
    ).fillna(0).reset_index()

    municipality_matrix = pd.merge(
        municipality_matrix,
        df_zones[df_zones["zone_level"] == "municipality"],
        left_on="municipality_id",
        right_on="zone_level_id",
    )
    del municipality_matrix["municipality_id"]

    quarter_matrix = pd.merge(
        quarter_matrix,
        df_zones[df_zones["zone_level"] == "quarter"],
        left_on="quarter_id",
        right_on="zone_level_id",
    )
    del quarter_matrix["quarter_id"]

    matrix = pd.concat((municipality_matrix, quarter_matrix))
    for column in ("zone_name", "zone_level", "zone_level_id"):
        del matrix[column]

    matrix = matrix.set_index("zone_id")
    matrix = matrix.reindex(index=pd.Index(zone_ids))
    matrix = matrix.values.astype(float)

    f_origin = df_zones["zone_level"].isin(("municipality", "quarter"))
    f_zero = np.sum(matrix, axis=1) == 0.0

    for index in np.where(f_origin & f_zero)[0]:
        matrix[index, :] = 0.0
        matrix[index, index] = 1.0

    matrix[~f_origin & f_zero] = 1.0

    pdf_matrix = matrix / np.sum(matrix, axis=1)[:, np.newaxis]
    pdf_matrix[~f_origin & f_zero, :] = np.nan

    cdf_matrix = np.cumsum(matrix, axis=1)
    cdf_matrix /= cdf_matrix[:, -1][:, np.newaxis]

    return pdf_matrix, cdf_matrix, np.count_nonzero(f_origin & f_zero)

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_od = get_filtered_data(context, "fixed")[[
        "home_municipality_id", "home_quarter_id", "home_zone_id", "home_zone_level",
        "work_municipality_id", "work_quarter_id", "work_zone_id", "work_zone_level",
        "mode", "weight", "crowfly_distance_to_work", "start_work", "sex", "age"
    ]]

    zone_ids = list(df_zones["zone_id"])
    municipality_ids = list(df_zones[df_zones["zone_level"] == "municipality"]["zone_level_id"])
    quarter_ids = list(df_zones[df_zones["zone_level"] == "quarter"]["zone_level_id"])

    # Aggregate matrix fallback (legacy behavior).
    pdf_matrix, cdf_matrix, fixed_count = _build_matrix(df_od, df_zones, zone_ids, municipality_ids, quarter_ids)

    pdf_matrices = {DEFAULT_SEGMENT_KEY: pdf_matrix}
    cdf_matrices = {DEFAULT_SEGMENT_KEY: cdf_matrix}

    df_segmented = df_od.copy()
    df_segmented["sex"] = pd.to_numeric(df_segmented["sex"], errors="coerce")
    df_segmented["age"] = pd.to_numeric(df_segmented["age"], errors="coerce")
    df_segmented = df_segmented[df_segmented["sex"].isin(SEX_VALUES) & df_segmented["age"].notna()].copy()
    df_segmented["sex"] = df_segmented["sex"].astype(int)
    df_segmented["age_bin"] = np.digitize(df_segmented["age"].to_numpy(dtype=float), AGE_BIN_EDGES, right=False)

    fallback_segments = 0
    for sex in SEX_VALUES:
        for age_bin in AGE_BINS:
            f_segment = (df_segmented["sex"] == sex) & (df_segmented["age_bin"] == age_bin)
            key = (int(sex), int(age_bin))

            if np.count_nonzero(f_segment) == 0:
                pdf_matrices[key] = pdf_matrix.copy()
                cdf_matrices[key] = cdf_matrix.copy()
                fallback_segments += 1
                continue

            seg_pdf, seg_cdf, _ = _build_matrix(
                df_segmented.loc[f_segment],
                df_zones,
                zone_ids,
                municipality_ids,
                quarter_ids,
            )
            pdf_matrices[key] = seg_pdf
            cdf_matrices[key] = seg_cdf

    logger.info("  - Finished (%d fixed municipalities)", fixed_count)
    logger.info(
        "  - Built %d sex/age OD segments (%d fallback to aggregate)",
        len(pdf_matrices) - 1,
        fallback_segments,
    )

    # A final note on the structure of these OD matrices:
    # - The origin counts for municipalities contain all originating trips, also
    #   those which are actually assigned to quarters within this zone
    # - The destination counts target the assigned top-level zone. So arrivals
    #   in a quarter are NOT included in the arrivals for the municipality. This
    #   way, arrivals in municipalities with quarters can only happen if the
    #   municipality is not covered 1:1 by the quarters, which is usually the case
    #   in our zoning system. This way the municipality itself will only have little
    #   arrivals, while the quarters will have more.

    return pdf_matrices, cdf_matrices
