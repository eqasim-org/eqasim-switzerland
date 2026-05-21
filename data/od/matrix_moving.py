import numpy as np
import pandas as pd
from data.structural_survey.structural_survey import get_filtered_data
import logging
from data.od.matrix import (_build_matrix, AGE_BIN_EDGES, SEX_VALUES, AGE_BINS, DEFAULT_SEGMENT_KEY)

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_od = get_filtered_data(context, "moving")[[
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

    return pdf_matrices, cdf_matrices
