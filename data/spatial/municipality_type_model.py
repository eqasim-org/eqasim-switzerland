import geopandas as gpd
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")
    context.stage("data.statpop.persons")
    context.stage("data.statent.statent")
    context.stage("data.spatial.municipality_type_data")
    context.config("municipality_type_outlier_iqr_factor", default=1.5)
    context.config("municipality_type_outlier_min_class_size", default=15)
    context.config("municipality_type_outlier_passes", default=2)


def _build_features(df_municipalities, df_persons, df_statent):
    person_mask = (df_persons["type_of_residence"].isin([1, 2]))
    df_population = (
        df_persons.loc[person_mask]
        .groupby("municipality_id")
        .size()
        .rename("population")
        .reset_index()
    )

    df_companies = (
        df_statent.groupby("municipality_id")
        .size()
        .rename("number_companies")
        .reset_index()
    )

    df_employees = (
        df_statent.groupby("municipality_id")["number_employees"]
        .sum()
        .rename("number_employees")
        .reset_index()
    )

    df_schools = (
        df_statent[df_statent["education_type"].notna()]
        .groupby("municipality_id")
        .size()
        .rename("number_schools")
        .reset_index()
    )

    df_features = gpd.GeoDataFrame(df_municipalities[["municipality_id", "geometry"]], crs=df_municipalities.crs)
    df_features["area_km2"] = df_features["geometry"].area / 1e6

    positive_areas = df_features.loc[df_features["area_km2"] > 0, "area_km2"]
    minimum_area = positive_areas.min() if len(positive_areas) > 0 else 1e-6
    df_features["area_km2"] = df_features["area_km2"].clip(lower=minimum_area)
    df_features = df_features[["municipality_id", "area_km2"]]

    for frame in (df_population, df_companies, df_employees, df_schools):
        df_features = pd.merge(df_features, frame, on="municipality_id", how="left")

    count_columns = ["population", "number_companies", "number_employees", "number_schools"]
    for col in count_columns:
        df_features[col] = df_features[col].fillna(0.0)

    df_features["number_employees"] = df_features["number_employees"].clip(lower=0.0)

    df_features["population_density"] = df_features["population"] / df_features["area_km2"]
    df_features["company_density"] = df_features["number_companies"] / df_features["area_km2"]
    df_features["employee_density"] = df_features["number_employees"] / df_features["area_km2"]
    df_features["school_density"] = df_features["number_schools"] / df_features["area_km2"]
    df_features["employees_per_company"] = (
        df_features["number_employees"] / np.maximum(df_features["number_companies"], 1.0)
    )

    transform_columns = [
        "population",
        "number_companies",
        "number_employees",
        "number_schools",
        "area_km2",
        "population_density",
        "company_density",
        "employee_density",
        "school_density",
        "employees_per_company",
    ]
    for col in transform_columns:
        df_features["log_" + col] = np.log1p(df_features[col])

    return df_features


def _compute_urbanity_score(df):
    score_columns = [
        "log_population_density",
        "log_employee_density",
        "log_company_density",
        "log_school_density",
    ]
    ranked = pd.DataFrame(index=df.index)
    for col in score_columns:
        ranked[col] = df[col].rank(method="average", pct=True)

    return ranked.mean(axis=1)


def _demote_lower_outliers(df, score_col, class_col, iqr_factor, min_class_size, passes):
    class_order = ["rural", "suburban", "urban", "urbancore"]
    demotion_chain = ["urbancore", "urban", "suburban"]
    next_class = {
        "urbancore": "urban",
        "urban": "suburban",
        "suburban": "rural",
    }

    df = df.copy()
    total_demotions = 0

    for _ in range(max(1, int(passes))):
        pass_demotions = 0

        for current_class in demotion_chain:
            class_mask = df[class_col] == current_class
            class_size = int(np.count_nonzero(class_mask))

            if class_size <= max(3, int(min_class_size)):
                continue

            class_scores = df.loc[class_mask, score_col]
            q1 = float(class_scores.quantile(0.25))
            q3 = float(class_scores.quantile(0.75))
            iqr = q3 - q1

            threshold = q1 - float(iqr_factor) * iqr if iqr > 0 else q1
            outlier_indices = df.index[class_mask & (df[score_col] < threshold)]

            max_demotable = max(0, class_size - int(min_class_size))
            if len(outlier_indices) > max_demotable:
                if max_demotable == 0:
                    outlier_indices = outlier_indices[:0]
                else:
                    outlier_indices = (
                        df.loc[class_mask]
                        .nsmallest(max_demotable, score_col)
                        .index
                    )

            if len(outlier_indices) > 0:
                df.loc[outlier_indices, class_col] = next_class[current_class]
                pass_demotions += int(len(outlier_indices))

        total_demotions += pass_demotions
        if pass_demotions == 0:
            break

    df[class_col] = pd.Categorical(df[class_col], categories=class_order)
    return df, total_demotions


def execute(context):
    iqr_factor = float(context.config("municipality_type_outlier_iqr_factor") or 1.5)
    min_class_size = int(context.config("municipality_type_outlier_min_class_size") or 15)
    passes = int(context.config("municipality_type_outlier_passes") or 2)

    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_persons = context.stage("data.statpop.persons")
    df_statent = context.stage("data.statent.statent")
    df_reference_types = context.stage("data.spatial.municipality_type_data")

    required_statent_columns = ["municipality_id", "number_employees", "education_type"]
    df_statent = df_statent[required_statent_columns]

    df_features = _build_features(df_municipalities, df_persons, df_statent)
    df_prediction = pd.merge(
        pd.DataFrame(df_municipalities[["municipality_id"]]),
        pd.DataFrame(df_reference_types[["municipality_id", "municipality_type", "imputed_municipality_type"]]),
        on="municipality_id",
        how="left",
    )

    df_prediction["municipality_type"] = df_prediction["municipality_type"].astype(str)

    df_prediction = pd.merge(
        df_prediction,
        df_features[[
            "municipality_id",
            "log_population_density",
            "log_employee_density",
            "log_company_density",
            "log_school_density",
            "log_population",
            "log_number_employees",
        ]],
        on="municipality_id",
        how="left",
    )

    df_prediction["urbanity_score"] = _compute_urbanity_score(df_prediction)
    df_prediction, total_demotions = _demote_lower_outliers(
        df_prediction,
        score_col="urbanity_score",
        class_col="municipality_type",
        iqr_factor=iqr_factor,
        min_class_size=min_class_size,
        passes=passes,
    )

    logger.info(
        "Applied municipality-type outlier correction (factor=%.2f, min_class_size=%d, passes=%d): demoted %d municipalities.",
        iqr_factor,
        min_class_size,
        passes,
        total_demotions,
    )

    df_prediction = df_prediction[["municipality_id", "municipality_type", "imputed_municipality_type"]]

    df_mapping = pd.merge(pd.DataFrame(df_municipalities[["municipality_id"]]), df_prediction, on="municipality_id", how="left")

    missing_mask = df_mapping["municipality_type"].isna()
    if np.any(missing_mask):
        logger.warning(
            "Could not predict municipality type for %d municipalities. Falling back to rural.",
            int(np.count_nonzero(missing_mask)),
        )
        df_mapping.loc[missing_mask, "municipality_type"] = "rural"

    df_mapping.loc[missing_mask, "imputed_municipality_type"] = True

    type_order = ["rural", "suburban", "urban", "urbancore"]
    df_mapping["municipality_type"] = pd.Categorical(
        df_mapping["municipality_type"], categories=type_order
    )

    type_counts = df_mapping["municipality_type"].value_counts().to_dict()
    logger.info("Predicted municipality type counts: %s", type_counts)

    assert len(df_mapping) == len(df_municipalities)
    assert set(np.unique(df_mapping["municipality_id"])) == set(np.unique(df_municipalities["municipality_id"]))

    return df_mapping[["municipality_id", "municipality_type", "imputed_municipality_type"]]