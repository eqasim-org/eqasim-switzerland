"""Audit pooled tertiary OD support using the existing survey and spatial stages."""

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from data.od.education_matrix import prepare_observations
from data.od.tertiary_education_smoothing import (
    DEFAULT_PRIOR_STRENGTH, DEFAULT_NEIGHBORS, DEFAULT_BANDWIDTH, DEFAULT_DISTANCE_DECAY)
from data.statent.education_institutions import DEFAULT_CAMPUS_DISTANCE


class InputContext:
    """Run the same input stages without fitting the optional municipality-type model."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.cache = {}

    def config(self, name):
        return {"data_path": self.data_path, "random_seed": 518745,
                "include_external_population": False, "cross_border_exclude_shapefiles": None,
                "tertiary_od_prior_strength": DEFAULT_PRIOR_STRENGTH,
                "tertiary_od_neighbors": DEFAULT_NEIGHBORS,
                "tertiary_od_bandwidth": DEFAULT_BANDWIDTH,
                "tertiary_od_distance_decay": DEFAULT_DISTANCE_DECAY,
                "education_campus_cluster_distance": DEFAULT_CAMPUS_DISTANCE}[name]

    def progress(self, items, **kwargs):
        return items

    def stage(self, name):
        if name not in self.cache:
            module = "data.spatial.municipality_type_data" if name == "data.spatial.municipality_types" else name
            self.cache[name] = importlib.import_module(module).execute(self)
        return self.cache[name]


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.spatial.zones")
    context.stage("data.od.tertiary_education_matrix")
    context.config("analysis_path")


def audit(df_survey, df_zones, matrices, output_path):
    """Report domestic pooled OD support without interpreting missing origins as zero students."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df = prepare_observations(df_survey, tertiary_only=True)
    df["squared_weight"] = df["weight"] ** 2
    cells = []
    for level in ["municipality", "quarter"]:
        grouped = df.groupby(["home_" + level + "_id", "education_zone_id"]).agg(
            respondents=("weight", "size"), weighted_persons=("weight", "sum"),
            squared_weight=("squared_weight", "sum"),
        ).reset_index().rename(columns={"home_" + level + "_id": "zone_level_id"})
        grouped = grouped.merge(
            df_zones[df_zones["zone_level"].eq(level)][["zone_id", "zone_level_id"]],
            on="zone_level_id", how="inner",
        ).rename(columns={"zone_id": "origin_zone_id", "education_zone_id": "destination_zone_id"})
        grouped["origin_level"] = level
        cells.append(grouped)
    cells = pd.concat(cells, ignore_index=True)
    cells["weight_based_effective_n"] = cells["weighted_persons"] ** 2 / cells["squared_weight"]
    cells.to_csv(output_path / "od_cells.csv", index=False)

    origins = df_zones[df_zones["zone_level"].isin(["municipality", "quarter"])].copy()
    support = cells.groupby("origin_zone_id").agg(
        respondents=("respondents", "sum"), weighted_persons=("weighted_persons", "sum"),
        squared_weight=("squared_weight", "sum"), destinations=("destination_zone_id", "size"),
        largest_destination_weight=("weighted_persons", "max"),
    )
    origins = origins.merge(support, left_on="zone_id", right_index=True, how="left")
    count_columns = ["respondents", "destinations"]
    origins[count_columns] = origins[count_columns].fillna(0).astype(int)
    origins[["weighted_persons", "squared_weight"]] = origins[["weighted_persons", "squared_weight"]].fillna(0)
    origins["weight_based_effective_n"] = origins["weighted_persons"] ** 2 / origins["squared_weight"].replace(0, np.nan)
    origins["largest_destination_share"] = origins["largest_destination_weight"] / origins["weighted_persons"].replace(0, np.nan)
    origins.to_csv(output_path / "origin_support.csv", index=False)

    df.groupby("survey_year").agg(
        respondents=("weight", "size"), weighted_persons=("weight", "sum"),
    ).reset_index().to_csv(output_path / "usable_observations.csv", index=False)
    df.groupby("education_zone_id").agg(
        respondents=("weight", "size"), weighted_persons=("weight", "sum"),
    ).reset_index().merge(
        df_zones, left_on="education_zone_id", right_on="zone_id", how="left",
    ).to_csv(output_path / "destination_support.csv", index=False)
    pdf, cdf = matrices
    np.savez_compressed(output_path / "tertiary_matrix.npz", pdf=pdf, cdf=cdf,
                        zone_ids=df_zones["zone_id"].to_numpy())
    (output_path / "README.md").write_text(
        "# Pooled tertiary OD support\n\n"
        "Current education codes 9–13 are pooled, without filtering on institution "
        "type or detailed commuting status. Reported foreign study and unusable "
        "home/education zones are excluded. Geography is normalized using the "
        "existing spatial and processed survey stages.\n\n"
        "Municipality origins include respondents assigned to their quarters; "
        "quarter origins are also reported. Do not sum the two levels as independent "
        "observations. Destination quarters are not additionally counted as destination "
        "municipalities, matching the existing matrix convention.\n\n"
        "Weights are pooled annual expansion weights. Row probabilities are normalized; "
        "pooled weighted totals are not a one-year population estimate. "
        "weight_based_effective_n is (sum weights)^2 / sum squared weights; it is "
        "a weight-concentration diagnostic, not a survey-design confidence interval.\n\n"
        "Municipality and quarter rows use parent/geographic smoothing, including "
        "origins without observations. Other zone levels remain NaN. "
        "The general education matrix retains its existing own-zone fallback.\n",
        encoding="utf-8",
    )
    return origins.groupby("zone_level", observed=True)["respondents"].agg(
        origins="size", unsupported=lambda x: x.eq(0).sum(),
        below_five=lambda x: ((x > 0) & (x < 5)).sum(),
        median="median",
    )


def execute(context):
    return audit(
        context.stage("data.structural_survey.structural_survey"),
        context.stage("data.spatial.zones"),
        context.stage("data.od.tertiary_education_matrix"),
        Path(context.config("analysis_path")) / "tertiary_education_od",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    context = InputContext(args.data_path)
    survey = context.stage("data.structural_survey.structural_survey")
    print(audit(survey, context.stage("data.spatial.zones"),
                context.stage("data.od.tertiary_education_matrix"),
                args.output_path).to_string())


if __name__ == "__main__":
    main()
