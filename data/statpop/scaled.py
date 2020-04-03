import numpy as np
import pandas as pd

import data.constants as c
from data.statpop.multilevelipf import multilevelipf


def configure(context):
    context.config("enable_scaling", default=False)
    context.config("scaling_year", default=c.BASE_SCALING_YEAR)
    context.stage("data.statpop.statpop")
    context.stage("data.statpop.projections.households")
    context.stage("data.statpop.projections.population")

def execute(context):
    df_statpop = context.stage("data.statpop.statpop")

    if context.config["enable_scaling"]:

        df_household_controls = context.stage("data.statpop.projections.households")
        df_population_controls = context.stage("data.statpop.projections.population")

        print("  Number of households in household controls :", df_household_controls["weight"].sum())

        print("  Number of households before scaling :", df_statpop["household_id"].unique().shape[0])
        print("  Number of persons before scaling :", df_statpop["person_id"].unique().shape[0])

        # we need to add a new household class column with only as many categories as the controls
        number_household_classes = len(df_household_controls["household_size_class_projection"].unique())
        df_statpop["household_size_class_projection"] = np.minimum(number_household_classes, df_statpop["household_size"]) - 1

        # set up fitting problem
        problem = multilevelipf.fitting_problem(df_statpop,
                                                group_controls=[df_household_controls], group_id="household_id",
                                                individual_controls=[df_population_controls], individual_id="person_id")
        # perform fitting
        df_statpop = multilevelipf.fit(problem, algorithm="ipu", tol_abs=1e-2, tol_rel=1e-2, maxiter=100, parallelize_on="canton_id")
        del df_statpop["household_size_class_projection"]

        # TODO: The expansion factors are rounded here by simply taking first the integer part
        # as the base value and the remainder as a probability of have an extra household.
        # An array of random doubles is then generated and compared to these probabilities to decide whether to add
        # this remaining household. However, KM used the "Truncate-Replicate-Sample" method in his version. We should
        # consider this maybe in the future.
        df_household_expansion_factors = df_statpop[["household_id", "expansion_factor"]].drop_duplicates("household_id")
        probability = (df_household_expansion_factors["expansion_factor"] - np.floor(df_household_expansion_factors["expansion_factor"])).values
        df_household_expansion_factors["expansion_factor"] = np.floor(df_household_expansion_factors["expansion_factor"])
        df_household_expansion_factors["expansion_factor"] += np.random.random(size = (len(probability),)) < probability
        del df_statpop["expansion_factor"]
        df_statpop = pd.merge(df_statpop, df_household_expansion_factors, on="household_id")

        # duplicate households
        df_households = df_statpop[["household_id", "expansion_factor"]].drop_duplicates("household_id")
        indices = np.repeat(np.arange(df_households.shape[0]), df_households["expansion_factor"].astype(np.int64).values)
        df_households = df_households.iloc[indices]
        df_households["household_id_new"] = np.arange(df_households.shape[0]) + 1
        del df_households["expansion_factor"]

        # merge duplicated households back into statpop
        df_statpop = pd.merge(df_statpop, df_households, on="household_id").drop("expansion_factor", axis=1)
        df_statpop["household_id"] = df_statpop["household_id_new"]
        del df_statpop["household_id_new"]

        # sort by household id and generate new person ids
        df_statpop = df_statpop.sort_values(by=["household_id", "person_id"])
        df_statpop["person_id"] = np.arange(df_statpop.shape[0]) + 1

        print("  Number of households after scaling :", df_statpop["household_id"].unique().shape[0])
        print("  Number of persons after scaling :", df_statpop["person_id"].unique().shape[0])

    return df_statpop
