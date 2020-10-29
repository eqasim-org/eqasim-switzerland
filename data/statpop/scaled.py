import numpy as np
import pandas as pd

import data.constants as c
from data.statpop.multilevelipf import multilevelipf
from data.statpop.multilevelipf.multilevelipf import FittingProblem, IPUSolver


def configure(context):
    context.config("enable_scaling", default=False)
    context.config("scaling_year", default=c.BASE_SCALING_YEAR)
    context.config("threads")
    context.stage("data.statpop.statpop")
    context.stage("data.statpop.projections.households")
    context.stage("data.statpop.projections.population")


def execute(context):
    df_statpop = context.stage("data.statpop.statpop")

    if context.config("enable_scaling"):

        scaling_year = context.config("scaling_year")

        print("Scaling STATPOP to year", scaling_year, "using IPU.")

        processes = context.config("threads")
        df_household_controls, hh_year = context.stage("data.statpop.projections.households")
        df_population_controls, pop_year = context.stage("data.statpop.projections.population")

        assert hh_year == scaling_year
        assert pop_year == scaling_year

        print("Number of households in household controls :", df_household_controls["weight"].sum())
        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of households before scaling :", len(df_statpop["household_id"].unique()))
        print("Number of persons before scaling :", len(df_statpop["person_id"].unique()))

        # we need to add a new household class column with only as many categories as the controls
        number_household_classes = len(df_household_controls["household_size_class_projection"].unique())
        df_statpop["household_size_class_projection"] = np.minimum(number_household_classes,
                                                                   df_statpop["household_size"]) - 1

        # create IPU fitting problem by canton
        problems = []
        canton_ids = list(df_statpop.sort_values("canton_id")["canton_id"].unique())

        for canton_id in context.progress(canton_ids, label="Constructing separate IPU fitting problems by canton..."):
            # select sub df
            df = df_statpop[df_statpop["canton_id"] == canton_id].copy()
            df = multilevelipf.add_expansion_factor_column(df)

            # get group controls, perform checks and convert to filters
            group_controls = [df_household_controls[df_household_controls["canton_id"] == canton_id]]
            group_id = "household_id"
            assert multilevelipf.check_control_has_weight_column(group_controls)
            group_controls = multilevelipf.compute_group_filters(df, group_controls)

            # get individual controls, perform checks and convert to filters
            individual_controls = [df_population_controls[df_population_controls["canton_id"] == canton_id]]
            individual_id = "individual_id"
            assert multilevelipf.check_control_has_weight_column(individual_controls)
            individual_controls = multilevelipf.compute_individual_filters(df, group_id, individual_controls)

            # create fitting problem
            problem = FittingProblem(df, group_controls, group_id, individual_controls, individual_id)
            problems.append(problem)

        # Run IPU algorithm in parallel
        with context.progress(label="Performing IPU on STATPOP by canton...", total=len(problems)):
            with context.parallel(processes=processes) as parallel:
                df_results, convergence = [], []

                for df_result_item, convergence_item in parallel.imap_unordered(process, enumerate(problems)):
                    df_results.append(df_result_item)
                    convergence.append(convergence_item)

        df_statpop = pd.concat(df_results).drop("household_size_class_projection", axis=1)
        print("Convergence rate:", np.round(np.mean(convergence), 3))

        # TODO: The expansion factors are rounded here by simply taking first the integer part
        # as the base value and the remainder as a probability of have an extra household.
        # An array of random doubles is then generated and compared to these probabilities to decide whether to add
        # this remaining household. However, KM used the "Truncate-Replicate-Sample" method in his version. We should
        # consider this maybe in the future.
        print("Duplicating STATPOP households based on expansion factors obtained by IPU.")
        df_household_expansion_factors = df_statpop[["household_id", "expansion_factor"]].drop_duplicates(
            "household_id")
        probability = (df_household_expansion_factors["expansion_factor"] - np.floor(
            df_household_expansion_factors["expansion_factor"])).values
        df_household_expansion_factors["expansion_factor"] = np.floor(
            df_household_expansion_factors["expansion_factor"])
        df_household_expansion_factors["expansion_factor"] += np.random.random(size=(len(probability),)) < probability
        del df_statpop["expansion_factor"]
        df_statpop = pd.merge(df_statpop, df_household_expansion_factors, on="household_id")

        # duplicate households
        df_households = df_statpop[["household_id", "expansion_factor"]].drop_duplicates("household_id")
        indices = np.repeat(np.arange(df_households.shape[0]),
                            df_households["expansion_factor"].astype(np.int64).values)
        df_households = df_households.iloc[indices]
        df_households["household_id_new"] = np.arange(df_households.shape[0]) + 1
        del df_households["expansion_factor"]

        # merge duplicated households back into statpop
        print("Generating new household ids.")
        df_statpop = pd.merge(df_statpop, df_households, on="household_id").drop("expansion_factor", axis=1)
        df_statpop["household_id"] = df_statpop["household_id_new"]
        del df_statpop["household_id_new"]

        # sort by household id and generate new person ids
        print("Generating new person ids.")
        df_statpop = df_statpop.sort_values(by=["household_id", "person_id"])
        df_statpop["person_id"] = np.arange(df_statpop.shape[0]) + 1

        print("Number of households in household controls :", df_household_controls["weight"].sum())
        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of households after scaling :", len(df_statpop["household_id"].unique()))
        print("Number of persons after scaling :", len(df_statpop["person_id"].unique()))

    return df_statpop


def process(context, args):
    ipu_solver = IPUSolver(tol_abs=1e-2, tol_rel=1e-2, max_iter=100)
    result, convergence = ipu_solver.fit(args)
    context.progress.update()

    return result, convergence
