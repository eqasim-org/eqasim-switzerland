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

        # rename household_size_class column
        df_household_controls = df_household_controls.rename({"household_size_class": "household_size_class_projection"}, axis=1)

        # we need to add a new household class column with only as many categories as the controls
        number_household_classes = len(df_household_controls["household_size_class_projection"].unique())
        df_statpop["household_size_class_projection"] = np.minimum(number_household_classes, df_statpop["household_size"]) - 1

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

        print("Constructed %d IPU fitting problems." % len(problems))
        print("Starting IPU.")

        # Run IPU algorithm in parallel
        with context.progress(label="Performing IPU on STATPOP by canton...", total=len(problems)):
            with context.parallel(processes=processes) as parallel:
                df_households, convergence = [], []

                for df_household_item, convergence_item in parallel.imap_unordered(process, problems):
                    df_households.append(df_household_item)
                    convergence.append(convergence_item)

        df_households = pd.concat(df_households)
        print("Convergence rate:", np.round(np.mean(convergence), 3))

        # Generate new unique ids
        print("Generating new household ids.")
        df_households["household_id_new"] = np.arange(df_households.shape[0]) + 1
        del df_statpop["household_id"]

        # Merge the new household ids onto statpop by statpop_household_id (i.e. original id)
        df_statpop = pd.merge(df_statpop, df_households, on="statpop_household_id")
        df_statpop["household_id"] = df_statpop["household_id_new"]
        del df_statpop["household_id_new"]

        # sort by household id and generate new person ids
        print("Generating new person ids.")
        df_statpop = df_statpop.sort_values(by=["household_id", "person_id"])
        df_statpop["person_id"] = np.arange(df_statpop.shape[0]) + 1

        # remove unneeded columns
        del df_statpop["household_size_class_projection"]

        print("Number of households in household controls :", df_household_controls["weight"].sum())
        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of households after scaling :", len(df_statpop["household_id"].unique()))
        print("Number of persons after scaling :", len(df_statpop["person_id"].unique()))

    return df_statpop


def process(context, problem):

    # Create the IPU solver
    ipu_solver = IPUSolver(group_rel_tol=1e-4, group_abs_tol=1, ind_rel_tol=1e-5, ind_abs_tol=10, max_iter=2000)

    # Fit the problem, which results a df with expansion factors and whether the algorithm converged
    df_result, convergence = ipu_solver.fit(problem)

    df_households = []
    # Integerize the results using the "Truncate-Replicate-Sample" method
    # We loop through the groups here to get a better fit by household size
    # because we know they are mutually exclusive (but this is not general)
    for i, group_control in enumerate(problem.group_controls):
        group_weight = group_control[0]
        group_filter = group_control[1]

        # 1) Truncate - here we compute both the integer part (count) and remainder of the weight for each household
        df_hh_group = (df_result[group_filter][["household_id", "statpop_household_id", "expansion_factor"]]
                       .drop_duplicates("household_id"))
        weights = df_hh_group["expansion_factor"].values
        counts = np.floor(weights).astype(np.int)
        remainders = weights - counts

        # 2) Replicate - We duplicate the households based on the count
        indices = np.repeat(list(df_hh_group.index), counts)
        df_replicate = df_hh_group.loc[indices]

        # 3) Sample - We sample the required remaining households based on the remainders without replacement
        indices = np.random.choice(list(df_hh_group.index), int(np.round(np.sum(weights) - np.sum(counts))),
                                   replace=False, p=(remainders / np.sum(remainders)))
        df_sample = df_hh_group.loc[indices]

        # We combine the replicated and sampled households
        df = pd.concat([df_replicate, df_sample])

        # We add them to our list by group
        df_households.append(df)

    df_households = pd.concat(df_households).drop("expansion_factor", axis=1)
    df_households["household_id"] = np.arange(df_households.shape[0]) + 1

    context.progress.update()

    # return only duplicated households with new and original ids
    return df_households, convergence
