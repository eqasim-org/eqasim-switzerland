import numpy as np
import pandas as pd

from data.statpop.multilevelipf import multilevelipf
from data.statpop.multilevelipf.multilevelipf import FittingProblem, IPUSolver


def configure(context):
    context.stage("synthesis.population.models.caravailability")
    context.stage("data.statpop.projections.households")
    context.stage("data.statpop.projections.population")
    context.stage("data.constants")

    context.config("enable_scaling", default=False)
    context.config("scaling_year", default=2050)
    
    context.config("random_seed")
    context.config("threads")    
    

def execute(context):
    df_statpop = context.stage("synthesis.population.models.caravailability")
    df_statpop = df_statpop.astype({"canton_id": int})
    
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
            individual_id = "person_id"
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


        print("Number of households in household controls :", df_household_controls["weight"].sum())
        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of households after scaling :", len(df_statpop["household_id"].unique()))
        print("Number of persons after scaling :", len(df_statpop["person_id"].unique()))

    return df_statpop


def process(context, problem):
    import numpy as np
    import pandas as pd

    # RNG
    rng = np.random.RandomState(context.config("random_seed"))

    # Solve IPU (tolerances unchanged)
    ipu_solver = IPUSolver(
        group_rel_tol=1e-2, group_abs_tol=1,
        ind_rel_tol=1e-5,  ind_abs_tol=1,
        max_iter=2000
    )
    df_result, convergence = ipu_solver.fit(problem)

    df_households = []

    # Integerize by group using Bernoulli rounding of the remainder
    for i, group_control in enumerate(problem.group_controls):
        group_filter = group_control[1]

        # Household-level rows for this group
        df_hh_group = (
            df_result[group_filter][["household_id", "statpop_household_id", "expansion_factor"]]
            .drop_duplicates("household_id")
        )

        if df_hh_group.empty:
            continue

        # 1) Split weight into integer part and remainder
        weights   = df_hh_group["expansion_factor"].to_numpy()
        counts    = np.floor(weights).astype(int)          # integer part
        remainders= weights - counts                       # in [0, 1)

        # 2) Replicate integer part
        idx_rep   = np.repeat(df_hh_group.index.to_numpy(), counts)
        df_rep    = df_hh_group.loc[idx_rep]

        # 3) Bernoulli add-one for the decimal part (independent per household)
        #    Each household gets one extra copy with probability equal to its remainder.
        u         = rng.rand(len(df_hh_group))
        add_one   = (u < remainders).astype(int)
        idx_add   = np.repeat(df_hh_group.index.to_numpy(), add_one)
        df_add    = df_hh_group.loc[idx_add]

        # Combine for this group
        df_out = pd.concat([df_rep, df_add], ignore_index=True)
        df_households.append(df_out)

    # Stack groups, drop fractional weight, and assign new sequential household ids
    df_households = pd.concat(df_households, ignore_index=True).drop(columns=["expansion_factor"])
    # Keep original id for later joins if useful
    df_households = df_households.rename(columns={"household_id": "orig_household_id"})
    df_households["household_id"] = np.arange(1, df_households.shape[0] + 1)

    context.progress.update()
    return df_households, convergence