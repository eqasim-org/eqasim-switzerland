import numpy as np
import pandas as pd

from data.statpop.multilevelipf import multilevelipf
from data.statpop.multilevelipf.multilevelipf import FittingProblem, IPUSolver
import logging

logger = logging.getLogger("synpp")

IPU_KEY_DTYPES = {
    "canton_id": "int64",
    "sex": "int64",
    "nationality": "int64",
    "age_class": "int64",
}


def _normalize_ipu_keys(df, dataset_name):
    missing = [column for column in IPU_KEY_DTYPES if column not in df.columns]
    if missing:
        raise KeyError(f"{dataset_name} is missing IPU key columns: {missing}")

    result = df.copy()
    for column, dtype in IPU_KEY_DTYPES.items():
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{dataset_name}.{column} cannot be converted to {dtype}"
            ) from error
    return result

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
    df_statpop = context.stage("synthesis.population.models.caravailability").copy()
    
    if context.config("enable_scaling"):

        scaling_year = context.config("scaling_year")

        logger.info("Scaling STATPOP to year %d using IPU.", scaling_year)

        processes = context.config("threads")
        df_household_controls, hh_year = context.stage("data.statpop.projections.households")
        df_population_controls, pop_year = context.stage("data.statpop.projections.population")

        df_statpop = _normalize_ipu_keys(df_statpop, "STATPOP")
        df_population_controls = _normalize_ipu_keys(
            df_population_controls, "population controls")
        df_household_controls = df_household_controls.copy()
        df_household_controls["canton_id"] = pd.to_numeric(
            df_household_controls["canton_id"], errors="raise").astype("int64")

        population_control_keys = list(IPU_KEY_DTYPES)
        if df_population_controls.duplicated(population_control_keys).any():
            raise ValueError("Population controls contain duplicate IPU cells")
        if df_household_controls["canton_id"].duplicated().any():
            raise ValueError("Household controls contain duplicate canton rows")
        for controls, name in [
            (df_population_controls, "population controls"),
            (df_household_controls, "household controls"),
        ]:
            controls["weight"] = pd.to_numeric(controls["weight"], errors="raise")
            if controls["weight"].isna().any() or not np.isfinite(controls["weight"]).all():
                raise ValueError(f"{name} contain non-finite weights")
        if (df_population_controls["weight"] < 0).any() or (df_household_controls["weight"] < 0).any():
            raise ValueError("IPU control weights must be non-negative")

        if hh_year != scaling_year or pop_year != scaling_year:
            raise ValueError(
                f"Projection years do not match scaling_year={scaling_year}: "
                f"households={hh_year}, population={pop_year}"
            )

        logger.info("Number of households in household controls : %d", df_household_controls["weight"].sum())
        logger.info("Number of persons in population controls : %d", df_population_controls["weight"].sum())
        logger.info("Number of households before scaling : %d", len(df_statpop["household_id"].unique()))
        logger.info("Number of persons before scaling : %d", len(df_statpop["person_id"].unique()))

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

        logger.info("Constructed %d IPU fitting problems.", len(problems))
        logger.info("Starting IPU.")

        # Run IPU algorithm in parallel
        with context.progress(label="Performing IPU on STATPOP by canton...", total=len(problems)):
            with context.parallel(processes=processes) as parallel:
                df_households, convergence = [], []

                for df_household_item, convergence_item in parallel.imap_unordered(process, problems):
                    df_households.append(df_household_item)
                    convergence.append(convergence_item)

        df_households = pd.concat(df_households)
        logger.info("Convergence rate: %f", np.round(np.mean(convergence), 3))
        if not all(convergence):
            failed = len(convergence) - sum(convergence)
            raise RuntimeError(
                f"IPU did not converge for {failed} of {len(convergence)} canton problems"
            )

        # Generate new unique ids
        logger.info("Generating new household ids.")
        df_households["household_id_new"] = np.arange(df_households.shape[0]) + 1
        del df_statpop["household_id"]

        # Merge the new household ids onto statpop by statpop_household_id (i.e. original id)
        df_statpop = pd.merge(df_statpop, df_households, on="statpop_household_id")
        df_statpop["household_id"] = df_statpop["household_id_new"]
        del df_statpop["household_id_new"]

        # sort by household id and generate new person ids
        logger.info("Generating new person ids.")
        df_statpop = df_statpop.sort_values(by=["household_id", "person_id"])
        df_statpop["person_id"] = np.arange(df_statpop.shape[0]) + 1

        # remove unneeded columns
        logger.info("Number of households in household controls : %d", df_household_controls["weight"].sum())
        logger.info("Number of persons in population controls : %d", df_population_controls["weight"].sum())
        logger.info("Number of households after scaling : %d", len(df_statpop["household_id"].unique()))
        logger.info("Number of persons after scaling : %d", len(df_statpop["person_id"].unique()))

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
