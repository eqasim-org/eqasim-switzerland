import numpy as np
import pandas as pd

from data.statpop.multilevelipf import multilevelipf
from data.statpop.multilevelipf.multilevelipf import FittingProblem, IPUSolver
from data.utils import coerce_boolean_series
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


def _build_problem(df, household_controls, population_controls):
    """Fit private households and all retained residents to their own margins."""
    df = multilevelipf.add_expansion_factor_column(df.copy())
    # Older private-only inputs have no recovery flag. A technical singleton
    # represents a recovered resident, not a private household in BFS statistics.
    recovered = df.get("is_synthetic_household", pd.Series(False, index=df.index))
    df["is_synthetic_household"] = coerce_boolean_series(recovered, name="is_synthetic_household")
    if df.groupby("household_id")["is_synthetic_household"].nunique().gt(1).any():
        raise ValueError("A household cannot mix private and recovered classifications")
    group_controls = [household_controls.assign(is_synthetic_household=False)]
    group_filters = multilevelipf.compute_group_filters(df, group_controls)
    individual_filters = multilevelipf.compute_individual_filters(df, "household_id", [population_controls])
    return FittingProblem(df, group_filters, "household_id", individual_filters, "person_id")

def configure(context):
    context.stage("synthesis.population.models.caravailability")
    context.stage("data.statpop.projections.households")
    context.stage("data.statpop.projections.population")
    context.stage("data.constants")

    context.config("enable_scaling", default=False)
    context.config("scaling_year", default=2050)
    
    context.config("random_seed")
    context.config("threads", volatile=True)    
    

def execute(context):
    df_statpop = context.stage("synthesis.population.models.caravailability").copy()
    
    if context.config("enable_scaling"):

        scaling_year = context.config("scaling_year")

        logger.info("Scaling STATPOP to 31 December %d using IPU (2025 reference scenarios).", scaling_year)

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
            problem = _build_problem(df,
                df_household_controls[df_household_controls["canton_id"] == canton_id],
                df_population_controls[df_population_controls["canton_id"] == canton_id])
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

        # Worker completion order must not change IDs or the downstream sample.
        df_households = df_households.sort_values("statpop_household_id", kind="stable").reset_index(drop=True)
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
        recovered = coerce_boolean_series(df_statpop.get(
            "is_synthetic_household", pd.Series(False, index=df_statpop.index)), name="is_synthetic_household")
        logger.info("Private households after scaling : %d (target %d)",
                    df_statpop.loc[~recovered, "household_id"].nunique(), df_household_controls["weight"].sum())
        logger.info("Recovered residents after scaling : %d; technical households excluded from the private margin.",
                    recovered.sum())

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

    # Integerize every fitted household once, including recovered singletons
    # which have person controls but deliberately no private-household margin.
    # Iterating group-control filters here would silently discard those residents.
    df_hh = df_result[["household_id", "statpop_household_id", "expansion_factor"]].drop_duplicates("household_id")
    weights = df_hh["expansion_factor"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("IPU household expansion factors must be finite and non-negative")
    counts = np.floor(weights).astype(int)
    remainders = weights - counts
    df_rep = df_hh.iloc[np.repeat(np.arange(len(df_hh)), counts)]
    df_add = df_hh.loc[rng.rand(len(df_hh)) < remainders]
    df_households = pd.concat([df_rep, df_add], ignore_index=True).drop(columns="expansion_factor")
    # Keep original id for later joins if useful
    df_households = df_households.rename(columns={"household_id": "orig_household_id"})
    df_households["household_id"] = np.arange(1, df_households.shape[0] + 1)

    context.progress.update()
    return df_households, convergence
