"""Write compact tour inputs in complete-person batches for repeated selection."""
from pathlib import Path
import polars as pl
from mode_choice.defaults import RUN_DEFAULTS, TOUR_BATCH_FILENAME, TOUR_BATCH_COLUMNS, MODES
from dmc.parameters import Parameters
from mode_choice.model.utilities import expressions
from mode_choice.tours.batching import person_batches


def configure(context):
    context.stage("mode_choice.data.inputs")
    context.stage("mode_choice.defaults")
    context.stage("dmc.parameter_files")
    for name in ("mode_choice_person_batch_size", "distance_cost_interaction", "income_cost_interaction", "use_income_in_dmc"):
        context.config(name, RUN_DEFAULTS[name])


def execute(context):
    trips = context.stage("mode_choice.data.inputs").sort("person_id", "trip_index")
    if trips.is_empty():
        return []
    utility_expressions = expressions(Parameters.from_context(context), distance_interaction=context.config("distance_cost_interaction"),
        income_interaction=context.config("income_cost_interaction") and context.config("use_income_in_dmc"))
    paths = []
    for i, batch in enumerate(person_batches(trips, context.config("mode_choice_person_batch_size"))):
        batch = batch.lazy().with_columns(utility_expressions).collect()
        for mode in MODES:
            if batch.filter(pl.col(f"{mode}_availability") & ~pl.col(f"utility_{mode}").is_finite().fill_null(False)).height:
                raise ValueError(f"Non-finite {mode} utilities in tour batch {i}")
        path = Path(context.path()) / TOUR_BATCH_FILENAME.format(i=i)
        batch.select(TOUR_BATCH_COLUMNS).write_parquet(path)
        paths.append(str(path))
    return paths


def validate(context):
    """Regenerate a cached batch list if its Parquet files are removed or edited."""
    paths = sorted(Path(context.path()).glob(TOUR_BATCH_FILENAME.format(i="*")))
    return [[path.name, path.stat().st_mtime_ns, path.stat().st_size] for path in paths]
