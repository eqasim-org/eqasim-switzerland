"""Complete partial router outputs and mark unroutable trips consistently."""
import logging
import polars as pl
from mode_choice.defaults import UNROUTABLE_TIME_MIN


def complete_routes(trips, routes, time_columns):
    # Missing results mean no route. Unexpected or ambiguous identifiers are errors.
    if routes["trip_id"].null_count() or routes["trip_id"].n_unique() != routes.height:
        raise ValueError("Null or duplicate routing identifiers")
    identifiers = trips.select("trip_id")
    if routes.select("trip_id").join(identifiers, on="trip_id", how="anti").height:
        raise ValueError("Router returned unknown trip identifiers")
    result = identifiers.join(routes, on="trip_id", how="left", validate="1:1", maintain_order="left")

    # A missing, negative or non-finite route component invalidates the whole route.
    columns = [column for column in routes.columns if column != "trip_id"]
    valid = pl.all_horizontal([
        pl.col(column).is_finite() & (pl.col(column) >= 0) for column in columns
    ]).fill_null(False)
    valid = result.select(valid.alias("valid")).to_series()
    failed = (~valid).sum()
    if not failed:
        return result
    logging.getLogger(__name__).warning(
        "%s of %s routes unavailable (%s)", failed, result.height, ", ".join(time_columns))

    # Keep downstream costs and utility arithmetic finite. Zero placeholders are
    # never used for selection: the high time makes the route unavailable.
    return result.with_columns([
        pl.when(valid).then(pl.col(column)).otherwise(
            UNROUTABLE_TIME_MIN if column in time_columns else 0.0
        ).alias(column) for column in columns
    ])
