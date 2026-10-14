"""Strict Polars joins for the trip-keyed stage contracts."""
import polars as pl


def join_variables(trips, variables):
    if "trip_id" not in variables.columns or variables["trip_id"].null_count():
        raise ValueError("Stage output requires non-null trip_id values")
    if variables["trip_id"].n_unique() != variables.height:
        raise ValueError("Duplicate trip_id values in stage output")
    if variables.height != trips.height or variables.select("trip_id").join(
            trips.select("trip_id"), on="trip_id", how="anti").height:
        raise ValueError("Stage trip_id values do not match the input trips")
    overlap = (set(trips.columns) & set(variables.columns)) - {"trip_id"}
    if overlap:
        raise ValueError(f"Stage output would overwrite columns: {sorted(overlap)}")
    return trips.join(variables, on="trip_id", how="left", validate="1:1", maintain_order="left")
