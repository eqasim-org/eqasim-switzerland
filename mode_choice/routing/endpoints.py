"""Join trips to the exact prepared facility coordinates and endpoint links."""
from numbers import Real
import polars as pl
from matsim.scenario.population import HOME_DESTINATION_ID
from matsim.config.tools import read_config


def attach(context, requests):
    prepared = context.stage("mode_choice.routing.preparation")
    facilities = pl.read_parquet(prepared["facilities"])
    locations = context.stage("synthesis.population.spatial.locations")
    persons = context.stage("synthesis.population.models.subscriptions")
    locations = locations[["person_id", "activity_index", "destination_id"]].copy()
    # PersonWriter retains string IDs and converts numeric IDs to integers.
    identifiers = {value: str(int(value)) if isinstance(value, Real) else value
                   for value in locations["destination_id"].dropna().unique()}
    locations["destination_id"] = locations["destination_id"].map(identifiers)
    activities = pl.from_pandas(locations)
    homes = pl.from_pandas(persons[["person_id", "household_id"]]).with_columns(
        pl.concat_str(pl.lit("home"), pl.col("household_id").cast(pl.Int64).cast(pl.String)).alias("home_facility"))
    activities = activities.join(homes.select("person_id", "home_facility"), on="person_id", validate="m:1").with_columns(
        pl.when(pl.col("destination_id") == str(HOME_DESTINATION_ID)).then(pl.col("home_facility"))
        .otherwise(pl.col("destination_id").cast(pl.String)).alias("facility_id"),
    ).join(facilities, on="facility_id", how="left", validate="m:1")
    for prefix, offset in (("origin", 0), ("destination", 1)):
        endpoints = activities.select("person_id", (pl.col("activity_index") - offset).alias("trip_index"),
                                     pl.col("x").alias(f"{prefix}_x"), pl.col("y").alias(f"{prefix}_y"),
                                     pl.col("link_id").alias(f"{prefix}_link_id"))
        requests = requests.drop(f"{prefix}_x", f"{prefix}_y").join(
            endpoints, on=["person_id", "trip_index"], how="left", validate="m:1", maintain_order="left")
    if requests.select(pl.any_horizontal(pl.col("origin_link_id", "destination_link_id").is_null()).any()).item():
        raise ValueError("Population route reuse requires a prepared facility for every trip endpoint")
    # PopulationWriter serializes activity end times to whole seconds.
    return requests.with_columns(pl.col("departure_time").cast(pl.Int64).cast(pl.Float64))


def compatible_requests(context, source, target, mode):
    """Omit forbidden network endpoints; missing results follow normal fallback.

    Java rejects endpoint links outside the requested mode's network. Skipping
    these rows keeps one bad endpoint from aborting a complete mixed request.
    Teleported modes need no allowed-link check.
    """
    parameter = read_config(context.stage("mode_choice.routing.config")).find(
        "module[@name='routing']/param[@name='networkModes']")
    network_modes = {value.strip() for value in parameter.get("value").split(",")} if parameter is not None else {"car", "bike"}
    if mode is not None and mode not in network_modes | {"car", "truck"}:
        return source
    links = pl.scan_parquet(context.stage("mode_choice.routing.preparation")["links"])
    rows = pl.scan_csv(source, schema_overrides={"person_id": pl.String, "identifier": pl.String,
                                                "origin_link_id": pl.String, "destination_link_id": pl.String})
    road_mode = pl.lit(mode) if mode is not None else pl.col("mode")
    for prefix in ("origin", "destination"):
        allowed = links.select(pl.col("link_id").alias(f"{prefix}_link_id"), "allowed_modes")
        rows = rows.join(allowed, on=f"{prefix}_link_id", how="left", validate="m:1", maintain_order="left").filter(
            pl.col("allowed_modes").list.contains(road_mode).fill_null(False)).drop("allowed_modes")
    rows.sink_csv(target)
    return str(target)
