"""Prepare synthetic tour features using the current DMC variable definitions.

Tabular operations use Polars. GeoPandas is confined to spatial enrichment of
unique destinations, using the same density and PT-quality helpers as estimation.
"""
import geopandas as gpd
import polars as pl
from dmc.constants import constants
from mode_choice import defaults as D
from mode_choice.model.parameters import load


def configure(context):
    context.stage("mode_choice.routing.server")
    context.stage("mode_choice.defaults")
    for stage in ("mode_choice.data.trips", "mode_choice.tours.build", "synthesis.population.spatial.locations",
                  "synthesis.population.models.subscriptions", "data.constants", "dmc.params",
                  "data.spatial.municipalities", "data.spatial.municipality_types",
                  "data.spatial.cantons", "data.statent.density", "data.statpop.density", "data.spatial.ovgk"):
        context.stage(stage)
    context.config("threads", volatile=True)


def prepare_persons(persons, c, p, cantons):
    columns = ["person_id", "age", "sex", "income_class", "household_size", "N_children_under_12",
               "N_children_under_18", "number_of_cars_class", "driving_license", "car_availability",
               "bike_availability", "ovgk", "canton_id", "home_x", "home_y", "pt_subscription"]
    df = pl.from_pandas(persons[columns]).with_columns(
        pl.col("number_of_cars_class").cast(pl.Float64),
        pl.col("driving_license").cast(pl.Int64).eq(1),
        pl.col("car_availability").cast(pl.Int64),
        pl.col("bike_availability").cast(pl.Int64),
    )
    adults = (pl.col("household_size") - pl.col("N_children_under_12")).clip(lower_bound=1)
    equivalent_size = 1 + D.OECD_ADDITIONAL_ADULT_WEIGHT * (adults - 1) + D.OECD_CHILD_WEIGHT * pl.col("N_children_under_12")
    df = df.with_columns(
        (pl.col("income_class").replace_strict(c.INCOME_CLASS_MAP, return_dtype=pl.Float64) / equivalent_size).alias("income"),
        pl.col("canton_id").replace_strict(constants.MS_REGIONS["cluster"].to_dict()).alias("ms_region"),
        ((pl.col("car_availability") == 1)
         & pl.col("driving_license") & (pl.col("age") >= D.DRIVING_AGE)).alias("car_availability"),
        (pl.col("bike_availability") == 1).alias("bike_availability"),
        (1 - pl.col("number_of_cars_class") /
         (pl.col("household_size") - pl.col("N_children_under_18")).clip(lower_bound=1)).clip(0, 1).alias("car_ownership_ratio"),
        (pl.col("number_of_cars_class") > 0).alias("has_car"),
        pl.col("ovgk").is_in(D.GOOD_PT_SERVICE).fill_null(False).alias("good_pt_service"),
        pl.col("ovgk").is_in(D.MEDIUM_PT_SERVICE).fill_null(False).alias("medium_pt_service"),
        (pl.col("age") >= D.RETIRED_AGE).alias("is_retired"), (pl.col("age") < D.JUNIOR_AGE).alias("is_junior"),
    ).with_columns(
        (pl.col("income") <= p["lowIncomeThreshold"]).alias("low_income"),
        (pl.col("income") >= p["highIncomeThreshold"]).alias("high_income"),
        (pl.col("ms_region") == 1).alias("region_1"), (pl.col("ms_region") == 2).alias("region_2"),
    )
    return df.join(pl.from_pandas(cantons[["canton_id", "canton_name_en"]]), on="canton_id", how="left", validate="m:1")


def spatial_features(context, coordinates):
    from data.statent.density import impute as statent
    from data.statpop.density import impute as statpop
    from data.spatial.ovgk import impute_parallel as ovgk

    df = coordinates.to_pandas()
    for measure in ("employees", "companies"):
        df = statent(context, df, x="destination_x", y="destination_y",
                     radius=constants.EMPLOYEES_DENSITY_RADIUS, measure=measure,
                     output_column=f"destination_{'employee' if measure == 'employees' else measure}_density")
    df = statpop(context, context.stage("data.statpop.density"), df,
                 x="destination_x", y="destination_y", radius=constants.POPULATION_DENSITY_RADIUS)
    df = df.rename(columns={"population_density": "destination_population_density"})
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.destination_x, df.destination_y), crs="EPSG:2056")
    municipalities, _ = context.stage("data.spatial.municipalities")
    municipalities = municipalities.merge(context.stage("data.spatial.municipality_types"), on="municipality_id")
    # Resolve boundary ties deterministically, retaining exactly one row per coordinate.
    df = df.sjoin_nearest(municipalities[["municipality_id", "municipality_type", "geometry"]], how="left")
    df = df.sort_values("municipality_id").loc[lambda x: ~x.index.duplicated()].sort_index()
    df = ovgk(context, df, geometry="geometry", output_column="ovgk", n_jobs=min(D.SPATIAL_MAX_THREADS, context.config("threads")))
    df = df.rename(columns={"ovgk": "destination_ovgk"})
    return pl.from_pandas(df.drop(columns=["geometry", "index_right"])).rename({"municipality_type": "destination_municipality"})


def execute(context):
    from mode_choice.routing.server import ensure_started
    from mode_choice.routing.RoutingServer import RoutingServer
    ensure_started(context)
    try:
        return _execute(context)
    except BaseException:
        RoutingServer.stop_server()
        raise


def _execute(context):
    p = load(context)
    trips = (context.stage("mode_choice.data.trips")
             .select("trip_id", "person_id", "trip_index", "departure_time", "preceding_purpose", "following_purpose")
             .join(context.stage("mode_choice.tours.build"), on="trip_id", how="left", validate="1:1")
             .filter(pl.col("mode_choice_eligible")).drop("mode_choice_eligible"))
    if trips.is_empty():
        return trips
    locations = context.stage("synthesis.population.spatial.locations")
    coordinates = pl.DataFrame({
        "person_id": locations.person_id.to_numpy(), "activity_index": locations.activity_index.to_numpy(),
        "x": locations.geometry.x.to_numpy(), "y": locations.geometry.y.to_numpy(),
    })
    for prefix, offset in (("origin", 0), ("destination", 1)):
        loc = coordinates.with_columns((pl.col("activity_index") - offset).alias("trip_index"))
        trips = trips.join(loc.select("person_id", "trip_index", pl.col("x").alias(f"{prefix}_x"), pl.col("y").alias(f"{prefix}_y")),
                           on=["person_id", "trip_index"], how="left", validate="m:1")
    required = ["origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]
    if trips.filter(~pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in required])).height:
        raise ValueError("Mode-choice trips contain missing/non-finite locations or departure times")
    persons = prepare_persons(context.stage("synthesis.population.models.subscriptions"), context.stage("data.constants"), p,
                              context.stage("data.spatial.cantons"))
    trips = trips.join(persons, on="person_id", how="left", validate="m:1")
    coords = trips.select("destination_x", "destination_y").unique(maintain_order=True)
    trips = trips.join(spatial_features(context, coords), on=["destination_x", "destination_y"], how="left", validate="m:1")
    trips = trips.with_columns(
        ((pl.col("destination_x") - pl.col("origin_x")).pow(2) +
         (pl.col("destination_y") - pl.col("origin_y")).pow(2)).sqrt().truediv(1000).alias("euclidean_distance_km"),
        pl.col("preceding_purpose").is_in(D.HOME_PURPOSES).alias("origin_home"),
        (pl.col("destination_municipality") == "urban").alias("urban_destination"),
        (pl.col("destination_municipality") == "urbancore").alias("urbancore_destination"),
        pl.col("destination_ovgk").is_in(D.GOOD_PT_SERVICE).fill_null(False).alias("destination_good_pt_service"),
        pl.col("destination_ovgk").is_in(D.MEDIUM_PT_SERVICE).fill_null(False).alias("destination_medium_pt_service"),
        *[pl.col("following_purpose").is_in(values).alias(f"destination_{purpose}") for purpose, values in D.DESTINATION_PURPOSES.items()],
        *[(pl.col("municipality_id") == number).alias(f"destination_{city}") for city, number in D.CITY_IDS.items()],
    ).with_columns(
        (pl.col("euclidean_distance_km") < p["shortDistance_km"]).alias("short_distance"),
        (pl.col("euclidean_distance_km") > p["longDistance_km"]).alias("long_distance"),
        (pl.col("euclidean_distance_km") > p["veryLongDistance_km"]).alias("very_long_distance"),
    )
    return trips.sort(["person_id", "trip_index"])
