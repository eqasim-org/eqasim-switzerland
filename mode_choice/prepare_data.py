import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults

logger = logging.getLogger(__name__)
SHORT_DISTANCE_LIMIT_KM = Defaults.SHORT_DISTANCE_LIMIT_KM
LONG_DISTANCE_LIMIT_KM = Defaults.LONG_DISTANCE_LIMIT_KM

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("mode_choice.tours.build")

    context.stage("mode_choice.variables.bike")
    context.stage("mode_choice.variables.car")
    context.stage("mode_choice.variables.pt")
    context.stage("mode_choice.variables.car_passenger")
    context.stage("mode_choice.variables.walk")

    context.stage("mode_choice.cost.car")
    context.stage("mode_choice.cost.parking")
    context.stage("mode_choice.cost.pt")
    context.stage("mode_choice.penalties.parking_search")
    
    context.stage("mode_choice.dmc.run_dmc")

    context.config("data_path")
    context.config("random_seed")

def execute(context):
    logger.info("="*40)
    logger.info("Preparing data...")
    
    # lead persons and convert to polars DataFrame
    logger.info("\t Loading persons...")
    persons = context.stage("mode_choice.trips.prepare_persons")[
        ["person_id","age","sex","region","driving_license","income"]]
    persons = pl.from_pandas(persons).with_columns([
        pl.col("age").cast(pl.Int8),
        pl.col("sex").cast(pl.Int8),
        pl.col("region").cast(pl.Int8),
        pl.col("driving_license").cast(pl.Int8),
        pl.col("income").cast(pl.Float32)
    ])
    
    # load tours and convert to polars DataFrame
    logger.info("\t Loading tours...")
    tours = context.stage("mode_choice.tours.build")
    tours = pl.from_pandas(tours).with_columns([
        pl.col("euclidean_distance_km").list.eval(pl.element().cast(pl.Float32))
    ])

    # load variables and merge necessary dataframes (persons attributes will be merged later in the TourUtility)
    logger.info("\t Loading modes variables...")

    # 1. bike
    bike = context.stage("mode_choice.variables.bike")
    bike = pl.from_pandas(bike)

    # 2. car
    car            = context.stage("mode_choice.variables.car")
    car_cost       = context.stage("mode_choice.cost.car")
    parking_cost   = context.stage("mode_choice.cost.parking")
    parking_search = context.stage("mode_choice.penalties.parking_search")
    car            = (car.merge(car_cost, on=["person_id","trip_id"], how="left")
                     .merge(parking_cost, on=["person_id","trip_id"], how="left")
                     .merge(parking_search, on=["person_id","trip_id"], how="left"))
    car            = pl.from_pandas(car)

    # 3. pt
    pt      = context.stage("mode_choice.variables.pt")
    pt_cost = context.stage("mode_choice.cost.pt")
    pt      = pt.merge(pt_cost, on=["person_id","trip_id"], how="left")
    pt      = pl.from_pandas(pt)

    # 4. car passenger
    car_passenger = context.stage("mode_choice.variables.car_passenger")
    car_passenger = pl.from_pandas(car_passenger)

    # 5. walk
    walk = context.stage("mode_choice.variables.walk")
    walk = pl.from_pandas(walk)

    variables = {"bike": bike, "car": car, "pt": pt, "car_passenger": car_passenger, "walk": walk }

    # cast all floats to Float32 to save memory
    for mode in variables:
        variables[mode] = variables[mode].with_columns([
            pl.col(col).cast(pl.Float32) for col in variables[mode].columns if variables[mode][col].dtype == pl.Float64
        ])
    
    # load trips
    logger.info("\t Loading trips...")
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["trip_id","preceding_purpose", "following_purpose", "euclidean_distance_km","destination_municipality"]]
    trips = pl.from_pandas(trips)
    trips = trips.with_columns([
        pl.when(pl.col("preceding_purpose") == "home").then(1).otherwise(0.).cast(pl.Int8).alias("origin_home"),
        pl.when(pl.col("euclidean_distance_km") < SHORT_DISTANCE_LIMIT_KM).then(1).otherwise(0.).cast(pl.Int8).alias("short_distance"),
        pl.when(pl.col("euclidean_distance_km") > LONG_DISTANCE_LIMIT_KM).then(1).otherwise(0.).cast(pl.Int8).alias("long_distance"),
        pl.when(pl.col("destination_municipality") == "urban").then(1).otherwise(0.).cast(pl.Int8).alias("urban_destination"),
        pl.when(pl.col("destination_municipality") == "suburban").then(1).otherwise(0.).cast(pl.Int8).alias("suburban_destination"),
        pl.when(pl.col("following_purpose") == "work").then(1).otherwise(0.).cast(pl.Int8).alias("destination_work"),
        pl.when(pl.col("following_purpose") == "other").then(1).otherwise(0.).cast(pl.Int8).alias("destination_other"),
        pl.when(pl.col("following_purpose") == "leisure").then(1).otherwise(0.).cast(pl.Int8).alias("destination_leisure"),
        pl.when(pl.col("following_purpose") == "home").then(1).otherwise(0.).cast(pl.Int8).alias("destination_home"),
        pl.col("euclidean_distance_km").cast(pl.Float32)
    ]).select([
        "trip_id", "origin_home", "short_distance", "long_distance", "urban_destination", "suburban_destination", 
        "destination_work", "destination_other", "destination_leisure", "destination_home", "euclidean_distance_km"
    ])
    
    return dict(      
        tours=tours,
        persons=persons,
        trips = trips,
        variables=variables        
    )
