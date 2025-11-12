import os
import polars as pl
import logging

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("mode_choice.tours.build")

    context.stage("mode_choice.travel_times.bike")
    context.stage("mode_choice.travel_times.car")
    context.stage("mode_choice.travel_times.pt")
    context.stage("mode_choice.travel_times.car_passenger")
    context.stage("mode_choice.travel_times.walk")

    context.stage("mode_choice.cost.car")
    context.stage("mode_choice.cost.parking")
    context.stage("mode_choice.cost.pt")
    context.stage("mode_choice.penalties.parking_search")

    # context.stage("mode_choice.dmc.run_dmc")

    context.config("data_path")
    context.config("mode_choice_parameters_file", 
                   default=os.path.join(context.config("data_path"), "dmc", "parameters.yaml"))

def execute(context):
    logger.info("="*40)
    logger.info("Running mode choice model...")
    
    # lead persons and convert to polars DataFrame
    logger.info("\t Loading persons data...")
    persons = context.stage("mode_choice.trips.prepare_persons")[
        ["person_id","age","sex","region","driving_license"]]
    persons = pl.from_pandas(persons).with_columns([
        pl.col("age").cast(pl.Int8),
        pl.col("sex").cast(pl.Int8),
        pl.col("region").cast(pl.Int8),
        pl.col("driving_license").cast(pl.Int8)
    ])
    
    # load tours and convert to polars DataFrame
    logger.info("\t Loading tours data...")
    tours = context.stage("mode_choice.tours.build")
    tours = pl.from_pandas(tours).with_columns([
        pl.col("Euclidean_distance_km").list.eval(pl.element().cast(pl.Float32))
    ])

    # load variables and merge necessary dataframes (persons attributes will be merged later in the TourUtility)
    logger.info("\t Loading variables data...")
    # 1. bike
    bike = context.stage("mode_choice.travel_times.bike")
    bike = pl.from_pandas(bike)
    # 2. car
    car = context.stage("mode_choice.travel_times.car")
    car_cost = context.stage("mode_choice.cost.car")
    parking_cost = context.stage("mode_choice.cost.parking")
    parking_search = context.stage("mode_choice.penalties.parking_search")
    car = (car.merge(car_cost, on=["person_id","trip_id"], how="left")
           .merge(parking_cost, on=["person_id","trip_id"], how="left")
           .merge(parking_search, on=["person_id","trip_id"], how="left"))
    car = pl.from_pandas(car)
    # 3. pt
    pt = context.stage("mode_choice.travel_times.pt")
    pt_cost = context.stage("mode_choice.cost.pt")
    pt = pt.merge(pt_cost, on=["person_id","trip_id"], how="left")
    pt = pl.from_pandas(pt)
    # 4. car passenger
    car_passenger = context.stage("mode_choice.travel_times.car_passenger")
    car_passenger = pl.from_pandas(car_passenger)
    # 5. walk
    walk = context.stage("mode_choice.travel_times.walk")
    walk = pl.from_pandas(walk)

    variables = {"bike": bike, "car": car, "pt": pt, "car_passenger": car_passenger, "walk": walk }
    
    # cast all floats to Float32 to save memory
    for mode in variables:
        variables[mode] = variables[mode].with_columns([
            pl.col(col).cast(pl.Float32) for col in variables[mode].columns if variables[mode][col].dtype == pl.Float64
        ])
    
    # Init DMC
    # logger.info("\t Initializing DMC model...")
    # DMC = context.stage("mode_choice.dmc.run_dmc")
    # dmc = DMC(
    #     parameters_file=context.config("mode_choice_parameters_file"),
    #     tours=tours,
    #     persons=persons,
    #     variables={
    #         "bike": bike,
    #         "car": car,
    #         "pt": pt,
    #         "car_passenger": car_passenger,
    #         "walk": walk
    #     }
    # )

    # # Run DMC
    # logger.info("\t Running DMC model...")
    # choices = dmc.run()
    # return choices[["person_id","trip_id","selected_mode"]].to_pandas()