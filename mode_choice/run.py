
import os
import polars as pl

def config(context):
    context.stage("mode_choice.prepare_trips")
    context.stage("mode_choice.tours.build")
    context.stage("mode_choice.travel_times.bike")
    context.stage("mode_choice.travel_times.car")
    context.stage("mode_choice.travel_times.pt")
    context.stage("mode_choice.travel_times.car_passenger")
    context.stage("mode_choice.travel_times.walk")
    context.stage("mode_choice.dmc.run_dmc")

    context.config("data_path")
    context.config("mode_choice_parameters", 
                   default=os.path.join(context.config("data_path"), "dmc", "parameters.yaml"))

def execute(context):
    persons = context.stage("synthesis.population.enriched")[
        ["person_id","age","sex","driving_license"]].as_type({
            "driving_license": int
    })
    
    tours = context.stage("mode_choice.tours.build")
    bike_tt = context.stage("mode_choice.travel_times.bike")
    car_tt = context.stage("mode_choice.travel_times.car")
    pt_tt = context.stage("mode_choice.travel_times.pt")
    car_passenger_tt = context.stage("mode_choice.travel_times.car_passenger")
    walk_tt = context.stage("mode_choice.travel_times.walk")

    DMC = context.stage("mode_choice.dmc.run_dmc")
    dmc = DMC(
        parameters_file=context.config("mode_choice_parameters"),
        tours=pl.from_pandas(tours),
        persons=pl.from_pandas(persons),
        travel_times={
            "bike": pl.from_pandas(bike_tt),
            "car": pl.from_pandas(car_tt),
            "pt": pl.from_pandas(pt_tt),
            "car_passenger": pl.from_pandas(car_passenger_tt),
            "walk": pl.from_pandas(walk_tt)
        }
    )

    choices = dmc.run()
    return choices[["person_id","trip_index","selected_mode"]].to_pandas()