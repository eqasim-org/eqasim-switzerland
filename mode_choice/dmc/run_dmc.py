from mode_choice.dmc.utilities.TourUtility import TourUtility
from mode_choice.dmc.selector.Selector import Selector
from mode_choice.dmc.utilities.Parameters import Parameters
from typing import Dict, Union 
import polars as pl
import logging
import numpy as np


logger = logging.getLogger("synpp: DMC")

class DMC:
    random_seed = 1102
    def __init__(self, 
                parameters_file: str,
                tours: Union[list,pl.DataFrame], # can be a list of paths to tour files or a polars DataFrame
                persons: pl.DataFrame,
                trips: pl.DataFrame,
                variables: Dict[str, pl.DataFrame],
                seed: int = random_seed
                ):
        # Set random seed for selector
        self.seed = seed
        Selector.set_seed(seed)
        # Load parameters
        Parameters.from_yaml(parameters_file)        
        # Init tour utility calculator if the tours are provided as DataFrame
        if not isinstance(tours, list):       
            self.init_tours(tours, persons, trips, variables)
            self.run_by_batch = False
        else:
            # If tours are provided as list of Tour paths, save these, to init multiple times later
            self.tours = tours
            self.persons = persons
            self.trips = trips
            self.variables = variables
            self.run_by_batch = True

    def init_tours(self, tours: Union[list, pl.DataFrame], persons: pl.DataFrame, 
                   trips: pl.DataFrame, variables: Dict[str, pl.DataFrame]):
        # clear it if it was initialized before
        TourUtility.clear()

        # if tours is a path, filter persons, trips, variables to only those needed
        if isinstance(tours, str):
            tours = pl.read_parquet(tours)
            unique_persons = tours.select("person_id").unique().to_series().to_list()
            unique_trips = tours.select('trip_id').explode('trip_id').unique().to_series().to_list()
            
            df_persons = persons.lazy().filter(pl.col("person_id").is_in(unique_persons))
            df_trips = trips.lazy().filter(pl.col("trip_id").is_in(unique_trips))
            df_variables = {k: v.lazy().filter(pl.col("person_id").is_in(unique_persons)) 
                            for k, v in variables.items()}
            # initialize it
            TourUtility.init(tours = tours, 
                             persons = df_persons, 
                             trips = df_trips,
                             variables = df_variables)                 
                            
        else:
            TourUtility.init(tours = tours, 
                            persons = persons, 
                            trips = trips,
                            variables = variables)

    def run(self, verbose: bool = True) -> pl.DataFrame:
        if self.run_by_batch:
            if verbose:
                logger.info("\t Running DMC model in batch mode...")
            
            all_choices = []
            for i, tour_path in enumerate(self.tours):
                if verbose:
                    logger.info(f"\t\t DMC by batch: {i+1} / {len(self.tours)}")
                    
                self.init_tours(tour_path, self.persons, self.trips, self.variables)
                tours = TourUtility.get_all_utilities().collect()
                tours = Selector.select(tours)
                all_choices.append(tours)
                Selector.set_seed(self.seed + (i + 1)*1000)  # change seed for next batch (important, to change the Gumbel noise)
            
            return pl.concat(all_choices)
        
        else:
            tours = TourUtility.get_all_utilities().collect()
            tours = Selector.select(tours)
            return tours
        
def configure(context):
    context.config("random_seed")
    context.stage("mode_choice.dmc_defaults")

def execute(context):
    random_seed = context.config("random_seed")
    DMC.random_seed = random_seed
    return DMC