from mode_choice.dmc.utilities.TourUtility import TourUtility
from mode_choice.dmc.selector.Selector import Selector
from mode_choice.dmc.utilities.Parameters import Parameters
from typing import Dict 
import polars as pl

class DMC:
    random_seed = 1102
    def __init__(self, 
                parameters_file: str,
                tours: pl.DataFrame,
                persons: pl.DataFrame,
                trips: pl.DataFrame,
                variables: Dict[str, pl.DataFrame],
                seed: int = random_seed
                ):
        # Set random seed for selector
        Selector.set_seed(seed)
        # Load parameters
        Parameters.from_yaml(parameters_file)
        # Init tour utility calculator
        TourUtility.init(tours = tours, 
                        persons = persons, 
                        trips = trips,
                        variables = variables)

    def run(self):
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