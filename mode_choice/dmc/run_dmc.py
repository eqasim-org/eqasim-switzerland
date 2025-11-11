from .utilities.TourUtility import TourUtility
from .utilities.BaseUtility import BaseUtility
from .selector.Selector import Selector
from .utilities.Parameters import Parameters
import pandas as pd
from typing import Dict 
import polars as pl

class DMC:
    def __init__(self, 
                 parameters_file: str,
                 tours: pl.DataFrame,
                 persons: pl.DataFrame,
                 travel_times: Dict[str, pl.DataFrame]
                 ):
        # Load parameters
        Parameters.from_yaml(parameters_file)
        # Init tour utility calculator
        TourUtility.init(tours, persons, travel_times)

    def run(self):
        tours = TourUtility.get_all_utilities().collect()
        tours = Selector.select(tours)
        return tours
    

def config(context):
    pass

def execute(context):
    return DMC