import pyproj
from types import SimpleNamespace

# TODO: Pandas is quite good at working with categorical data. Refactor everything to make use of that.
# It will not only be more readable but will also bring a speedup!

class Constants:
    def __init__(self, census = "statpop"):
        self.CH1903      = "epsg:21781"
        self.LV05        = self.CH1903
        self.CH1903_PLUS = "epsg:2056"
        self.LV95        = self.CH1903_PLUS
        self.WGS84       = "epsg:4326"

        self.MAXIMUM_HOUSEHOLD_SIZE    = 12
        self.MINIMUM_AGE_PER_HOUSEHOLD = 16
        self.ACTIVE_AGE                = 60

        self.MARITAL_STATUS_SINGLE   = 0
        self.MARITAL_STATUS_MARRIED  = 1
        self.MARITAL_STATUS_SEPARATE = 2

        self.SEPARATE_SINGLE_THRESHOLD = 45

        self.census = census

        if self.census == "statpop":
            #self.AGE_CLASS_UPPER_BOUNDS      = [6, 15, 18, 24, 30, 45, 65, 80]
            self.AGE_CLASS_UPPER_BOUNDS      = [6, 15, 18, 24, 45, 65, 80]

        elif self.census == "are_synpop":
            self.AGE_CLASS_UPPER_BOUNDS      = [6, 18, 25, 45, 65, 75]

        else:
            raise RuntimeError(f"Unknown census type: {census}")

        self.CAR_AVAILABILITY_ALWAYS    = 0
        self.CAR_AVAILABILITY_SOMETIMES = 1
        self.CAR_AVAILABILITY_NEVER     = 2

        self.SEX_MALE   = 0
        self.SEX_FEMALE = 1

        self.BIKE_AVAILABILITY_ALWAYS  = 0
        self.BIKE_AVAILABILITY_SOMETIMES = 1
        self.BIKE_AVAILABILITY_NEVER = 2

        self.MZ_AGE_THRESHOLD = 6

        self.INCOME_CLASSES = 9

        self.MAX_NUMBER_OF_CARS_CLASS = 3
        self.MAX_NUMBER_OF_BIKES_CLASS = 3
        self.POPULATION_DENSITY_RADIUS = 2.5 * 1e3

        self.BASE_SCALING_YEAR = 2023
        self.BASE_PROJECTED_YEAR = 2040

        # This dictionnary is used to convert income class to income per capita
        self.INCOME_CLASS_MAP = {0: 2000, 1: 3000, 2: 5000, 3: 7000, 4: 9000, 5: 11000,  6: 13000, 7: 15000, 8: 17000}
        
        self.LOW_INCOME_THRESHOLD = 3000
        self.HIGH_INCOME_THRESHOLD = 8000

        # Models related constants
        self.MAP_JOB_POSITIONS_MZ_TO_SURVEY = {
                                11: 11,
                                12: 12,
                                20: 20,
                                31: 11,
                                32: 12,
                                41: 31,
                                42: 32,
                                43: 33,
                                50: 40,
                                60: 50,
                                70: 60,
                            }

        # Employements
        self.EMPLOYED = 1
        self.UNEMPLOYED = 2
        self.INACTIVE = 3

        # Employement status
        self.EMPLOYEMENT_STATUS = SimpleNamespace(
            INACTIVE = 0,
            EMPLOYED = 1,
            STUDENT = 2,            
            EMPLOYED_STUDENT = 3,
        )


def configure(context):
    context.config("census")


def execute(context):
    census = context.config("census")

    constants = Constants(census = census)

    return constants




