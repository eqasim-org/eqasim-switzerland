import pyproj

# TODO: Pandas is quite good at working with categorical data. Refactor everything to make use of that.
# It will not only be more readable but will also bring a speedup!

class Constants:
    def __init__(self, census = "statpop"):
        self.CH1903 = "epsg:21781"
        self.LV05 = self.CH1903
        self.CH1903_PLUS = "epsg:2056"
        self.LV95 = self.CH1903_PLUS
        self.WGS84 = "epsg:4326"

        self.MAXIMUM_HOUSEHOLD_SIZE = 12
        self.MINIMUM_AGE_PER_HOUSEHOLD = 16
        self.ACTIVE_AGE = 60

        self.MARITAL_STATUS_SINGLE = 0
        self.MARITAL_STATUS_MARRIED = 1
        self.MARITAL_STATUS_SEPARATE = 2

        self.SEPARATE_SINGLE_THRESHOLD = 45

        if census == "statpop":
            self.AGE_CLASS_UPPER_BOUNDS = [6, 15, 18, 24, 30, 45, 65, 80]
        elif census == "are_synpop":
            self.AGE_CLASS_UPPER_BOUNDS = [6, 18, 25, 45, 65, 75]

        self.CAR_AVAILABILITY_ALWAYS = 0
        self.CAR_AVAILABILITY_SOMETIMES = 1
        self.CAR_AVAILABILITY_NEVER = 2

        self.SEX_MALE = 0
        self.SEX_FEMALE = 1

        self.BIKE_AVAILABILITY_FOR_ALL = 2
        self.BIKE_AVAILABILITY_FOR_SOME = 1
        self.BIKE_AVAILABILITY_FOR_NONE = 0

        self.MZ_AGE_THRESHOLD = 6

        self.INCOME_CLASSES = 9

        self.MAX_NUMBER_OF_CARS_CLASS = 3

        self.POPULATION_DENSITY_RADIUS = 2.5 * 1e3

        self.BASE_SCALING_YEAR = 2015
        self.BASE_PROJECTED_YEAR = 2018


def configure(context):
    context.config("census")


def execute(context):
    census = context.config("census")

    constants = Constants(census = census)

    return constants




