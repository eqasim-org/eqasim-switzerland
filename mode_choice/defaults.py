"""Single source for population mode-choice defaults and policy constants.

Fitted coefficients/scales stay in the DMC parameter files; shared estimation
constants stay in dmc.constants. Override runtime defaults in config_dib_10.yml.
"""
from dmc.constants import constants as DMC

MODES = ("car", "pt", "bike", "walk", "car_passenger")
VEHICLE_MODES = {MODES.index("car"): 0, MODES.index("bike"): 1}
SAMPLING_METHODS = ("probability", "gumbel")

PERSON_BATCH_SIZE = 10_000
ROUTING_BATCH_SIZE = 2_048
SAMPLING_METHOD = "probability"
MAX_TRIPS_PER_SEGMENT = 8
CACHE_MAX_ENTRIES = 1_024
CACHE_MAX_EDGES = 200_000
PARAMETER_CACHE_SIZE = 8
RANDOM_CACHE_MAX_VALUES = 2_000_000

RUN_DEFAULTS = {
    "car_cost_model": DMC.CAR_COST_MODEL,
    "walk_speed_m_per_s": 1.3,
    "walk_distance_factor": 1.3,
    "bike_speed_m_per_s": 4.0,
    "bike_distance_factor": 1.4,
    "distance_cost_interaction": True,
    "income_cost_interaction": True,
    "use_income_in_dmc": True,
    "mode_choice_person_batch_size": PERSON_BATCH_SIZE,
    "mode_choice_sampling_method": SAMPLING_METHOD,
    "mode_choice_max_trips_per_tour": MAX_TRIPS_PER_SEGMENT,
    "mode_choice_cache_max_entries": CACHE_MAX_ENTRIES,
    "mode_choice_cache_max_edges": CACHE_MAX_EDGES,
    "mode_choice_random_cache_max_values": RANDOM_CACHE_MAX_VALUES,
    "mode_choice_cache_prepared_batches": True,
}
# Population ASC calibration; independent of MATSim's own calibration.
CALIBRATION_DEFAULTS = {
    "mode_choice_calibrate_alphas": False,
    "mode_choice_calibration_level": "global",
    "mode_choice_calibration_max_evaluations": 50,
    "mode_choice_calibration_tolerance": 1e-4,
}
CALIBRATION_LEVELS = ("global", "canton")
CALIBRATION_CANTON_COLUMN = "canton_name_en"
CANTON_ALPHA_KEYS = {mode: f"swissCanton.{'cp' if mode == 'car_passenger' else mode}" for mode in MODES}
CALIBRATION_REFERENCE_MODE = "pt"
CALIBRATION_REFERENCE_INDEX = MODES.index(CALIBRATION_REFERENCE_MODE)
CALIBRATION_SHARE_FLOOR = 1e-6
CALIBRATION_MAX_INERTIA = 0.8
CALIBRATION_DAMPING_RATE = 0.5
CALIBRATION_MIN_DISTANCE_KM = 1e-3
ALPHA_KEYS = {mode: f"{'cp' if mode == 'car_passenger' else mode}.alpha_u" for mode in MODES}
CALIBRATION_COLUMNS = (
    "person_id", "trip_index", "tour_index", "preceding_purpose",
    "origin_x", "origin_y", "destination_x", "destination_y",
    "euclidean_distance_km", *(f"utility_{mode}" for mode in MODES),
)
CAR_ROUTING_BACKENDS = ("java", "pandana")
ROUTING_DEFAULTS = {
    "mode_choice_routing_batch_size": ROUTING_BATCH_SIZE,
    "mode_choice_car_routing_backend": "java",
    "mode_choice_car_linkstats_path": "",
    "mode_choice_car_events_path": "",
}
HOURS_PER_DAY = 24
SECONDS_PER_HOUR = 3600
LINKSTATS_MIN_SPEED_M_PER_S = 1 / 3.6
LINKSTATS_TIME_COLUMNS = tuple(f"TRAVELTIME{hour}-{hour + 1}avg" for hour in range(HOURS_PER_DAY))
SPATIAL_MAX_THREADS = 8
DISTANCE_COST_INTERACTION = RUN_DEFAULTS["distance_cost_interaction"]
INCOME_COST_INTERACTION = RUN_DEFAULTS["income_cost_interaction"]

HOME_PURPOSES = ("home", "home_secondary")
TOUR_HOME_PURPOSE = "home"
WORK_PURPOSES = ("work", "work_secondary")
DESTINATION_PURPOSES = {
    "home": HOME_PURPOSES, "work": WORK_PURPOSES,
    "education": ("education", "education_secondary"),
    "shopping": ("shop", "shopping"), "leisure": ("leisure",), "other": ("other",),
}
CITY_IDS = {"zurich": 261, "bern": 351, "luzern": 1061, "basel": 2701, "lausanne": 5586, "geneva": 6621}
GOOD_PT_SERVICE = ("A", "B")
MEDIUM_PT_SERVICE = ("C", "D")
PT_GA = (1,)
PT_HALBTAX = (3, 4)
PT_VERBUND = (2, 4)

DRIVING_AGE = 18
RETIRED_AGE = 65
JUNIOR_AGE = 16
ADULT_AGE_REFERENCE = 17
CP_DISTANCE_REFERENCE_KM = 50.0
POWER_FLOOR = 1e-300
OECD_ADDITIONAL_ADULT_WEIGHT = 0.5
OECD_CHILD_WEIGHT = 0.3

MIN_MOTORIZED_TOUR_TIME_MIN = 0.5
MAX_ROUTED_TIME_MIN = 300.0
MIN_MOTORIZED_TOUR_DISTANCE_KM = 0.05
MAX_PT_TRANSFERS = 6
MAX_PT_TRANSFER_TIME_MIN = 60.0
MAX_BIKE_DISTANCE_KM = 12.0
PARKING_START_S = 8 * 3600
PARKING_END_S = 19 * 3600
PARKING_MAX_DURATION_MIN = (PARKING_END_S - PARKING_START_S) / 60
PARKING_FREE_HOURS = 1.0
PARKING_MAX_COST_CHF = 40.0


def execute(context):
    """Synpp dependency: changing defaults invalidates mode-choice stages."""
    return {**RUN_DEFAULTS, **CALIBRATION_DEFAULTS}

ROUTER_SERVER_CLASS = "org.eqasim.switzerland.ch_cmdp.utils.routing.MultiModeRouterServer"
ROUTER_SERVER_MODES = ("car", "pt", "walk", "bike")
SERVER_DEFAULTS = {
    "mode_choice_server_startup_timeout": 1800.0,
    "mode_choice_server_request_timeout": 7200.0,
    "mode_choice_server_shutdown_timeout": 30.0,
    "mode_choice_server_poll_interval": 0.5,
    "mode_choice_server_chunk_size": 65536,
}
SERVER_HEALTH_TIMEOUT = 2.0

SERVER_LOG_FILENAME_FORMAT = "server_logs_starting_at_%Y-%m-%d_%H-%M-%S_%f_UTC.logs"
ROUTING_COST_FILENAME = "cost.parquet"
# Files read by MultiModeRouter, relative to the original MATSim config.
ROUTER_CONFIG_INPUTS = {
    "transit": ("transitScheduleFile", "vehiclesFile"),
    "ptZones": ("ptZonesFilePath", "sbbDistancesPath", "pricingDescriptionPath"),
}

# Stage output contracts.
SOURCE_TRIP_ID = "source_trip_id"
TOUR_BATCH_FILENAME = "tours_batch_num{i}.parquet"
ROUTING_VARIABLE_STAGES = tuple(f"mode_choice.routing_variables.{mode}" for mode in MODES)
COST_STAGES = ("mode_choice.costs.car", "mode_choice.costs.pt")
AVAILABILITY_COLUMNS = tuple(f"{mode}_availability" for mode in MODES)
PERSON_AVAILABILITY_COLUMNS = ("car_availability", "bike_availability")
TOUR_BATCH_COLUMNS = ("trip_id", *CALIBRATION_COLUMNS, CALIBRATION_CANTON_COLUMN)
PT_ROUTING_COLUMNS = (
    "pt_total_travel_time_min", "pt_total_distance_km", "pt_in_vehicle_time_min", "pt_in_vehicle_distance_km",
    "pt_access_egress_time_min", "pt_transfer_time_min", "pt_transfers",
)
