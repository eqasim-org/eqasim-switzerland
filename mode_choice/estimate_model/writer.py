import pandas as pd
import numpy as np
import yaml
from .constants import constants
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class writer:
    def __init__(self, context, biogeme_model, output_path):
        self.context = context
        self.model = biogeme_model
        self.output_path = output_path
        self.params = biogeme_model.getEstimatedParameters()["Value"].to_dict()

    def write(self):
        params_dict = {self.rename(k): float(v) for k, v in self.params.items()}
        # interactions reference values
        params_dict["referenceIncome"] = constants.REF_INCOME_CHF
        params_dict["referenceEuclideanDistance_km"] = constants.REF_EUCLIDEAN_DISTANCE_KM
        params_dict["travelTimeFactor"] = constants.TRAVEL_TIME_FACTOR
        # parking informations
        params_dict["parking.urbanParkingSearchDuration_min"] = self.context.config("urban_parking_search_min")
        params_dict["parking.suburbanParkingSearchDuration_min"] = self.context.config("suburban_parking_search_min")
        params_dict["parking.urbancoreParkingSearchDuration_min"] = self.context.config("urbancore_parking_search_min")
        # set the rest to 0 and 1 for exponents
        params_dict = self.set_the_rest_to_zeros(params_dict)

        # Organize parameters by mode
        modes = ["car",  "pt", "bike", "walk", "cp", "parking"]
        grouped_params = {}
        for mode in modes:
            grouped_params[mode] = {k: v for k, v in params_dict.items() if k.startswith(mode + ".")}
        # Add non-mode parameters
        grouped_params["cost"] = {k: v for k, v in params_dict.items() if not any(k.startswith(m + ".") for m in modes)}

        # Write to YAML with #mode comments
        with open(self.output_path, 'w') as f:
            for mode in modes:
                f.write(f"# {mode}\n")
                yaml.dump(grouped_params[mode], f, sort_keys=False)
            f.write("# cost\n")
            yaml.dump(grouped_params["cost"], f, sort_keys=False)

        logger.info(f"Model parameters written to {self.output_path}")

    def rename(self, name):
        """
        Rename the parameters to a more readable format.
        """
        return NAMES_CONVERSION[name]

    def set_the_rest_to_zeros(self, params):
        """
        Set all parameters that are not estimated in the model to zero.
        """
        for new_name in NAMES_CONVERSION.values():
            if new_name not in params:
                params[new_name] = 1.0 if "exponent" in new_name.lower() else 0.0
        return params

    @staticmethod
    def to_yaml(params, file_path):        
        """
        Save all parameters to a YAML file.
        """
        
        with open(file_path, 'w') as f:
            yaml.dump(params, f, sort_keys=False)
        
        logger.info(f"All DMC parameters saved to: {file_path}")

NAMES_CONVERSION = {
    # Bike
    'beta_bike_asc': 'bike.alpha_u',
    'beta_bike_travel_time_min': 'bike.betaTravelTime_u_min',
    'lambda_bike': 'bike.travelTimeExponent',
    'beta_bike_age': 'bike.betaAge_u',
    'beta_bike_sex': 'bike.betaSex_u',
    'beta_bike_region_2': 'bike.betaRegion1_u',
    'beta_bike_region_3': 'bike.betaRegion2_u',  
    'beta_bike_origin_home': 'bike.betaOriginHome_u',
    'beta_bike_short_distance': 'bike.betaShortDistance_u',
    'beta_bike_long_distance': 'bike.betaLongDistance_u',
    'beta_bike_work_destination': 'bike.betaDestinationWork_u',
    'beta_bike_urban_destination': 'bike.betaUrbanDestination_u',
    'beta_bike_urbancore_destination': 'bike.betaUrbancoreDestination_u',
    'beta_bike_rural_destination': 'bike.betaRuralDestination_u',
    'beta_bike_destination_other': 'bike.betaDestinationOther_u',
    'beta_bike_destination_leisure': 'bike.betaDestinationLeisure_u',
    'beta_bike_destination_education': 'bike.betaDestinationEducation_u',
    'beta_bike_destination_home': 'bike.betaDestinationHome_u',
    'beta_bike_working_hour': 'bike.betaWorkingHour_u',
    'beta_bike_is_retired': 'bike.betaIsRetired_u',

    # Car
    'beta_car_asc': 'car.alpha_u',
    'beta_car_travel_time_min': 'car.betaTravelTime_u_min',
    'beta_car_access_egress_time_min': 'car.betaAccessEgressTime_u_min',
    'lambda_car_travel_time': 'car.travelTimeExponent',
    'lambda_car_access_egress_time': 'car.accessEgressTimeExponent',
    'beta_car_age': 'car.betaAge_u',
    'beta_car_sex': 'car.betaSex_u',
    'beta_car_region_2': 'car.betaRegion1_u',
    'beta_car_region_3': 'car.betaRegion2_u', 
    'beta_car_origin_home': 'car.betaOriginHome_u',
    'beta_car_urban_destination': 'car.betaUrbanDestination_u',
    'beta_car_urbancore_destination': 'car.betaUrbancoreDestination_u',
    'beta_car_rural_destination': 'car.betaRuralDestination_u',
    'beta_car_work_destination': 'car.betaDestinationWork_u',
    'beta_car_short_distance': 'car.betaShortDistance_u',
    'beta_car_long_distance': 'car.betaLongDistance_u',
    'beta_car_destination_other': 'car.betaDestinationOther_u',
    'beta_car_destination_leisure': 'car.betaDestinationLeisure_u',
    'beta_car_destination_education': 'car.betaDestinationEducation_u',
    'beta_car_destination_home': 'car.betaDestinationHome_u',
    'beta_car_working_hour': 'car.betaWorkingHour_u',
    'beta_car_ownership_ratio': 'car.betaCarOwnershipRatio_u',
    'beta_car_is_retired': 'car.betaIsRetired_u',

    # Car Passenger
    'beta_car_passenger_asc': 'cp.alpha_u',
    'beta_car_passenger_travel_time_min': 'cp.betaTravelTime_u_min',
    'lambda_car_passenger_travel_time': 'cp.travelTimeExponent',
    'beta_car_passenger_age': 'cp.betaAge_u',
    'beta_car_passenger_sex': 'cp.betaSex_u',
    'beta_car_passenger_region_2': 'cp.betaRegion1_u',
    'beta_car_passenger_region_3': 'cp.betaRegion2_u',
    'beta_car_passenger_origin_home': 'cp.betaOriginHome_u',
    'beta_car_passenger_urban_destination': 'cp.betaUrbanDestination_u',
    'beta_car_passenger_urbancore_destination': 'cp.betaUrbancoreDestination_u',
    'beta_car_passenger_rural_destination': 'cp.betaRuralDestination_u',
    'beta_car_passenger_work_destination': 'cp.betaDestinationWork_u',
    'beta_car_passenger_driving_permit': 'cp.betaDrivingLicense_u',
    'beta_car_passenger_short_distance': 'cp.betaShortDistance_u',
    'beta_car_passenger_long_distance': 'cp.betaLongDistance_u',
    'beta_car_passenger_destination_other': 'cp.betaDestinationOther_u',
    'beta_car_passenger_destination_leisure': 'cp.betaDestinationLeisure_u',
    'beta_car_passenger_destination_education': 'cp.betaDestinationEducation_u',
    'beta_car_passenger_destination_home': 'cp.betaDestinationHome_u',
    'beta_car_passenger_working_hour': 'cp.betaWorkingHour_u',
    'beta_car_passenger_is_retired': 'cp.betaIsRetired_u',

    # PT
    'beta_pt_asc': 'pt.alpha_u',
    'beta_pt_in_vehicle_time_min': 'pt.betaInVehicleTime_u_min',
    'beta_pt_access_egress_time_min': 'pt.betaAccessEgressTime_u_min',
    'beta_pt_waiting_time_min': 'pt.betaWaitingTime_u_min',
    'beta_pt_transfers': 'pt.betaLineSwitch_u',
    'beta_pt_distance_km': 'pt.betaDistance_u_km',

    'lambda_pt_in_vehicle_time': 'pt.inVehicleTimeExponent',
    'lambda_pt_access_egress_time': 'pt.accessEgressTimeExponent',    
    'lambda_pt_transfers': 'pt.lineSwitchExponent', 
    'lambda_pt_waiting_time': 'pt.waitingTimeExponent',   
    'lambda_pt_distance': 'pt.distanceExponent',

    'beta_pt_age': 'pt.betaAge_u',
    'beta_pt_sex': 'pt.betaSex_u',
    'beta_pt_region_2': 'pt.betaRegion1_u',
    'beta_pt_region_3': 'pt.betaRegion2_u', 
    'beta_pt_origin_home': 'pt.betaOriginHome_u',
    'beta_pt_work_destination': 'pt.betaDestinationWork_u',
    'beta_pt_urban_destination': 'pt.betaUrbanDestination_u',
    'beta_pt_urbancore_destination': 'pt.betaUrbancoreDestination_u',
    'beta_pt_rural_destination': 'pt.betaRuralDestination_u',
    'beta_pt_short_distance': 'pt.betaShortDistance_u',
    'beta_pt_long_distance': 'pt.betaLongDistance_u',
    'beta_pt_destination_other': 'pt.betaDestinationOther_u',
    'beta_pt_destination_leisure': 'pt.betaDestinationLeisure_u',
    'beta_pt_destination_education': 'pt.betaDestinationEducation_u',
    'beta_pt_destination_home': 'pt.betaDestinationHome_u',
    'beta_pt_working_hour': 'pt.betaWorkingHour_u',
    'beta_pt_good_service': 'pt.betaGoodService_u',
    'beta_pt_medium_service': 'pt.betaMediumService_u',
    'beta_pt_is_retired': 'pt.betaIsRetired_u',
    'beta_pt_contains_rail': 'pt.betaContainsRail_u',
    'beta_pt_contains_bus': 'pt.betaContainsBus_u',

    # Walk
    'beta_walk_asc': 'walk.alpha_u',
    'beta_walk_travel_time_min': 'walk.betaTravelTime_u_min',
    'lambda_walk': 'walk.travelTimeExponent',
    'beta_walk_age': 'walk.betaAge_u',
    'beta_walk_sex': 'walk.betaSex_u',
    'beta_walk_region_2': 'walk.betaRegion1_u',
    'beta_walk_region_3': 'walk.betaRegion2_u', 
    'beta_walk_origin_home': 'walk.betaOriginHome_u',    
    'beta_walk_work_destination': 'walk.betaDestinationWork_u',
    'beta_walk_urban_destination': 'walk.betaUrbanDestination_u',
    'beta_walk_urbancore_destination': 'walk.betaUrbancoreDestination_u',
    'beta_walk_rural_destination': 'walk.betaRuralDestination_u',
    'beta_walk_short_distance': 'walk.betaShortDistance_u',
    'beta_walk_long_distance': 'walk.betaLongDistance_u',
    'beta_walk_destination_other': 'walk.betaDestinationOther_u',
    'beta_walk_destination_leisure': 'walk.betaDestinationLeisure_u',
    'beta_walk_destination_education': 'walk.betaDestinationEducation_u',
    'beta_walk_destination_home': 'walk.betaDestinationHome_u',
    'beta_walk_working_hour': 'walk.betaWorkingHour_u',
    'beta_walk_is_retired': 'walk.betaIsRetired_u',

    # Cost
    'beta_cost_CHF': 'betaCost_u_MU',
    'lambda_cost_income': 'lambdaCostIncome',
    'lambda_cost_distance': 'lambdaCostEuclideanDistance'
}