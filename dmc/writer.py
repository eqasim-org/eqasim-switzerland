import os
import pandas as pd
import numpy as np
import yaml
from dmc.constants import constants
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class writer:
    def __init__(self, context, biogeme_model, 
                 mode_parameters_file = "model_parameters.yaml",
                 cost_parameters_file = "cost_parameters.yaml"):
        self.context = context
        self.model = biogeme_model
        self.mode_output_path = os.path.join(context.path(), mode_parameters_file)
        self.cost_output_path = os.path.join(context.path(), cost_parameters_file)
        self.params = biogeme_model.getEstimatedParameters()["Value"].to_dict()

    def write(self):
        self.write_mode_parameters()
        self.write_cost_parameters()
        return (self.mode_output_path, self.cost_output_path)
    
    def write_cost_parameters(self):
        params_dict = dict()
        
        # parking cost parameters
        params_dict["urbanParkingCost_CHF_h"] = self.context.config("parking_cost_per_hour_CHF_urban")
        params_dict["urbancoreParkingCost_CHF_h"] = self.context.config("parking_cost_per_hour_CHF_urbancore")
        params_dict["suburbanParkingCost_CHF_h"] = self.context.config("parking_cost_per_hour_CHF_suburban")
        params_dict["parkingPriceReductionForWork"] = self.context.config("parking_price_reduction_for_work")
        
        # car cost parameters
        params_dict["carCost_CHF_km"] = self.context.config("car_cost_per_km") #CHF per km        
        
        # public transport cost parameters
        params_dict["ptRegionalRadius_km"] = self.context.config("pt_regional_radius_km") # km

        # Write to YAML with 
        with open(self.cost_output_path, 'w') as f:
            yaml.dump(params_dict, f, sort_keys=False)

        logger.info(f"Cost parameters written to {self.cost_output_path}")

    def write_mode_parameters(self):
        params_dict = {self.rename(k): float(v) for k, v in self.params.items()}
        
        # interactions reference values
        params_dict["referenceIncome"] = self.context.config("reference_income_chf")
        params_dict["referenceEuclideanDistance_km"] = self.context.config("reference_euclidean_distance_km")
        
        # scale parameters
        params_dict["timeScale_min"] = constants.TIME_SCALE_MIN
        params_dict["distanceScale_km"] = constants.DISTANCE_SCALE_KM

        # parking informations
        params_dict["parking.urbancoreParkingSearchDuration_min"] = self.context.config("urbancore_parking_search_min")
        params_dict["parking.urbanParkingSearchDuration_min"] = self.context.config("urban_parking_search_min")        
        params_dict["parking.suburbanParkingSearchDuration_min"] = self.context.config("suburban_parking_search_min")

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
        with open(self.mode_output_path, 'w') as f:
            for mode in modes:
                f.write(f"# {mode}\n")
                yaml.dump(grouped_params[mode], f, sort_keys=False)
            f.write("# cost\n")
            yaml.dump(grouped_params["cost"], f, sort_keys=False)

        logger.info(f"Model parameters written to {self.mode_output_path}")
    
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
    'beta_bike_low_income': 'bike.betaLowIncome_u',
    'beta_bike_region_1': 'bike.betaRegion1_u',
    'beta_bike_region_2': 'bike.betaRegion2_u',  
    'beta_bike_origin_home': 'bike.betaOriginHome_u',
    'beta_bike_short_distance': 'bike.betaShortDistance_u',
    'beta_bike_destination_work': 'bike.betaDestinationWork_u',
    'beta_bike_destination_urban': 'bike.betaUrbanDestination_u',
    'beta_bike_destination_urbancore': 'bike.betaUrbancoreDestination_u',
    'beta_bike_destination_home': 'bike.betaDestinationHome_u',
    'beta_bike_destination_education': 'bike.betaDestinationEducation_u',
    'beta_bike_destination_shopping': 'bike.betaDestinationShopping_u',
    'beta_bike_destination_leisure': 'bike.betaDestinationLeisure_u',
    'beta_bike_destination_other': 'bike.betaDestinationOther_u',
    'beta_bike_retired': 'bike.betaRetired_u',
    'beta_bike_junior': 'bike.betaJunior_u',
    'beta_bike_long_distance': 'bike.betaLongDistance_u',
    
    # Car
    'beta_car_asc': 'car.alpha_u',
    'beta_car_travel_time_min': 'car.betaTravelTime_u_min',
    'lambda_car_travel_time': 'car.travelTimeExponent',
    'beta_car_age': 'car.betaAge_u',
    'beta_car_sex': 'car.betaSex_u',
    'beta_car_low_income': 'car.betaLowIncome_u',
    'beta_car_region_1': 'car.betaRegion1_u',
    'beta_car_region_2': 'car.betaRegion2_u', 
    'beta_car_origin_home': 'car.betaOriginHome_u',
    'beta_car_destination_urban': 'car.betaUrbanDestination_u',
    'beta_car_destination_urbancore': 'car.betaUrbancoreDestination_u',
    'beta_car_destination_work': 'car.betaDestinationWork_u',
    'beta_car_destination_home': 'car.betaDestinationHome_u',
    'beta_car_destination_education': 'car.betaDestinationEducation_u',
    'beta_car_destination_shopping': 'car.betaDestinationShopping_u',
    'beta_car_destination_leisure': 'car.betaDestinationLeisure_u',
    'beta_car_destination_other': 'car.betaDestinationOther_u',
    'beta_car_retired': 'car.betaRetired_u',
    'beta_car_junior': 'car.betaJunior_u',
    'beta_car_ownership_ratio': 'car.betaCarOwnershipRatio_u',
    'beta_car_short_distance': 'car.betaShortDistance_u',
    'beta_car_long_distance': 'car.betaLongDistance_u',

    # Car Passenger
    'beta_car_passenger_asc': 'cp.alpha_u',
    'beta_car_passenger_travel_time_min': 'cp.betaTravelTime_u_min',
    'lambda_car_passenger_travel_time': 'cp.travelTimeExponent',
    'beta_car_passenger_age': 'cp.betaAge_u',
    'beta_car_passenger_sex': 'cp.betaSex_u',
    'beta_car_passenger_low_income': 'cp.betaLowIncome_u',
    'beta_car_passenger_region_1': 'cp.betaRegion1_u',
    'beta_car_passenger_region_2': 'cp.betaRegion2_u',
    'beta_car_passenger_origin_home': 'cp.betaOriginHome_u',
    'beta_car_passenger_destination_urban': 'cp.betaUrbanDestination_u',
    'beta_car_passenger_destination_urbancore': 'cp.betaUrbancoreDestination_u',
    'beta_car_passenger_destination_work': 'cp.betaDestinationWork_u',
    'beta_car_passenger_destination_home': 'cp.betaDestinationHome_u',
    'beta_car_passenger_destination_education': 'cp.betaDestinationEducation_u',
    'beta_car_passenger_destination_shopping': 'cp.betaDestinationShopping_u',
    'beta_car_passenger_destination_leisure': 'cp.betaDestinationLeisure_u',
    'beta_car_passenger_destination_other': 'cp.betaDestinationOther_u',
    'beta_car_passenger_driving_permit': 'cp.betaDrivingLicense_u',
    'beta_car_passenger_short_distance': 'cp.betaShortDistance_u',
    'beta_car_passenger_retired': 'cp.betaRetired_u',
    'beta_car_passenger_junior': 'cp.betaJunior_u',
    'beta_car_passenger_long_distance': 'cp.betaLongDistance_u',
    'beta_car_passenger_ownership_ratio': 'cp.betaCarOwnershipRatio_u',
    'beta_car_passenger_has_car': 'cp.betaHasCar_u',
    "beta_car_passenger_very_long_distance": 'cp.betaVeryLongDistance_u',    

    # PT
    'beta_pt_asc': 'pt.alpha_u',
    'beta_pt_in_vehicle_time_min': 'pt.betaInVehicleTime_u_min',
    'beta_pt_access_egress_time_min': 'pt.betaAccessEgressTime_u_min',
    'beta_pt_transfer_time_min': 'pt.betaWaitingTime_u_min',
    'beta_pt_transfers': 'pt.betaLineSwitch_u',
    'beta_pt_distance_km': 'pt.betaDistance_u_km',    

    'lambda_pt_in_vehicle_time': 'pt.inVehicleTimeExponent',
    'lambda_pt_access_egress_time': 'pt.accessEgressTimeExponent',    
    'lambda_pt_transfers': 'pt.lineSwitchExponent', 
    'lambda_pt_transfer_time': 'pt.waitingTimeExponent',   
    'lambda_pt_distance': 'pt.distanceExponent',

    'beta_pt_age': 'pt.betaAge_u',
    'beta_pt_sex': 'pt.betaSex_u',
    'beta_pt_low_income': 'pt.betaLowIncome_u',
    'beta_pt_region_1': 'pt.betaRegion1_u',
    'beta_pt_region_2': 'pt.betaRegion2_u', 
    'beta_pt_origin_home': 'pt.betaOriginHome_u',
    'beta_pt_destination_work': 'pt.betaDestinationWork_u',
    'beta_pt_destination_urban': 'pt.betaUrbanDestination_u',
    'beta_pt_destination_urbancore': 'pt.betaUrbancoreDestination_u',
    'beta_pt_destination_home': 'pt.betaDestinationHome_u',
    'beta_pt_destination_education': 'pt.betaDestinationEducation_u',
    'beta_pt_destination_shopping': 'pt.betaDestinationShopping_u',
    'beta_pt_destination_leisure': 'pt.betaDestinationLeisure_u',
    'beta_pt_destination_other': 'pt.betaDestinationOther_u',
    'beta_pt_short_distance': 'pt.betaShortDistance_u',
    'beta_pt_retired': 'pt.betaRetired_u',
    'beta_pt_junior': 'pt.betaJunior_u',
    'beta_pt_long_distance': 'pt.betaLongDistance_u',
    'beta_pt_good_service': 'pt.betaGoodService_u',
    'beta_pt_medium_service': 'pt.betaMediumService_u',
    
    # Walk
    'beta_walk_asc': 'walk.alpha_u',
    'beta_walk_travel_time_min': 'walk.betaTravelTime_u_min',
    'lambda_walk': 'walk.travelTimeExponent',
    'beta_walk_age': 'walk.betaAge_u',
    'beta_walk_sex': 'walk.betaSex_u',
    'beta_walk_low_income': 'walk.betaLowIncome_u',
    'beta_walk_region_1': 'walk.betaRegion1_u',
    'beta_walk_region_2': 'walk.betaRegion2_u', 
    'beta_walk_origin_home': 'walk.betaOriginHome_u',
    'beta_walk_short_distance': 'walk.betaShortDistance_u',
    'beta_walk_destination_work': 'walk.betaDestinationWork_u',
    'beta_walk_destination_urban': 'walk.betaUrbanDestination_u',
    'beta_walk_destination_urbancore': 'walk.betaUrbancoreDestination_u',
    'beta_walk_destination_home': 'walk.betaDestinationHome_u',
    'beta_walk_destination_education': 'walk.betaDestinationEducation_u',
    'beta_walk_destination_shopping': 'walk.betaDestinationShopping_u',
    'beta_walk_destination_leisure': 'walk.betaDestinationLeisure_u',
    'beta_walk_destination_other': 'walk.betaDestinationOther_u',
    'beta_walk_retired': 'walk.betaRetired_u',
    'beta_walk_junior': 'walk.betaJunior_u',
    'beta_walk_long_distance': 'walk.betaLongDistance_u',

    # Cost
    'beta_cost_CHF': 'betaCost_u_MU',
    'lambda_cost_income': 'lambdaCostIncome',
    'lambda_cost_distance': 'lambdaCostEuclideanDistance',
}
