#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:00:03 2025

@author: dabdelkader
"""

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Parameters(ABC):
    """
    Abstract base class for utility computation.
    Provides shared structure and namespaced parameters for different transport modes.
    """

    class bike:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0

        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaWorkingHour_u: float = 0.0

    class car:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0
        betaAccessEgressTime_u_min: float = 0.0
        accessEgressTimeExponent: float = 1.0
        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0        
        betaDestinationEducation_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDrivingLicense_u: float = 0.0
        betaWorkingHour_u: float = 0.0


    class parking:
        urbanParkingSearchDuration_min: float = 0.0
        suburbanParkingSearchDuration_min: float = 0.0

    class pt:
        alpha_u: float = 0.0

        betaLineSwitch_u: float = 0.0
        betaInVehicleTime_u_min: float = 0.0
        betaWaitingTime_u_min: float = 0.0
        betaAccessEgressTime_u_min: float = 0.0

        betaDistance_u_km: float = 0.0

        inVehicleTimeExponent: float = 1.0
        waitingTimeExponent: float = 1.0
        accessEgressTimeExponent: float = 1.0
        lineSwitchExponent: float = 1.0
        distanceExponent: float = 1.0

        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaWorkingHour_u: float = 0.0

    class walk:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0

        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaWorkingHour_u: float = 0.0

    class cp:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0

        betaDistance_km: float = 0.0
        distanceExponent: float = 1.0

        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaDrivingLicense_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaWorkingHour_u: float = 0.0

    class cost:        
        betaCost_u_MU: float = 0
        lambdaCostEuclideanDistance: float = 0
        lambdaCostIncome: float = 0
        referenceIncome: float = 0
        referenceEuclideanDistance_km: float = 0

    @classmethod
    def to_yaml(cls, file_path: str):        
        """
        Save all parameters to a YAML file.
        """
        data = {}

        for name, obj in inspect.getmembers(cls):
            if inspect.isclass(obj) and obj != cls:                
                for key, value in inspect.getmembers(obj):
                    if (not key.startswith('__')) and ((isinstance(value, float)) or (isinstance(value, int))):
                        if name=="cost":
                            data[key] = float(value)
                        else:                        
                            data[name+"."+key] = float(value)

        with open(file_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        
        logger.info(f"All DMC parameters saved to: {file_path}")

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Load parameters from a YAML file, updating class attributes.
        """
        logger.info(f"Reading DMC parameters from: {file_path}")
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        for k, v in data.items():
            if "." in k:
                class_name, attribute_name = k.split('.')
            else:
                class_name, attribute_name = "cost", k
            
            if hasattr(cls, class_name):
                section_class = getattr(cls, class_name)
                if hasattr(section_class, attribute_name):
                    setattr(section_class, attribute_name, v)
                        
    @staticmethod
    def get_parameters(parameters_names: list):
        """
        Retrieve the parameter classes for the given names.
        """
        if isinstance(parameters_names, str):
            parameters_names = [parameters_names]
            
        if len(parameters_names):
            out_dict = dict()
            for name in parameters_names:
                class_name, attr_name = name.split('.', 1)
                param_class = getattr(Parameters, class_name, None)
                if param_class is None:
                    raise ValueError(f"Unknown parameter: '{class_name}'")
                    
                out_dict[name] = getattr(param_class, attr_name, None)
                if out_dict[name] is None:
                    raise ValueError(f"Unknown parameter: '{name}'")
                
            return out_dict 

    @staticmethod
    def set_parameters(updates: dict):
        """
        Update parameters using dot notation keys like 'car.alpha_u'.
    
        Parameters:
        - updates: dict of form {
            'car.alpha_u': 0.9,
            'pt.betaWaitingTime_u_min': -0.04
          }
        """
        if len(updates):
            for key, value in updates.items():
                try:
                    class_name, attr_name = key.split('.', 1)
                except ValueError:
                    raise ValueError(f"Invalid key format: '{key}', expected 'class_name.attribute_name'")
                
                param_class = getattr(Parameters, class_name, None)
                if param_class is None:
                    raise ValueError(f"Unknown parameter group: '{class_name}'")
                
                if not hasattr(param_class, attr_name):
                    raise AttributeError(f"'{class_name}' has no attribute '{attr_name}'")
                
                setattr(param_class, attr_name, value)
