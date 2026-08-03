import inspect
import logging

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Parameters:
    """
    Container for DMC mode and cost parameters.

    The class mirrors the flat parameter names used by the default calibration
    definitions so that values can be accessed by their original keys and saved
    back to YAML without losing nested dictionaries such as swissCanton.*.
    """

    class bike:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0
        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaLowIncome_u: float = 0.0
        betaHighIncome_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaUrbancoreDestination_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationShopping_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaRetired_u: float = 0.0
        betaJunior_u: float = 0.0
        betaDensities_u: float = 0.0

    class car:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0
        betaAccessEgressTime_u_min: float = 0.0
        accessEgressTimeExponent: float = 1.0
        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaLowIncome_u: float = 0.0
        betaHighIncome_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaUrbancoreDestination_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationShopping_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaRetired_u: float = 0.0
        betaJunior_u: float = 0.0
        betaCarOwnershipRatio_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDensities_u: float = 0.0
        betaDestinationZurich_u: float = 0.0
        betaDestinationGeneva_u: float = 0.0
        betaDestinationBasel_u: float = 0.0
        betaDestinationLausanne_u: float = 0.0
        betaDestinationLuzern_u: float = 0.0
        betaDestinationBern_u: float = 0.0
        additionalAccessEgressWalkTime_min: float = 0.0
        constantParkingSearchPenalty_min: float = 0.0

    class parking:
        urbanParkingSearchDuration_min: float = 0.0
        urbancoreParkingSearchDuration_min: float = 0.0
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
        betaLowIncome_u: float = 0.0
        betaHighIncome_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaUrbancoreDestination_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationShopping_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaRetired_u: float = 0.0
        betaJunior_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaGoodService_u: float = 0.0
        betaMediumService_u: float = 0.0
        betaDestinationGoodService_u: float = 0.0
        betaDestinationMediumService_u: float = 0.0
        betaDensities_u: float = 0.0
        betaContainsRail_u: float = 0.0
        betaContainsBus_u: float = 0.0

    class walk:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0
        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaLowIncome_u: float = 0.0
        betaHighIncome_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaUrbancoreDestination_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationShopping_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaRetired_u: float = 0.0
        betaJunior_u: float = 0.0
        betaDensities_u: float = 0.0

    class cp:
        alpha_u: float = 0.0
        betaTravelTime_u_min: float = 0.0
        travelTimeExponent: float = 1.0
        betaDistance_u_km: float = 0.0
        betaDistance_km: float = 0.0
        distanceExponent: float = 1.0
        betaAge_u: float = 0.0
        betaSex_u: float = 0.0
        betaLowIncome_u: float = 0.0
        betaHighIncome_u: float = 0.0
        betaRegion1_u: float = 0.0
        betaRegion2_u: float = 0.0
        betaOriginHome_u: float = 0.0
        betaDestinationWork_u: float = 0.0
        betaUrbanDestination_u: float = 0.0
        betaUrbancoreDestination_u: float = 0.0
        betaDestinationHome_u: float = 0.0
        betaDestinationEducation_u: float = 0.0
        betaDestinationShopping_u: float = 0.0
        betaDestinationLeisure_u: float = 0.0
        betaDestinationOther_u: float = 0.0
        betaDrivingLicense_u: float = 0.0
        betaShortDistance_u: float = 0.0
        betaRetired_u: float = 0.0
        betaJunior_u: float = 0.0
        betaLongDistance_u: float = 0.0
        betaCarOwnershipRatio_u: float = 0.0
        betaHasCar_u: float = 0.0
        betaVeryLongDistance_u: float = 0.0
        betaDensities_u: float = 0.0

    class swissCanton:
        car: dict = {}
        pt: dict = {}
        bike: dict = {}
        walk: dict = {}
        cp: dict = {}

    class cost:
        betaCost_u_MU: float = 0.0
        lambdaCostEuclideanDistance: float = 0.0
        lambdaCostIncome: float = 0.0
        referenceIncome: float = 0.0
        referenceEuclideanDistance_km: float = 0.0
        travelTimeFactor: float = 1.0
        urbanParkingCost_CHF_h: float = 0.0
        urbancoreParkingCost_CHF_h: float = 0.0
        suburbanParkingCost_CHF_h: float = 0.0
        parkingPriceReductionForWork: float = 0.0
        carCost_CHF_km: float = 0.0
        ptRegionalRadius_km: float = 0.0
        betaDestinationEmployeeDensity_u: float = 0.0
        betaDestinationPopulationDensity_u: float = 0.0
        betaDestinationCompaniesDensity_u: float = 0.0
        timeScale_min: float = 0.0
        distanceScale_km: float = 0.0
        ageScale_year: float = 0.0
        populationDensityScale: float = 0.0
        populationDensityExponent: float = 0.0
        employeesDensityScale: float = 0.0
        employeesDensityExponent: float = 0.0
        companiesDensityScale: float = 0.0
        companiesDensityExponent: float = 0.0
        lowIncomeThreshold: float = 0.0
        highIncomeThreshold: float = 0.0
        shortDistance_km: float = 0.0
        longDistance_km: float = 0.0
        veryLongDistance_km: float = 0.0

    @classmethod
    def _section_to_dict(cls, section_name: str, section_obj) -> dict:
        data = {}
        for key, value in vars(section_obj).items():
            if key.startswith("__") or key.startswith("_"):
                continue
            if inspect.isclass(value):
                continue
            if callable(value):
                continue
            if section_name == "cost":
                data[key] = value
            else:
                data[f"{section_name}.{key}"] = value
        return data

    @classmethod
    def to_yaml(cls, file_path: str):
        """
        Save all parameters to a YAML file.
        """
        data = {}

        for name, obj in inspect.getmembers(cls):
            if inspect.isclass(obj) and obj != cls:
                data.update(cls._section_to_dict(name, obj))

        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

        logger.info(f"All DMC parameters saved to: {file_path}")

    @classmethod
    def from_yaml(cls, file_path: str):
        """
        Load parameters from a YAML file, updating class attributes.
        """
        logger.info(f"Reading DMC parameters from: {file_path}")
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}

        for key, value in data.items():
            if "." in key:
                class_name, attribute_name = key.split(".", 1)
            else:
                class_name, attribute_name = "cost", key

            if hasattr(cls, class_name):
                section_class = getattr(cls, class_name)
                if hasattr(section_class, attribute_name):
                    setattr(section_class, attribute_name, value)

    @staticmethod
    def get_parameters(parameters_names: list):
        """
        Retrieve the parameter classes for the given names.
        """
        if isinstance(parameters_names, str):
            parameters_names = [parameters_names]

        if len(parameters_names):
            out_dict = {}
            for name in parameters_names:
                class_name, attr_name = name.split(".", 1)
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
                    class_name, attr_name = key.split(".", 1)
                except ValueError as exc:
                    raise ValueError(f"Invalid key format: '{key}', expected 'class_name.attribute_name'") from exc

                param_class = getattr(Parameters, class_name, None)
                if param_class is None:
                    raise ValueError(f"Unknown parameter group: '{class_name}'")

                if not hasattr(param_class, attr_name):
                    raise AttributeError(f"'{class_name}' has no attribute '{attr_name}'")

                setattr(param_class, attr_name, value)