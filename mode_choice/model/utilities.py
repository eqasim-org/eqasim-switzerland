"""Polars implementation of the current DMC export's utility equations."""
import polars as pl
from dmc.models.specification import LINEAR_TERMS
from dmc.parameters import NAMES_CONVERSION

from mode_choice.defaults import (MODES, ADULT_AGE_REFERENCE, CP_DISTANCE_REFERENCE_KM, POWER_FLOOR,
                                  DISTANCE_COST_INTERACTION, INCOME_COST_INTERACTION, CANTON_ALPHA_KEYS, CALIBRATION_CANTON_COLUMN)


def power(base, exponent):
    # Match dmc.models.model_jax._safe_power, including zero ** zero.
    if exponent == 1.0:
        return pl.when(base > 0).then(base).otherwise(0.0)
    return pl.when(base > 0).then(base.clip(lower_bound=POWER_FLOOR).pow(exponent)).otherwise(0.0)


def expressions(p, *, distance_interaction=DISTANCE_COST_INTERACTION, income_interaction=INCOME_COST_INTERACTION):
    def beta(name):
        return p[NAMES_CONVERSION[name]]

    distance = pl.col("euclidean_distance_km")
    cost_factor = pl.lit(1.0)
    if distance_interaction:
        cost_factor *= power(distance / p["referenceEuclideanDistance_km"], p["lambdaCostEuclideanDistance"])
    if income_interaction:
        cost_factor *= power(pl.col("income") / p["referenceIncome"], p["lambdaCostIncome"])
    cost_factor *= p["betaCost_u_MU"]

    density = pl.lit(0.0)
    for feature, key, coefficient in [
        ("employee", "employees", "Employee"),
        ("population", "population", "Population"),
        ("companies", "companies", "Companies"),
    ]:
        density += (p[f"betaDestination{coefficient}Density_u"]
                    * power(pl.col(f"destination_{feature}_density"), p[f"{key}DensityExponent"])
                    / p[f"{key}DensityScale"])

    result = []
    for mode in MODES:
        prefix = "beta_car_passenger" if mode == "car_passenger" else f"beta_{mode}"
        key = "cp" if mode == "car_passenger" else mode
        u = (pl.lit(beta(f"{prefix}_asc"))
             + beta(f"{prefix}_age") * (pl.col("age").cast(pl.Float64) - ADULT_AGE_REFERENCE).clip(lower_bound=0) / p["ageScale_year"])
        for suffix, column in LINEAR_TERMS[mode]:
            coefficient = beta(f"{prefix}_{suffix}")
            if coefficient != 0.0:
                u += coefficient * pl.col(column).cast(pl.Float64)
        u += beta(f"{prefix}_densities") * density
        cantons = p.get(CANTON_ALPHA_KEYS[mode], {})
        if cantons:
            u += pl.col(CALIBRATION_CANTON_COLUMN).cast(pl.String).replace_strict(cantons, default=0.0, return_dtype=pl.Float64)
        if mode == "car":
            time = pl.col("car_travel_time_min") + pl.col("parking_searching_duration_min")
            u += p["car.betaTravelTime_u_min"] * power(time / p["timeScale_min"], p["car.travelTimeExponent"])
            u += cost_factor * (pl.col("car_cost_CHF") + pl.col("parking_cost_CHF"))
        elif mode == "pt":
            for column, coefficient, exponent in [
                ("pt_access_egress_time_min", "betaAccessEgressTime_u_min", "accessEgressTimeExponent"),
                ("pt_in_vehicle_time_min", "betaInVehicleTime_u_min", "inVehicleTimeExponent"),
            ]:
                u += p[f"pt.{coefficient}"] * power(pl.col(column) / p["timeScale_min"], p[f"pt.{exponent}"])
            u += p["pt.betaWaitingTime_u_min"] * pl.col("pt_transfer_time_min") / p["timeScale_min"]
            u += p["pt.betaLineSwitch_u"] * power(pl.col("pt_transfers"), p["pt.lineSwitchExponent"])
            correction = (p["ptRegionalRadius_km"] - distance).clip(lower_bound=0) / p["distanceScale_km"]
            u += cost_factor * (pl.col("pt_cost_CHF") + p["pt.betaDistance_u_km"] * power(correction, p["pt.distanceExponent"]))
        else:
            u += p[f"{key}.betaTravelTime_u_min"] * power(pl.col(f"{mode}_travel_time_min") / p["timeScale_min"], p[f"{key}.travelTimeExponent"])
            if mode == "car_passenger":
                u += p["cp.betaDistance_u_km"] * (pl.col("car_passenger_distance_km") - CP_DISTANCE_REFERENCE_KM).clip(lower_bound=0) / p["distanceScale_km"]
        result.append(pl.when(pl.col(f"{mode}_availability")).then(u).otherwise(float("-inf")).alias(f"utility_{mode}"))
    return result
