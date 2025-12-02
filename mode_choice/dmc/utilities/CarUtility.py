#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:48:11 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import polars as pl
from mode_choice.dmc_defaults import Defaults
LONG_DISTANCE_LIMIT_KM = Defaults.LONG_DISTANCE_LIMIT_KM


class CarUtility(BaseUtility):    
    
    @staticmethod
    def estimateRegionalUtility():        
        beta1 = BaseUtility.car.betaRegion1_u
        beta2 = BaseUtility.car.betaRegion2_u        
        return ( pl.when(pl.col("region") == 1)
                   .then(beta1)
                   .when(pl.col("region") == 2)
                   .then(beta2)
                   .otherwise(0.0)  )
    @staticmethod
    def estimateCostUtility():
        interaction_distance = BaseUtility.interaction(
            pl.col("euclidean_distance_km"), 
            reference=BaseUtility.cost.referenceEuclideanDistance_km, 
            lambda_val=BaseUtility.cost.lambdaCostEuclideanDistance            
        )

        interaction_income = BaseUtility.interaction(
            pl.col("income"), 
            reference=BaseUtility.cost.referenceIncome, 
            lambda_val=BaseUtility.cost.lambdaCostIncome            
        )
        cost_CHF = pl.col("cost_CHF") + pl.col("parking_cost_CHF")
        interaction = interaction_distance * interaction_income
        cost_utility = BaseUtility.cost.betaCost_u_MU * interaction * cost_CHF
        return cost_utility
    
    @staticmethod
    def estimateTraveltimeUtility():
        # Combine travel time and parking search duration before exponentiation for efficiency
        total_time = pl.col("travel_time_min") + pl.col("parking_searching_duration_min")
        return BaseUtility.car.betaTravelTime_u_min * total_time.pow(BaseUtility.car.travelTimeExponent)

    @staticmethod
    def estimateAcessEgressTimeUtility():
        return BaseUtility.car.betaAccessEgressTime_u_min * pl.col("access_egress_time_min").pow(BaseUtility.car.accessEgressTimeExponent)
    
    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.car.alpha_u
            + CarUtility.estimateTraveltimeUtility()
            + CarUtility.estimateAcessEgressTimeUtility()
            + CarUtility.estimateCostUtility()
            + BaseUtility.car.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 18)
            + BaseUtility.car.betaSex_u * pl.col("sex")
            + CarUtility.estimateRegionalUtility()
            + BaseUtility.car.betaOriginHome_u * pl.col("origin_home")
            + BaseUtility.car.betaShortDistance_u * pl.col("short_distance")
            + BaseUtility.car.betaLongDistance_u * pl.max_horizontal(0.0, pl.col("distance_km") - LONG_DISTANCE_LIMIT_KM)
            + BaseUtility.car.betaUrbanDestination_u * pl.col("urban_destination")
            + BaseUtility.car.betaDestinationWork_u * pl.col("destination_work")
            + BaseUtility.car.betaDestinationOther_u * pl.col("destination_other")
            + BaseUtility.car.betaDestinationLeisure_u * pl.col("destination_leisure")
                  
        )

        return utility
        
        