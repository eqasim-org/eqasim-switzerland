#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:48:11 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import polars as pl

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
    
        cost = BaseUtility.cost.betaCost_u_MU * interaction_distance * interaction_income * pl.col("cost_CHF")
        return cost
    

    @staticmethod
    def parkingSearchDuration():
        return ( pl.when((pl.col("destination_home") == 1) | (pl.col("destination_work") == 1))
                   .then(0.0) 
                   .when(pl.col("urban_destination") == 1)
                   .then(BaseUtility.parking.urbanParkingSearchDuration_min)
                   .when(pl.col("suburban_destination") == 1)
                   .then(BaseUtility.parking.suburbanParkingSearchDuration_min)
                   .otherwise(0.0)  )
    
    @staticmethod
    def estimateTraveltimeUtility():
        # Combine travel time and parking search duration before exponentiation for efficiency
        total_time = pl.col("travel_time_min") + CarUtility.parkingSearchDuration()
        return BaseUtility.car.betaTravelTime_u_min * total_time.pow(BaseUtility.car.travelTimeExponent)

    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.car.alpha_u +
            CarUtility.estimateTraveltimeUtility() +
            CarUtility.estimateCostUtility() +
            BaseUtility.car.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 18) +
            BaseUtility.car.betaSex_u * pl.col("sex") +
            CarUtility.estimateRegionalUtility() +
            BaseUtility.car.betaOriginHome_u * pl.col("origin_home") +
            BaseUtility.car.betaShortDistance_u * pl.col("short_distance") +
            BaseUtility.car.betaUrbanDestination_u * pl.col("urban_destination") +
            BaseUtility.car.betaDestinationWork_u * pl.col("destination_work")         
        )

        return utility
        
        