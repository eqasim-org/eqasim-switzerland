#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 18:16:46 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import polars as pl

class PtUtility(BaseUtility):
    
    @staticmethod
    def estimateRegionalUtility():        
        beta1 = BaseUtility.pt.betaRegion1_u
        beta2 = BaseUtility.pt.betaRegion2_u        
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

        cost_correction = (10.0-pl.col("euclidean_distance_km")).clip(lower_bound=0.0) ** BaseUtility.pt.distanceExponent
        corrected_cost = pl.col("cost_CHF") + BaseUtility.pt.betaDistance_u_km * cost_correction
        cost = BaseUtility.cost.betaCost_u_MU * interaction_distance * interaction_income * corrected_cost
        return cost
    
    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.pt.alpha_u
            + BaseUtility.pt.betaInVehicleTime_u_min * pl.col("in_vehicle_time_min").pow(BaseUtility.pt.inVehicleTimeExponent)
            + BaseUtility.pt.betaAccessEgressTime_u_min * pl.col("access_egress_time_min").pow(BaseUtility.pt.accessEgressTimeExponent)
            + BaseUtility.pt.betaWaitingTime_u_min * pl.col("waiting_time_min").pow(BaseUtility.pt.waitingTimeExponent)
            + BaseUtility.pt.betaLineSwitch_u * pl.col("transfers").pow(BaseUtility.pt.lineSwitchExponent)
            + PtUtility.estimateCostUtility()
            + BaseUtility.pt.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 17)
            + BaseUtility.pt.betaSex_u * pl.col("sex")
            + PtUtility.estimateRegionalUtility()
            + BaseUtility.pt.betaOriginHome_u * pl.col("origin_home")
            + BaseUtility.pt.betaShortDistance_u * pl.col("short_distance")
            + BaseUtility.pt.betaLongDistance_u * pl.col("long_distance")
            + BaseUtility.pt.betaUrbanDestination_u * pl.col("urban_destination")
            + BaseUtility.pt.betaDestinationWork_u * pl.col("destination_work")
            + BaseUtility.pt.betaDestinationOther_u * pl.col("destination_other")
            + BaseUtility.pt.betaDestinationLeisure_u * pl.col("destination_leisure")
            + BaseUtility.pt.betaDestinationEducation_u * pl.col("destination_education")
            + BaseUtility.pt.betaDestinationHome_u * pl.col("destination_home")
            + BaseUtility.pt.betaWorkingHour_u * pl.col("working_hour")
        )

        return utility
        
        
    
    
    
    
    
    
    
    
    
    
    
    