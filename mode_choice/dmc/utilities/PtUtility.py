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
            pl.col("euclideanDistance_km"), 
            reference=BaseUtility.cost.referenceEuclideanDistance_km, 
            lambda_val=BaseUtility.cost.lambdaCostEuclideanDistance            
        )

        interaction_income = BaseUtility.interaction(
            pl.col("income"), 
            reference=BaseUtility.cost.referenceIncome, 
            lambda_val=BaseUtility.cost.lambdaCostIncome            
        )

        cost_correction = (10.0-pl.col("euclideanDistance_km")).clip(lower_bound=0.0) ** BaseUtility.pt.distanceExponent
        corrected_cost = pl.col("cost_MU") + BaseUtility.pt.betaDistance_u_km * cost_correction
        cost = BaseUtility.cost.betaCost_u_MU * interaction_distance * interaction_income * corrected_cost
        return cost
    
    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.pt.alpha_u +
            BaseUtility.pt.betaInVehicleTime_u_min * pl.col("inVehicleTime_min").pow(BaseUtility.pt.inVehicleTimeExponent) +
            BaseUtility.pt.betaAccessEgressTime_u_min * pl.col("accessEgressTime_min").pow(BaseUtility.pt.accessEgressTimeExponent) +
            BaseUtility.pt.betaWaitingTime_u_min * pl.col("waitingTime_min").pow(BaseUtility.pt.waitingTimeExponent) +
            BaseUtility.pt.betaLineSwitch_u * pl.col("numberOfLineSwitches").pow(BaseUtility.pt.lineSwitchExponent) +
            PtUtility.estimateCostUtility() +
            BaseUtility.pt.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 18) +
            BaseUtility.pt.betaSex_u * pl.col("sex") +
            PtUtility.estimateRegionalUtility() +
            BaseUtility.pt.betaOriginHome_u * pl.col("originHome") +
            BaseUtility.pt.betaShortDistance_u * pl.col("shortDistance") +
            BaseUtility.pt.betaUrbanDestination_u * pl.col("urbanDestination") +
            BaseUtility.pt.betaDestinationWork_u * pl.col("destinationWork")         
        )

        return utility
        
        
    
    
    
    
    
    
    
    
    
    
    
    