#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:21:26 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class BikeUtility(BaseUtility):
    
    @staticmethod
    def estimateRegionalUtility():
        beta1 = BaseUtility.bike.betaRegion1_u
        beta2 = BaseUtility.bike.betaRegion2_u
        return ( pl.when(pl.col("region") == 1)
                   .then(beta1)
                   .when(pl.col("region") == 2)
                   .then(beta2)
                   .otherwise(0.0)  )
                   
    @staticmethod
    def estimateLongDistanceUtility():        
        return ( pl.when(pl.col("euclidean_distance_km") > 20.0)
                   .then(-1e3)
                   .otherwise(0.0)  )
    
    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.bike.alpha_u +
            BaseUtility.bike.betaTravelTime_u_min * pl.col("travel_time_min").pow(BaseUtility.bike.travelTimeExponent) +
            BaseUtility.bike.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 18) +
            BaseUtility.bike.betaSex_u * pl.col("sex") +
            BikeUtility.estimateRegionalUtility() +
            BaseUtility.bike.betaOriginHome_u * pl.col("origin_home") +
            BaseUtility.bike.betaShortDistance_u * pl.col("short_distance") +
            BaseUtility.bike.betaUrbanDestination_u * pl.col("urban_destination") +
            BaseUtility.bike.betaDestinationWork_u * pl.col("destination_work") +
            BikeUtility.estimateLongDistanceUtility()
        )

        return utility

    
    
    
    
    
    
    
    
    
    