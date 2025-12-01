#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:55:47 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import pandas as pd
import polars as pl

class WalkUtility(BaseUtility):
    
    @staticmethod
    def estimateRegionalUtility():
        beta1 = BaseUtility.walk.betaRegion1_u
        beta2 = BaseUtility.walk.betaRegion2_u
        return ( pl.when(pl.col("region") == 1)
                   .then(beta1)
                   .when(pl.col("region") == 2)
                   .then(beta2)
                   .otherwise(0.0)  )

    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.walk.alpha_u
            + BaseUtility.walk.betaTravelTime_u_min * pl.col("travel_time_min").pow(BaseUtility.walk.travelTimeExponent)
            + BaseUtility.walk.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 18)
            + BaseUtility.walk.betaSex_u * pl.col("sex")
            + WalkUtility.estimateRegionalUtility()
            + BaseUtility.walk.betaOriginHome_u * pl.col("origin_home")
            + BaseUtility.walk.betaShortDistance_u * pl.col("short_distance")
            + BaseUtility.walk.betaLongDistance_u * pl.col("long_distance")
            + BaseUtility.walk.betaUrbanDestination_u * pl.col("urban_destination")
            + BaseUtility.walk.betaDestinationWork_u * pl.col("destination_work")
            + BaseUtility.walk.betaDestinationOther_u * pl.col("destination_other")
            + BaseUtility.walk.betaDestinationLeisure_u * pl.col("destination_leisure")
        )

        return utility