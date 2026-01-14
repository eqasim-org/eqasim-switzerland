#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:48:11 2025

@author: dabdelkader
"""
from .BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class CpUtility(BaseUtility):    
    
    @staticmethod
    def estimateRegionalUtility():        
        beta1 = BaseUtility.cp.betaRegion1_u
        beta2 = BaseUtility.cp.betaRegion2_u
        return ( pl.when(pl.col("region") == 1)
                   .then(beta1)
                   .when(pl.col("region") == 2)
                   .then(beta2)
                   .otherwise(0.0)  )
                    
    @staticmethod
    def compute_lazy():
        
        utility = (
            BaseUtility.cp.alpha_u
            + BaseUtility.cp.betaTravelTime_u_min * (pl.col("travel_time_min") / BaseUtility.cost.travelTimeFactor).pow(BaseUtility.cp.travelTimeExponent)
            + BaseUtility.cp.betaDrivingLicense_u * pl.col("driving_license")
            + BaseUtility.cp.betaAge_u * pl.max_horizontal(0.0, pl.col("age") - 17)
            + BaseUtility.cp.betaSex_u * pl.col("sex")
            + CpUtility.estimateRegionalUtility()
            + BaseUtility.cp.betaOriginHome_u * pl.col("origin_home")
            + BaseUtility.cp.betaShortDistance_u * pl.col("short_distance")
            + BaseUtility.cp.betaLongDistance_u * pl.col("long_distance")
            + BaseUtility.cp.betaUrbanDestination_u * pl.col("urban_destination")
            + BaseUtility.cp.betaUrbancoreDestination_u * pl.col("urbancore_destination")
            + BaseUtility.cp.betaRuralDestination_u * pl.col("rural_destination")
            + BaseUtility.cp.betaDestinationWork_u * pl.col("destination_work")
            + BaseUtility.cp.betaDestinationOther_u * pl.col("destination_other")
            + BaseUtility.cp.betaDestinationLeisure_u * pl.col("destination_leisure")
            + BaseUtility.cp.betaDestinationEducation_u * pl.col("destination_education")
            + BaseUtility.cp.betaDestinationHome_u * pl.col("destination_home")
            + BaseUtility.cp.betaWorkingHour_u * pl.col("working_hour")
            + BaseUtility.cp.betaIsRetired_u * pl.col("is_retired")
                 
        )

        return utility
        
        