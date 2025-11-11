#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:12:48 2025

@author: dabdelkader
"""

from .BaseUtility import BaseUtility
import pandas as pd
import numpy as np
import polars as pl

class ZeroUtility(BaseUtility):
    
    @staticmethod
    def compute(variables):
        return variables["euclideanDistance_km"]*0.0 #just to make it same ttype
    
    @staticmethod
    def compute_lazy():
        return pl.lit(0.0) 
    