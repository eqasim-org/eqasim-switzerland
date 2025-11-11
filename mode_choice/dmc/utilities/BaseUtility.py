#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:46:14 2025

@author: dabdelkader
"""

from abc import ABC, abstractmethod, ABCMeta
import pandas as pd
import numpy as np
import polars as pl
from .Parameters import Parameters

class MetaCls(ABCMeta):
    """
    Metaclass that:
    - For attributes containing certain prefixes, proxies to cls.parameters
    - Otherwise, falls back to normal class-level lookup
    """
    def __getattr__(cls, name):
        _prefixes = ["walk", "bike", "pt", "car", "cp", "cost", "swissBike", "swissCar", "parking"]
        
        if any(prefix in name for prefix in _prefixes):            
            if hasattr(cls.parameters, name):
                return getattr(cls.parameters, name)
            else:
                raise AttributeError(f"'{name}' not found in {cls.__name__}.parameters" )
        try:
            return object.__getattribute__(cls, name)
        except AttributeError:
            raise AttributeError( f"'{cls.__name__}' object has no attribute '{name}'" )
    
class BaseUtility(ABC, metaclass=MetaCls):
    """
    Abstract base class for utility computation.
    Provides shared structure and namespaced parameters for different transport modes.
    """
    parameters = Parameters
    
    @staticmethod
    def to_yaml(file_path: str):
        BaseUtility.parameters.to_yaml(file_path)
    
    @staticmethod
    def from_yaml(file_path: str):
        BaseUtility.parameters.from_yaml(file_path)
    
    @staticmethod
    def get_parameters(parameters_names: list):
        return BaseUtility.parameters.get_parameters(parameters_names)
        
    @staticmethod
    def set_parameters(updates: dict):
        BaseUtility.parameters.set_parameters(updates)
    
    @staticmethod
    def interaction(value, reference, lambda_val):
        """
        Computes the distance interaction factor.
        
        Handles both Polars Series and scalar-like inputs.
        """
        DEFAULT_MINIMUM_VALUE = 1e-3
        clipped_value = value.clip(lower_bound=DEFAULT_MINIMUM_VALUE)
        return (clipped_value / reference) ** lambda_val
    
    @staticmethod
    @abstractmethod
    def compute_lazy(variables):
        """
        Abstract method to compute utility from variables, which is a polars dataframe.
        Must be implemented by subclasses.
        """
        pass

    @staticmethod    
    def read_csv(file_path):
        df = pl.read_csv(file_path, separator=";")
        df = df.with_columns(
            (pl.col("person_id").cast(pl.Utf8) + "_" + pl.col("trip_index").cast(pl.Utf8)).alias("trip_key")
        )
        
        df = df.with_columns(
            pl.col("age").cast(pl.Int32),
            pl.col("sex").cast(pl.Int8),
            pl.col("region").cast(pl.Int8),
            pl.col("originHome").cast(pl.Int8),
            pl.col("destinationWork").cast(pl.Int8),
            pl.col("urbanDestination").cast(pl.Int8),
            pl.col("shortDistance").cast(pl.Int8),
            pl.col("euclideanDistance_km").cast(pl.Float64),
            pl.col("utility").cast(pl.Float64),
            pl.col("subUrbanDestination").cast(pl.Int8) if "subUrbanDestination" in df.columns else None,
        )
        return df


