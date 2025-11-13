#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:19:01 2025

@author: dabdelkader
"""

from utilities.TourUtility import TourUtility
import numpy as np
import pandas as pd
import polars as pl
import random

# fix the seed for reproducibility
random.seed(1102)
np.random.seed(1102)


class Selector():
    minimum_utility = -700.0
    maximum_utility = 700.0
    considerMinimumUtility = False
    selector = "MultinomialLogit"  # one of ["MultinomialLogit", "Maximum"]
    gumble  = None
    
    @staticmethod
    def select(tours: pd.DataFrame):                        
        if Selector.selector == "MultinomialLogit":
            return Selector._multinomial_logit_selection_polars(tours)
            
        elif Selector.selector == "Maximum":
            return Selector._maximum_utility_selection_polars(tours)
        else:
            raise ValueError(f"Unknown selector: {Selector.selector}")


    @staticmethod
    def _multinomial_logit_selection_polars(df: pl.DataFrame) -> pl.DataFrame:
        # Optional filter step (only if condition enabled)
        if Selector.considerMinimumUtility:
            df = df.filter(pl.col("utility") > Selector.minimum_utility)
        
        if Selector.gumble is None: # only the first time
            Selector.gumble = pl.lit(np.random.gumbel(size=df.height))
        
        # Clip utility and add Gumbel noise in a single with_columns call
        df = df.with_columns([
            pl.col("utility").clip(upper_bound=Selector.maximum_utility).alias("utility"),
            (pl.col("utility") + Selector.gumble).alias("noisy_utility")
        ])

        return (
        df.with_columns(
            pl.col("noisy_utility")
                .rank(method="ordinal", descending=True)
                .over("tour_id")
                .alias("rn")
        )
        .filter(pl.col("rn") == 1)
        .drop(["rn", "noisy_utility"])
       )

    @staticmethod
    def _maximum_utility_selection_polars(df:pl.DataFrame)-> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("utility")
                    .rank(method="ordinal", descending=True)
                    .over("tour_id")
                    .alias("rn")
            )
            .filter(pl.col("rn") == 1)
            .drop(["rn"])
        )

    @staticmethod
    def set_selector(selector:str):
        if selector not in ["MultinomialLogit", "Maximum"]:
            raise ValueError(f"Unknown selector: {selector}")
        Selector.selector = selector
    
    @staticmethod
    def get_selector()->str:
        return Selector.selector

    @staticmethod
    def set_seed(seed:int):
        random.seed(seed)
        np.random.seed(seed)
        Selector.gumble = None  # reset gumble to regenerate with new seed
