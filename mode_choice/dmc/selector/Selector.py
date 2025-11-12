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
    def _maximum_utility_selection(df):        

        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]

        # Group by person_id and selection_id
        grouped = df.groupby(['person_id', 'selection_id'], sort=False)

        # Find the row with max utility per group
        max_rows = df.loc[grouped['utility'].idxmax()].index

        # Create selected column
        df['selected'] = False
        df.loc[max_rows, 'selected'] = True

        return df

    @staticmethod
    def _multinomial_logit_selection(df):        
        if Selector.considerMinimumUtility:
            df = df[df['utility'] > Selector.minimum_utility]
        
        df.loc[:,'utility'] = np.minimum(df['utility'], Selector.maximum_utility)
        
        # Compute the probability
        df.loc[:,"exp_util"] = np.exp(df['utility'])        
        group_sum = df.groupby(['person_id', 'selection_id'])['exp_util'].transform('sum')
        df.loc[:,'probability'] = df["exp_util"] / group_sum
               
        # Generate a random number per row
        df.loc[:,'rand'] = np.random.rand(len(df))
        df.loc[:,'rand'] = df.groupby(['person_id', 'selection_id'])['rand'].transform('first')
        
        # Compute cumulative probability within each group
        df.loc[:,'cum_prob'] = df.groupby(['person_id', 'selection_id'])['probability'].cumsum()
        
        # Mark selected where rand < cum_prob and previous cum_prob <= rand
        df.loc[:,'prev_cum_prob'] = df.groupby(['person_id', 'selection_id'])['cum_prob'].shift(fill_value=0)
        df.loc[:,'selected'] = (df['rand'] >= df['prev_cum_prob']) & (df['rand'] < df['cum_prob'])
        
        # Clean up unnecessary columns
        df.drop(columns=['exp_util', 'probability', 'rand', 'cum_prob', 'prev_cum_prob'], inplace=True)

        return df
    

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
                .over("selection_id")
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
                    .over("selection_id")
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

