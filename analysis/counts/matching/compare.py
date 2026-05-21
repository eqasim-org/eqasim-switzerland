#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:35:11 2025

@author: dabdelkader
"""
import pandas as pd
from typing import Optional, Union, List
import numpy as np
import logging
import os
import glob

logger = logging.getLogger("synpp")

def configure(context):
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("output_prefix", "switzerland_")

def execute(context):        
    output_path = context.config("output_path")
    output_id   = context.config("output_id")
    simulation_directory = context.config("simulation_directory")
    iterations_directory = os.path.join(output_path, output_id, simulation_directory, "ITERS")
    
    # searching for link stats file 
    linkstats_file = glob.glob(os.path.join(iterations_directory, "it.*", f"*.linkstats.txt.gz"))
    counts_file = glob.glob(os.path.join(iterations_directory, "it.*", f"*.traffic_flow_daily_counts.csv"))
    files = counts_file if counts_file else linkstats_file

    if not files:
        raise FileNotFoundError(f"No counts nor link stats files found in {iterations_directory}")
    else:
        if files == counts_file:
            logger.info(f"Found {len(files)} counts files for comparison. Considering the last 10 iterations.")
            # Sort files by modification time and take the last 10
            files = sorted(files, key=os.path.getmtime)[-10:]
            logger.info(f"Using the following files for averaging: {files}")
            
            # Read and average the data
            dfs = [pd.read_csv(file, sep=";", usecols=["linkId", "dailyCount"], dtype={"linkId": str, "dailyCount": float}) for file in files]
            combined_df = pd.concat(dfs).groupby("linkId", as_index=False).mean()            
            
            # Save the averaged data to a temporary file for comparison
            temp_file = files[-1].replace(".csv", "_averagedOverLast10Iterations.csv")
            combined_df.to_csv(temp_file, index=False, sep=";")
            logger.info(f"Averaged counts file saved to {temp_file}")
            file = temp_file
        else:
            logger.info(f"Found {len(files)} linkstats files for comparison. Using the latest one.")
            # Get the latest file (by modification time)
            file = max(files, key=os.path.getmtime)
            logger.info(f"\t Using file {file} for comparison.")

    # build the compare object and return it
    cmp = Compare(file)
    
    return cmp
    
  









class Compare:
    def __init__(self, link_stats_file: Optional[str] = None):
        self.link_stats_file = link_stats_file
        self.link_stats = None

        if link_stats_file:
            self.link_stats = self.read_link_stats(link_stats_file)

    @staticmethod
    def read_link_stats(file_path: str, columns: Union[str, List[str]] = "HRS0-24avg") -> pd.DataFrame:
        if "linkstats" in file_path:
            if isinstance(columns, str):
                columns = [columns]

            dtype = {"LINK": str}
            dtype.update({col: float for col in columns})

            logger.info("Reading link statistics from %s...", file_path)
            df = pd.read_csv(file_path, sep="\t", usecols=["LINK", *columns], dtype=dtype)

            column_renames = {"LINK": "link_id"}
            if "HRS0-24avg" in df.columns:
                column_renames["HRS0-24avg"] = "flow"

            df = df.rename(columns=column_renames)
            logger.info("Link statistics successfully read.")
            return df
        else:
            df = pd.read_csv(file_path, sep=";", usecols=["linkId","dailyCount"], dtype={"linkId":str,"dailyCount":float})
            df = df.rename(columns={"linkId": "link_id", "dailyCount": "flow"})
            return df

    def get_link_stats(self, file_path: Optional[str] = None) -> pd.DataFrame:
        if file_path:
            return self.read_link_stats(file_path)
        elif self.link_stats is not None:
            return self.link_stats
        elif self.link_stats_file:
            return self.read_link_stats(self.link_stats_file)
        else:
            raise ValueError("No link statistics file provided or loaded.")
        
        
    def compare_flow_total_efficient(self, counts, matched, network, file_path: Optional[str] = None, 
                                           sample_size: float = None, get_average: bool = False,
                                           flow_col:str="flow"):
        logger.info("Comparing the simulated flow vs. counts ...")
    
        # Load and filter simulation flow data
        df_sim = self.get_link_stats(file_path).copy()
        df_counts = counts.counts[['id', flow_col]].copy().rename(columns={flow_col:"flow"})
    
        # Get unique simulation links for filtering
        all_links = network.get_in_simulation_links(matched.link_id.unique())
        assert all(pd.notna(all_links)), "Not all links are found in the network"
        df_sim = df_sim[df_sim.link_id.isin(all_links)]
    
        # Expand matched with real simulation links
        _matched = matched.copy()
        _matched['sim_link'] = network.get_in_simulation_links(_matched['link_id'])
    
        # Merge simulated flow values
        df_sim = df_sim[['link_id', 'flow']].rename(columns={'link_id': 'sim_link', 'flow': 'sim_flow'})
        _matched = _matched.merge(df_sim, on='sim_link', how='left')
        _matched = _matched[_matched["sim_flow"].notna()].reset_index(drop=True)
        
        if get_average:
            # Separate by direction and aggregate
            same = _matched[_matched['direction'] == 'same'].groupby('id')['sim_flow'].mean()
            oppo = _matched[_matched['direction'] == 'opposite'].groupby('id')['sim_flow'].mean()
            
            simulated_flow = same.add(oppo, fill_value=np.nan, level="index").reset_index()
            simulated_flow = simulated_flow[simulated_flow.sim_flow.notna()].reset_index(drop=True)
        else:
            assert all(_matched.groupby('id').apply(len)<3), "More than two links for one count"
            simulated_flow = _matched.groupby('id')['sim_flow'].sum().reset_index()
    
        # Merge with counts
        result = df_counts.merge(simulated_flow, on='id', how='left')
        result.rename(columns={'sim_flow': 'simulated_flow'}, inplace=True)
        result = result[result.isna().sum(axis=1)==0].reset_index(drop=True)
        
        # Rescale if needed
        if sample_size is not None:
            result['simulated_flow'] *= 1 / sample_size
    
        result['pdiff'] = ((result['simulated_flow'] - result['flow']) / result['flow'] * 100).astype(int)
        result['adiff'] = (result['simulated_flow'] - result['flow']).astype(int)
    
        return result
                
            
            
            
            