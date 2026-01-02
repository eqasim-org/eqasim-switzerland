#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:34:19 2025

@author: dabdelkader
"""

import matplotlib.pyplot as plt
import seaborn as sns        
import pandas as pd
import geopandas as gpd
from .counts import Counts
from .network import Network
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pydeck as pdk
import folium
from typing import Union
from sklearn.metrics import r2_score
from .matcher_utils import MatcherUtils

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Plotter:
    def plot_flow(self, flows, counts:Counts=None, output_file:str=None, 
                        distance_to_border:int=5000, title:str=None, show_range=False):        
        flows = flows.copy()
        flows = flows.sort_values("flow")
        
        if show_range:
            flows = flows.merge(counts.counts[['id', 'flow_lower', 'flow_upper']], on="id",how="left")
            flow_lower = flows.flow_lower
            flow_upper = flows.flow_upper
                            
        x = flows.flow
        y = flows.simulated_flow
        # Create the figure
        plt.figure(figsize=(8, 8))
        
        # Scatter plot with better styling
        plt.scatter(x, y, alpha=0.6, edgecolor='k', linewidth=0.5, s=60, c='steelblue', label='Count stations')
        if show_range:
            xerr = [np.maximum(x - flow_lower,0), np.maximum(flow_upper - x,0)]
            plt.errorbar(x, y, xerr=xerr, fmt='o', ecolor='gray', alpha=0.6, label='IQR (10-90%)')
            
        # Plot 1:1 line
        max_val = max(x.max(), y.max()) * 1.05
        plt.plot([0, max_val], [0, max_val], 'k--', lw=1.5, label='1:1 Reference Line')
        
        # Optional: linear trend line (regression)
        slope = np.sum(x * y) / np.sum(x * x) 
        plt.plot(x, slope * x, color='crimson', lw=2, linestyle='-', label=f'Trend: y={slope:.2f} x')
        
        # Get R2
        r2 = r2_score(x,y)
        
        if counts is not None and distance_to_border>0:
            border_points = counts.get_near_border_ids(distance_to_border)
            border_flow = flows[flows["id"].isin(border_points["id"])]
            plt.scatter(border_flow.flow, border_flow.simulated_flow , 
                        alpha=0.6, s=10, c='orange', label=f"Within {distance_to_border} m to borders")            
            # Optional: linear trend line (regression)
            in_flow = flows[~flows["id"].isin(border_points["id"])].reset_index(drop=True)
            slope = np.sum(in_flow.flow * in_flow.simulated_flow) / np.sum(in_flow.flow * in_flow.flow) 
            
            plt.plot(x, slope * x, color='darkmagenta', lw=2, linestyle='-', label=f'Trend without borders: y={slope:.2f} x')
            r2_in = r2_score(in_flow.flow, in_flow.simulated_flow)
            
        # Add R2 Score
        
        plt.text( 0.02 * max_val, 0.6 * max_val, 
                 f"$R^2$ = {r2:.3f}\n$R^2_{{\\mathrm{{in}}}}$ = {r2_in:.3f}" if (counts is not None and distance_to_border>0) else f"$R^2$ = {r2:.3f}",            # text
                 fontsize=14,
                 color='crimson',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')  # white box
                 )
        
        # Axis labels and title
        plt.xlabel("Observed Flow (Weekday Avg, 2023)", fontsize=15, labelpad=13)
        plt.ylabel("Simulated Flow (MATSim, 10%)", fontsize=15, labelpad=13)
        plt.title("Observed vs Simulated Traffic Flows" if title is None else title, fontsize=17)
        
        # Add grid, legend        
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=14, loc='upper left')
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close()
        
    def plot_flow_by_road_type(self, flows, network, matched, counts = None,
                                     distance_to_border = 0, 
                                     title:str=None,
                                     output_file:str=None):
        flows = flows.copy()
        flows = flows.merge(matched[["id","link_id"]], on="id", how="left"
                    ).merge(network.links[["link_id","highway"]], on="link_id",how="left")
        flows = flows[flows.highway.notna()]
        
        highway_to_plot = list(MatcherUtils.get_link_types())[:14]
        flows = flows[flows.highway.isin(highway_to_plot)]
        
        if counts is not None and distance_to_border>0:
            border_points = counts.get_near_border_ids(distance_to_border)
            flows = flows[~flows["id"].isin(border_points["id"])]
            logger.info(f"Points within {distance_to_border} meters to borders are excluded from bars plot!")
        
        # Count the number of entries per highway
        highway_counts = flows['highway'].value_counts().to_dict()
        
        # Compute averages and reshape for plotting
        df_avg = flows.groupby('highway')[['flow', 'simulated_flow']].mean().reset_index()
        df_melted = df_avg.melt(id_vars='highway', 
                                var_name='Flow Type', 
                                value_name='Average Flow')
        
        df_melted = df_melted.sort_values("Average Flow", ascending=False)
        # Plot
        plt.figure(figsize=(12,6))
        ax = sns.barplot(data=df_melted, x='highway', y='Average Flow', hue='Flow Type')
        
        # Add counts above the group of bars for each highway
        # Get unique highway types in the same order as x-axis
        highway_order = df_avg['highway'].tolist()
        xticks = ax.get_xticks()
        
        # Add text labels
        for tick, highway in zip(xticks, highway_order):
            count = highway_counts.get(highway, 0)
            ax.text(tick,             # x-position (center of group)
                    df_melted[df_melted['highway'] == highway]['Average Flow'].max() + 0.05,  # y-position just above highest bar
                    f'n={count}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Customize plot        
        plt.title("Average Observed and Simulated Flow by Highway Type" if title is None else title, 
                  fontsize=17, pad = 12)
        plt.xlabel("Highway Type", fontsize=15, labelpad = 12)
        plt.ylabel("Average Flow", fontsize=15, labelpad = 12)
        plt.xticks(rotation=30)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=15)
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
        plt.close()
                

        
    
    def plot_network_with_counts(self, counts: Counts = None, 
                                       matched: gpd.GeoDataFrame = None, 
                                       network: Network=None,
                                       output: str = None, 
                                       lw:float=1,
                                       markersize:float=6, 
                                       figsize:tuple=(50, 50),
                                       dpi:int=200, 
                                       return_matched_links:bool = False,
                                       road_types = ['motorway', 'trunk', 'primary', 
                                                     'motorway_link', 'trunk_link',
                                                     'primary_link', 'secondary', 
                                                     'secondary_link', 'tertiary'],
                                       cut:bool = False,
                                       highlight_stations=None):
             
        if road_types=="all":
            roads = network.get_geometry()
        elif road_types=="default":
            road_types = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link',
                          'primary_link', 'secondary', 'secondary_link']
            roads = network.get_ways(road_types=road_types)
        else:
            roads = network.get_ways(road_types=road_types)
        
        if cut:
            west,south,east,north = counts.counts.total_bounds              
            roads = roads.cx[west-200:east+200, south-200:north+200]                    
        
        fig, ax = plt.subplots(figsize=figsize)
        roads.plot(ax=ax, color='gray', linewidth=lw)
        
        if not matched is None:
            matched_ids = matched.link_id.unique()
            matched_links = network.net_geo[network.net_geo.link_id.isin(matched_ids)]
            
            if network is not None:
                replicates = network.links.loc[ network.links.link_id.isin(matched_ids) & 
                                                network.links.replicate_of.notna(), 
                                                "replicate_of"
                                               ].unique()
                old_ids = network.links.loc[ network.links.link_id.isin(replicates), 
                                             "attributes"
                                            ].map(lambda x: x.get("old_link_id", "")).str.split("_")
                
                replicate_ids = pd.Series([item for sublist in old_ids if isinstance(sublist, list) for item in sublist]).unique()

                # Combine original and replicate IDs
                all_ids = np.union1d(matched_ids, replicate_ids)
                matched_links = network.net_geo[network.net_geo.link_id.isin(all_ids)]
                
            matched_links.plot(ax=ax, color='blue', linewidth=2*lw)
            
        if counts is not None:
            counts.counts.plot(ax=ax, color='red', markersize=markersize)
            if highlight_stations is not None:
                stations = counts.counts[counts.counts['id'].isin(highlight_stations)]
                stations.plot(ax=ax, color='lime', markersize=3*markersize, alpha=0.4)

        plt.axis('off')
        if output is not None:
            plt.savefig(output, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        if return_matched_links:
            return matched_links
    
    @staticmethod
    def extract_coords(geom):
        if geom.geom_type == "LineString":
            return list(geom.coords)
        elif geom.geom_type == "MultiLineString":
            # Flatten each LineString within the MultiLineString
            return [coord for line in geom.geoms for coord in line.coords]
        else:
            return []  # fallback
    
    @staticmethod    
    def create_map(df:Union[gpd.GeoDataFrame,list],
                   data_to_show=["link_id"], 
                   point_gdf: Union[gpd.GeoDataFrame,list] = None,
                   point_data_to_show:list=[],
                   border: gpd.GeoDataFrame =  None,
                   path_to_save=None):
        
        #Make points as list
        if not isinstance(point_gdf, list):
            point_gdf = [point_gdf]
        if not isinstance(df, list):
            df = [df]            
            
        # Copy and reset index
        df = [dfi.copy().reset_index(drop=True) for dfi in df]
        
        # Ensure the DataFrame contains either 'path' or 'geometry'
        for dfi in df:
            if "path" not in dfi:
                if "geometry" in dfi:
                    dfi["path"] = dfi.geometry.apply(Plotter.extract_coords)
                else:
                    raise Exception("Dataframe should contain 'path' or 'geometry'.")
        
        # Filter the DataFrame to include only relevant columns
        layers = []
        for i, dfi in enumerate(df):
            dfi = dfi[["path", *data_to_show]]
            dfi = dfi[dfi.path.apply(len)>0]
            # Define the PathLayer
            path_layer = pdk.Layer(
                "PathLayer",
                dfi,
                pickable=True,
                auto_highlight=True,      
                get_path="path",
                get_width=1.5,
                get_color=[0,0,255] if i==0 else [199, 21, 133],  
                highlight_color=[255, 0, 0],      
                width_min_pixels=1 if i==0 else 2,
            )
            
            layers.append(path_layer)
        
        # Add optional point layer
        def color_map(diff,min_diff,max_diff):
            vmax = max(abs(min_diff), abs(max_diff))
            vmin = -vmax           
            diff_clipped = np.clip(diff, vmin, vmax)
            norm = (diff_clipped - vmin) / (vmax - vmin)
            colormap = plt.get_cmap('bwr') 
            color = colormap(norm)
            return [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255]  # RGBA format

        if point_gdf is not None and len(point_gdf):
            colors = [[255, 0, 0], [0, 102, 51], [128, 0, 128]]
            for i, gdf in enumerate(point_gdf):
                if not gdf.empty:
                    gdf = gdf.copy().reset_index(drop=True)
                    gdf["coordinates"] = gdf.geometry.apply(lambda geom: [geom.x, geom.y])
                    col_to_show = "pdiff" if "pdiff" in gdf else "adiff"
                    if col_to_show in gdf:
                        gdf = gdf[gdf[col_to_show].notna()]
                        min_diff = max(gdf[col_to_show].min(),-100) if col_to_show=="pdiff" else gdf[col_to_show].min()
                        max_diff = min(gdf[col_to_show].max(),100) if col_to_show=="pdiff" else gdf[col_to_show].max()
                        
                        gdf["color"] = gdf[col_to_show].apply(lambda x: color_map(x,min_diff,max_diff))
                        
                    point_layer = pdk.Layer("ScatterplotLayer",
                                            gdf,
                                            pickable=True,
                                            get_position="coordinates",
                                            get_color="color" if "color" in gdf else colors[i],
                                            get_radius=2,           # meters — adjust as needed                                    
                                            radius_min_pixels=6       # prevents total disappearance
                                            )       
                    layers.append(point_layer)
        
        # plot the border
        if border is not None:            
            border["coordinates"] = border.geometry.exterior.apply(lambda x: list(x.coords))
            polygon_layer = pdk.Layer( "PolygonLayer",
                                        border,
                                        get_polygon="coordinates",
                                        get_fill_color=[0, 0, 0, 0],  # Semi-transparent green
                                        stroked=True,
                                        get_line_color=[0, 0, 0],
                                        line_width_min_pixels=2,
                                        pickable=False)
            layers.append(polygon_layer)
            
        # Calculate view center (centroid of all paths)
        if "geometry" in df[0]:
            mean_y = df[0].geometry.centroid.y.mean()
            mean_x = df[0].geometry.centroid.x.mean()
        else:
            mean_x = df[0].path.apply(lambda x: x[0][0]).mean()
            mean_y = df[0].path.apply(lambda x: x[0][1]).mean()
    
        view_state = pdk.ViewState(
            longitude=mean_x,
            latitude=mean_y,
            zoom=12,
            pitch=0,
        )
    
        # Dynamically construct the tooltip HTML
        tooltip_html = "<table>"
        for field in data_to_show:
            tooltip_html += f"<tr><td><b>{field.capitalize()}:</b></td><td>{{{field}}}</td></tr>"
        for field in point_data_to_show:
            tooltip_html += f"<tr><td><b>{field.capitalize()}:</b></td><td>{{{field}}}</td></tr>"
        tooltip_html += "</table>"
    
        # Tooltip configuration
        tooltip = {
            "html": tooltip_html,
            "style": {"color": "white", "background-color": "black", "padding": "10px"}
        }
    
        # Use OpenStreetMap street-style background (Voyager)
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
        )
    
        # Save the map as an HTML file
        if path_to_save:
            r.to_html(path_to_save, notebook_display=False)
            logger.info(f"Map saved to {path_to_save}")
        else:
            return r
        

        
        
        
        