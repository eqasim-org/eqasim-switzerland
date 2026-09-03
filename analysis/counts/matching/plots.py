#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:34:19 2025

@author: dabdelkader
"""

import json
from html import escape
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns        
import pandas as pd
import geopandas as gpd
from .counts import Counts
from .network import RoadNetwork
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pydeck as pdk
import folium
from typing import Union
from sklearn.metrics import r2_score
from .road_matching import ROAD_TYPE_PRIORITY
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection, LineString, MultiLineString

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FLOW_MAP_TOOLTIP_FIELDS = [
    "id",
    "matched_link_ids",
    "flow",
    "simulated_flow",
    "pdiff",
    "adiff",
    "geh",
]
FLOW_MAP_FIELD_LABELS = {
    "link_id": "Link ID",
    "id": "Station",
    "matched_link_ids": "Matched link ID(s)",
    "flow": "Traffic count (vehicles/day)",
    "simulated_flow": "Simulated flow (vehicles/day)",
    "pdiff": "Percentage difference (%)",
    "adiff": "Absolute difference (vehicles/day)",
    "geh": "GEH",
}



def GEH(x_d, y_d, return_vector=False, directions_represented=2):
    """Compute GEH from daily flows using the represented direction count.

    ``directions_represented`` is 1 for directional observations (for example,
    Geneva and Zurich) and 2 for observations aggregated across both road
    directions. It may be either a scalar or one value per observation.
    """
    directions = np.asarray(directions_represented, dtype=float)
    if np.any(~np.isfinite(directions)) or np.any(directions <= 0):
        raise ValueError("directions_represented must contain positive values")

    x = np.asarray(x_d, dtype=float) / 24 / directions
    y = np.asarray(y_d, dtype=float) / 24 / directions
    geh_values = np.sqrt(2 * (x - y) ** 2 / (x + y + 1e-6))
    if return_vector:
        return geh_values
    geh_within_5 = int(np.sum(geh_values <= 5))
    geh_within_10 = int(np.sum(geh_values <= 10))
    geh_within_15 = int(np.sum(geh_values <= 15))
    geh_within_25 = int(np.sum(geh_values <= 25))
    n_points = len(geh_values)
    geh_within_5_pct = (geh_within_5 / n_points) * 100
    geh_within_10_pct = (geh_within_10 / n_points) * 100
    geh_within_15_pct = (geh_within_15 / n_points) * 100
    geh_within_25_pct = (geh_within_25 / n_points) * 100
    return geh_within_5_pct, geh_within_10_pct, geh_within_15_pct, geh_within_25_pct

def SGV(x_d,y_d):
    f = 10_000
    n_points = len(x_d)
    x = x_d / 2 # this is because these counts or for two directions
    y = y_d / 2 # this is because these counts or for two directions
    sqv = 1/( 1 + np.sqrt(  (y-x)**2/(f*x)  ) )
    sqv_09_pct = round(np.sum(sqv >= 0.9) / n_points * 100, 1)
    sqv_085_pct = round(np.sum(sqv >= 0.85) / n_points * 100, 1)
    sqv_08_pct = round(np.sum(sqv >= 0.8) / n_points * 100, 1)
    sqv_07_pct = round(np.sum(sqv >= 0.7) / n_points * 100, 1)
    return sqv_09_pct, sqv_085_pct, sqv_08_pct, sqv_07_pct

class Plotter:
    FLOW_MAP_TOOLTIP_FIELDS = FLOW_MAP_TOOLTIP_FIELDS

    @staticmethod
    def prepare_flow_map_points(points, flows, directions_represented=None):
        """Combine station locations and comparison metrics for an HTML map."""
        required = {"id", "flow", "simulated_flow", "pdiff", "adiff"}
        missing = required.difference(flows.columns)
        if missing:
            raise ValueError(
                "Flow map data is missing required columns: "
                + ", ".join(sorted(missing))
            )

        flow_columns = ["id", "flow", "simulated_flow", "pdiff", "adiff"]
        for optional_column in ("directions_represented", "matched_link_ids", "link_id"):
            if optional_column in flows.columns:
                flow_columns.append(optional_column)
        result = points[["id", "geometry"]].merge(
            flows[flow_columns],
            on="id",
            how="inner",
        )
        if "matched_link_ids" not in result.columns:
            if "link_id" in result.columns:
                result["matched_link_ids"] = result["link_id"].map(
                    lambda value: list(value)
                    if isinstance(value, (list, tuple, np.ndarray))
                    else [value]
                )
            else:
                result["matched_link_ids"] = [[] for _ in range(len(result))]

        if directions_represented is None:
            directions_represented = (
                result["directions_represented"]
                if "directions_represented" in result.columns
                else 2
            )
        result["geh"] = GEH(
            result["flow"],
            result["simulated_flow"],
            return_vector=True,
            directions_represented=directions_represented,
        ).round(2)
        return gpd.GeoDataFrame(result, geometry="geometry", crs=points.crs)

    def plot_flow(self, flows, counts:Counts=None, output_file:str=None, 
                        distance_to_border:int=5000, title:str=None, show_range=False, show_geh=False,
                        remove_near_border=False, directions_represented=None):
        flows = flows.copy()
        flows = flows.sort_values("flow")
        
        if show_range:
            flows = flows.merge(counts.counts[['id', 'flow_lower', 'flow_upper']], on="id",how="left")
            flow_lower = flows.flow_lower
            flow_upper = flows.flow_upper
                            
        x = flows.flow
        y = flows.simulated_flow

        if remove_near_border:
            border_points = counts.get_near_border_ids(distance_to_border)
            flows = flows[~flows["id"].isin(border_points["id"])]
            x = flows.flow
            y = flows.simulated_flow
            logger.info(f"Points within {distance_to_border} meters to borders are removed from flow comparison plot!")

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
        
        include_border = False
        if counts is not None and distance_to_border>0:
            border_points = counts.get_near_border_ids(distance_to_border)
            border_flow = flows[flows["id"].isin(border_points["id"])]
            if not border_flow.empty:
                plt.scatter(border_flow.flow, border_flow.simulated_flow , 
                            alpha=0.6, s=10, c='orange', label=f"Within {distance_to_border} m to borders")            
                # Optional: linear trend line (regression)
                in_flow = flows[~flows["id"].isin(border_points["id"])].reset_index(drop=True)
                slope = np.sum(in_flow.flow * in_flow.simulated_flow) / np.sum(in_flow.flow * in_flow.flow) 
                
                plt.plot(x, slope * x, color='darkmagenta', lw=2, linestyle='-', label=f'Trend without borders: y={slope:.2f} x')
                r2_in = r2_score(in_flow.flow, in_flow.simulated_flow)
                include_border = True
            
        # Add R2 Score
        plt.text( 0.02 * max_val, 0.6 * max_val, 
                 f"$R^2$ = {r2:.3f}\n$R^2_{{\\mathrm{{in}}}}$ = {r2_in:.3f}" if include_border else f"$R^2$ = {r2:.3f}",            # text
                 fontsize=14,
                 color='crimson',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')  # white box
                 )
        
        # Add GEH statistics
        if show_geh:
            if directions_represented is None:
                directions = (
                    flows.loc[x.index, "directions_represented"]
                    if "directions_represented" in flows.columns
                    else 2
                )
            else:
                directions = directions_represented
            geh = GEH(x, y, directions_represented=directions)
            plt.text( 0.7 * max_val, 0.02 * max_val, 
                    f"GEH ≤ 5: {geh[0]:.1f}%\nGEH ≤ 10: {geh[1]:.1f}%\nGEH ≤ 15: {geh[2]:.1f}%\nGEH ≤ 25: {geh[3]:.1f}%" ,  # text
                    fontsize=14,
                    color='steelblue',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')  # white box
                    )
            
            sqv = SGV(x,y)
            plt.text( 0.7 * max_val, 0.25 * max_val, 
                    f"SQV ≥ 0.9: {sqv[0]}%\nSQV ≥ 0.85: {sqv[1]}%\nSQV ≥ 0.8: {sqv[2]}%\nSQV ≥ 0.7: {sqv[3]}%" ,  # text
                    fontsize=14,
                    color='darkmagenta',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')  # white box
                    )
        
        # Axis labels and title
        plt.xlabel("Observed Flow (Weekday Avg)", fontsize=15, labelpad=13)
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
        
        highway_to_plot = ROAD_TYPE_PRIORITY[:14]
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
                                       network: RoadNetwork=None,
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
        
        if matched is not None:
            matched_ids = matched.link_id.unique()
            matched_links = network.get_link_geometries(
                matched_ids, expand_merged=True
            )
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
        if isinstance(geom, LineString):
            return list(geom.coords)
        return []

    @staticmethod
    def extract_paths(geom):
        """Return independent paths without drawing connectors between parts."""
        if isinstance(geom, LineString):
            return [list(geom.coords)]
        if isinstance(geom, (MultiLineString, GeometryCollection)):
            return [
                path
                for part in geom.geoms
                for path in Plotter.extract_paths(part)
            ]
        return []

    @staticmethod
    def _prepare_path_data(data, data_to_show):
        if "path" in data:
            prepared = data[["path", *data_to_show]].copy()
            return prepared[prepared.path.map(len) > 1].reset_index(drop=True)
        if "geometry" not in data:
            raise ValueError("Dataframe should contain 'path' or 'geometry'.")

        if data.geometry.geom_type.eq("LineString").all():
            prepared = data[[*data_to_show]].copy()
            prepared.insert(
                0, "path", data.geometry.map(lambda geometry: list(geometry.coords))
            )
            return prepared[prepared.path.map(len) > 1].reset_index(drop=True)

        records = []
        for _, row in data[["geometry", *data_to_show]].iterrows():
            properties = {field: row[field] for field in data_to_show}
            for path in Plotter.extract_paths(row.geometry):
                if len(path) > 1:
                    records.append({"path": path, **properties})
        return pd.DataFrame(records, columns=["path", *data_to_show])

    @staticmethod
    def _tooltip_value(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            formatted = [Plotter._tooltip_value(item) for item in value]
            formatted = [item for item in formatted if item is not None]
            return ", ".join(formatted) if formatted else None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return escape(str(value))

    @staticmethod
    def _build_tooltip_html(row, fields):
        rows = []
        for field in fields:
            if field not in row:
                continue
            value = Plotter._tooltip_value(row[field])
            if value is None:
                continue
            label = FLOW_MAP_FIELD_LABELS.get(
                field, field.replace("_", " ").title()
            )
            rows.append(f"<tr><td><b>{escape(label)}:</b></td><td>{value}</td></tr>")
        return "<table>" + "".join(rows) + "</table>"
    @staticmethod
    def _flow_metric_settings(point_gdf):
        metric_columns = {"pdiff", "adiff", "geh"}
        valid_points = [
            gdf for gdf in point_gdf
            if gdf is not None and not gdf.empty
        ]
        if not valid_points or not all(
            metric_columns.issubset(gdf.columns) for gdf in valid_points
        ):
            return {}

        absolute_differences = pd.concat(
            [pd.to_numeric(gdf["adiff"], errors="coerce") for gdf in valid_points]
        ).abs()
        maximum = absolute_differences.max()
        if pd.isna(maximum) or maximum <= 0:
            absolute_bound = 1.0
        else:
            magnitude = 10 ** np.floor(np.log10(maximum))
            absolute_bound = float(np.ceil(maximum / magnitude) * magnitude)

        return {
            "pdiff": {
                "label": "Percentage difference (%)",
                "lower": -100.0,
                "upper": 100.0,
                "step": 1,
                "quality_inner": 10.0,
                "quality_outer": 20.0,
                "quality_unit": "%",
            },
            "adiff": {
                "label": "Absolute difference (vehicles/day)",
                "lower": -absolute_bound,
                "upper": absolute_bound,
                "step": max(1.0, absolute_bound / 100),
                "quality_inner": 1000.0,
                "quality_outer": 2000.0,
                "quality_unit": "vehicles/day",
            },
            "geh": {
                "label": "GEH",
                "lower": 0.0,
                "upper": 25.0,
                "step": 0.1,
                "quality_inner": 5.0,
                "quality_outer": 10.0,
                "quality_unit": "",
            },
        }

    @staticmethod
    def _inject_flow_metric_controls(path_to_save, metric_settings):
        if not metric_settings:
            return

        controls = """
<div id="counts-map-controls">
  <div class="counts-control-title">Count point colors</div>
  <label for="counts-color-metric">Metric</label>
  <select id="counts-color-metric">
    <option value="pdiff">Percentage difference (%)</option>
    <option value="adiff">Absolute difference (vehicles/day)</option>
    <option value="geh">GEH</option>
  </select>
  <div class="counts-bound-row">
    <div class="counts-bound-group">
      <label for="counts-lower-bound">Lower</label>
      <input id="counts-lower-bound" type="number">
    </div>
    <div class="counts-bound-group">
      <label for="counts-upper-bound">Upper</label>
      <input id="counts-upper-bound" type="number">
    </div>
  </div>
  <div class="counts-color-ramp"></div>
  <div class="counts-ramp-labels">
    <span id="counts-lower-label"></span>
    <span id="counts-upper-label"></span>
  </div>
  <div id="counts-bound-error">Lower bound must be smaller than upper bound.</div>
  <div class="counts-quality-section">
    <div class="counts-quality-title">Accuracy centers</div>
    <label for="counts-quality-metric">Center metric</label>
    <select id="counts-quality-metric">
      <option value="pdiff">Absolute percentage difference (%)</option>
      <option value="adiff">Absolute flow difference (vehicles/day)</option>
      <option value="geh">GEH</option>
    </select>
    <div class="counts-quality-row">
      <label><input id="counts-show-green" type="checkbox" checked> Green</label>
      <label for="counts-green-threshold">Below</label>
      <input id="counts-green-threshold" type="number" min="0">
      <span id="counts-green-unit"></span>
    </div>
    <div class="counts-quality-row">
      <label><input id="counts-show-gold" type="checkbox" checked> Gold</label>
      <label for="counts-gold-threshold">Below</label>
      <input id="counts-gold-threshold" type="number" min="0">
      <span id="counts-gold-unit"></span>
    </div>
    <div id="counts-quality-error">Thresholds must be non-negative numbers.</div>
  </div>
</div>
"""
        style = """
<style>
#counts-map-controls {
  position: absolute;
  z-index: 20;
  top: 12px;
  right: 12px;
  width: min(400px, calc(100vw - 24px));
  box-sizing: border-box;
  padding: 12px;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.28);
  color: #222;
  font: 13px/1.35 Arial, sans-serif;
}
#counts-map-controls .counts-control-title {
  margin-bottom: 8px;
  font-size: 15px;
  font-weight: 700;
}
#counts-map-controls select {
  width: 100%;
  box-sizing: border-box;
  margin: 3px 0 9px;
  padding: 5px;
}
#counts-map-controls .counts-bound-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
#counts-map-controls .counts-bound-group {
  min-width: 0;
}
#counts-map-controls .counts-bound-group label {
  display: block;
  margin-bottom: 3px;
}
#counts-map-controls input[type="number"] {
  min-width: 0;
  width: 100%;
  box-sizing: border-box;
  padding: 4px;
}
#counts-map-controls .counts-color-ramp {
  height: 13px;
  margin-top: 10px;
  border: 1px solid #777;
  background: linear-gradient(to right, rgb(33,102,172), rgb(247,247,247), rgb(178,24,43));
}
#counts-map-controls .counts-ramp-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}
#counts-bound-error {
  display: none;
  margin-top: 6px;
  color: #b2182b;
  font-size: 11px;
}
#counts-map-controls .counts-quality-section {
  margin-top: 10px;
  padding-top: 9px;
  border-top: 1px solid #bbb;
}
#counts-map-controls .counts-quality-title {
  margin-bottom: 6px;
  font-weight: 700;
}
#counts-map-controls .counts-quality-row {
  display: grid;
  grid-template-columns: minmax(78px, 1fr) auto minmax(58px, 72px) auto;
  align-items: center;
  gap: 5px;
  margin-top: 5px;
}
#counts-map-controls .counts-quality-row > label:first-child {
  white-space: nowrap;
}
#counts-quality-error {
  display: none;
  margin-top: 6px;
  color: #b2182b;
  font-size: 11px;
}
</style>
"""
        script = r"""
<script>
(function () {
  const metricSettings = __METRIC_SETTINGS__;
  const select = document.getElementById("counts-color-metric");
  const lowerInput = document.getElementById("counts-lower-bound");
  const upperInput = document.getElementById("counts-upper-bound");
  const lowerLabel = document.getElementById("counts-lower-label");
  const upperLabel = document.getElementById("counts-upper-label");
  const error = document.getElementById("counts-bound-error");
  const showGreen = document.getElementById("counts-show-green");
  const showGold = document.getElementById("counts-show-gold");
  const greenThresholdInput = document.getElementById("counts-green-threshold");
  const goldThresholdInput = document.getElementById("counts-gold-threshold");
  const qualityError = document.getElementById("counts-quality-error");
  const qualityMetricSelect = document.getElementById("counts-quality-metric");
  const greenUnit = document.getElementById("counts-green-unit");
  const goldUnit = document.getElementById("counts-gold-unit");
  const deck = typeof deckInstance === "undefined" ? null : deckInstance;
  const bounds = JSON.parse(JSON.stringify(metricSettings));
  const qualityBounds = JSON.parse(JSON.stringify(metricSettings));
  let activeMetric = select.value;
  let activeQualityMetric = qualityMetricSelect.value;

  function interpolate(first, second, fraction) {
    return first.map(function (value, index) {
      return Math.round(value + (second[index] - value) * fraction);
    });
  }

  function colorFor(value, lower, upper) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return [128, 128, 128, 180];
    }
    const clipped = Math.max(lower, Math.min(upper, numeric));
    const position = (clipped - lower) / (upper - lower);
    const blue = [33, 102, 172, 230];
    const white = [247, 247, 247, 230];
    const red = [178, 24, 43, 230];
    return position <= 0.5
      ? interpolate(blue, white, position * 2)
      : interpolate(white, red, (position - 0.5) * 2);
  }

  function loadBounds() {
    const setting = bounds[activeMetric];
    lowerInput.value = setting.lower;
    upperInput.value = setting.upper;
    lowerInput.step = setting.step;
    upperInput.step = setting.step;
    lowerLabel.textContent = setting.lower;
    upperLabel.textContent = setting.upper;
  }

  function loadQualityBounds() {
    const setting = qualityBounds[activeQualityMetric];
    greenThresholdInput.value = setting.quality_outer;
    goldThresholdInput.value = setting.quality_inner;
    greenThresholdInput.step = setting.step;
    goldThresholdInput.step = setting.step;
    greenUnit.textContent = setting.quality_unit;
    goldUnit.textContent = setting.quality_unit;
  }

  function recolor() {
    if (!deck || !deck.props || !deck.props.layers) {
      return;
    }
    const lower = Number(lowerInput.value);
    const upper = Number(upperInput.value);
    const valid = Number.isFinite(lower) && Number.isFinite(upper) && lower < upper;
    error.style.display = valid ? "none" : "block";
    if (!valid) {
      return;
    }

    bounds[activeMetric].lower = lower;
    bounds[activeMetric].upper = upper;
    lowerLabel.textContent = lower;
    upperLabel.textContent = upper;

    const layers = deck.props.layers.map(function (layer) {
      if (!layer.id.startsWith("counts-points-")) {
        return layer;
      }
      return layer.clone({
        getFillColor: function (datum) {
          return colorFor(datum[activeMetric], lower, upper);
        },
        updateTriggers: {
          getFillColor: [activeMetric, lower, upper]
        }
      });
    });
    deck.setProps({layers: layers});
  }

  function updateQualityMarkers() {
    if (!deck || !deck.props || !deck.props.layers) {
      return;
    }
    const greenThreshold = Number.parseFloat(greenThresholdInput.value);
    const goldThreshold = Number.parseFloat(goldThresholdInput.value);
    const valid = Number.isFinite(greenThreshold) && greenThreshold >= 0
      && Number.isFinite(goldThreshold) && goldThreshold >= 0;
    qualityError.style.display = valid ? "none" : "block";
    if (!valid) {
      return;
    }

    qualityBounds[activeQualityMetric].quality_outer = greenThreshold;
    qualityBounds[activeQualityMetric].quality_inner = goldThreshold;

    const layers = deck.props.layers.map(function (layer) {
      let threshold;
      let enabled;
      let color;
      if (layer.id.startsWith("counts-quality-green-")) {
        threshold = greenThreshold;
        enabled = showGreen.checked;
        color = [0, 170, 90, 255];
      } else if (layer.id.startsWith("counts-quality-gold-")) {
        threshold = goldThreshold;
        enabled = showGold.checked;
        color = [255, 193, 7, 255];
      } else {
        return layer;
      }
      return layer.clone({
        getFillColor: function (datum) {
          const difference = Number(datum[activeQualityMetric]);
          return enabled && Number.isFinite(difference)
            && Math.abs(difference) < threshold
            ? color
            : [0, 0, 0, 0];
        },
        updateTriggers: {
          getFillColor: [activeQualityMetric, enabled, threshold]
        }
      });
    });
    deck.setProps({layers: layers});
  }

  select.addEventListener("change", function () {
    activeMetric = select.value;
    loadBounds();
    recolor();
  });
  qualityMetricSelect.addEventListener("change", function () {
    activeQualityMetric = qualityMetricSelect.value;
    loadQualityBounds();
    updateQualityMarkers();
  });
  lowerInput.addEventListener("input", recolor);
  upperInput.addEventListener("input", recolor);
  showGreen.addEventListener("change", updateQualityMarkers);
  showGold.addEventListener("change", updateQualityMarkers);
  greenThresholdInput.addEventListener("input", updateQualityMarkers);
  goldThresholdInput.addEventListener("input", updateQualityMarkers);

  loadBounds();
  loadQualityBounds();
  recolor();
  updateQualityMarkers();
})();
</script>
"""
        script = script.replace(
            "__METRIC_SETTINGS__", json.dumps(metric_settings, separators=(",", ":"))
        )

        path = Path(path_to_save)
        html = path.read_text(encoding="utf-8")
        html = html.replace("</head>", style + "\n</head>", 1)
        html = html.replace("</body>", controls + "\n</body>", 1)
        html = html.replace("</html>", script + "\n</html>", 1)
        path.write_text(html, encoding="utf-8")

    
    @staticmethod    
    def create_map(df:Union[gpd.GeoDataFrame,list],
                   data_to_show=["link_id"], 
                   point_gdf: Union[gpd.GeoDataFrame,list] = None,
                   point_data_to_show=None,
                   border: gpd.GeoDataFrame =  None,
                   path_to_save=None, 
                   cut_network = False):
        
        point_data_to_show = point_data_to_show or []

        # Make points and paths lists.
        if point_gdf is None:
            point_gdf = []
        elif not isinstance(point_gdf, list):
            point_gdf = [point_gdf]
        if not isinstance(df, list):
            df = [df]

        metric_settings = Plotter._flow_metric_settings(point_gdf)

        # cut df to point gdf (just plot the regional network
        if cut_network:
            geo_union = []
            for points in point_gdf:
                if points is not None and not points.empty:
                    geo_union.extend(points.geometry)

            if geo_union:
                geo_union = (gpd.GeoSeries([unary_union(geo_union)],crs="EPSG:4326")
                            .to_crs("EPSG:2056")
                            .buffer(1_000)
                            .to_crs("EPSG:4326"))
                xmin, ymin, xmax, ymax = geo_union.total_bounds
                for i in range(len(df)):
                    df[i] = df[i].cx[xmin:xmax, ymin:ymax]

        # Preserve every real vertex and keep multipart geometries as
        # independent paths rather than joining them with artificial chords.
        df = [Plotter._prepare_path_data(dfi, data_to_show) for dfi in df]
        for path_data in df:
            path_data["_tooltip_html"] = path_data.apply(
                lambda row: Plotter._build_tooltip_html(row, data_to_show), axis=1
            )

        layers = []
        for i, dfi in enumerate(df):
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
        
        # Add optional point layers. The initial view uses percentage
        # difference and browser controls can switch the accessor dynamically.
        def color_map(value, lower, upper):
            clipped = np.clip(value, lower, upper)
            normalized = (clipped - lower) / (upper - lower)
            color = plt.get_cmap("bwr")(normalized)
            return [
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
                230,
            ]

        initial_metric = "pdiff" if metric_settings else None
        if point_gdf:
            colors = [[255, 0, 0], [0, 102, 51], [128, 0, 128]]
            for i, gdf in enumerate(point_gdf):
                if not gdf.empty:
                    gdf = gdf.copy().reset_index(drop=True)
                    gdf["coordinates"] = gdf.geometry.apply(
                        lambda geometry: [geometry.x, geometry.y]
                    )
                    if initial_metric:
                        gdf = gdf[gdf[initial_metric].notna()].copy()
                        settings = metric_settings[initial_metric]
                        gdf["color"] = gdf[initial_metric].apply(
                            lambda value: color_map(
                                value, settings["lower"], settings["upper"]
                            )
                        )

                    gdf["_tooltip_html"] = gdf.apply(
                        lambda row: Plotter._build_tooltip_html(row, point_data_to_show), axis=1
                    )
                    point_layer = pdk.Layer(
                        "ScatterplotLayer",
                        gdf,
                        id=f"counts-points-{i}",
                        pickable=True,
                        get_position="coordinates",
                        get_fill_color="color" if "color" in gdf else colors[i],
                        get_radius=2,
                        radius_min_pixels=6,
                    )
                    layers.append(point_layer)

                    if "pdiff" in gdf.columns:
                        absolute_pdiff = pd.to_numeric(
                            gdf["pdiff"], errors="coerce"
                        ).abs()
                        transparent = [0, 0, 0, 0]
                        gdf["quality_green_color"] = [
                            [0, 170, 90, 255] if difference < 20 else transparent
                            for difference in absolute_pdiff
                        ]
                        gdf["quality_gold_color"] = [
                            [255, 193, 7, 255] if difference < 10 else transparent
                            for difference in absolute_pdiff
                        ]
                        layers.append(
                            pdk.Layer(
                                "ScatterplotLayer",
                                gdf,
                                id=f"counts-quality-green-{i}",
                                pickable=False,
                                get_position="coordinates",
                                get_fill_color="quality_green_color",
                                get_radius=1,
                                radius_min_pixels=4,
                                radius_max_pixels=4,
                            )
                        )
                        layers.append(
                            pdk.Layer(
                                "ScatterplotLayer",
                                gdf,
                                id=f"counts-quality-gold-{i}",
                                pickable=False,
                                get_position="coordinates",
                                get_fill_color="quality_gold_color",
                                get_radius=1,
                                radius_min_pixels=3,
                                radius_max_pixels=3,
                            )
                        )
        
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
            
        # Calculate the view center from all real vertices in the first layer.
        coordinates = [coordinate for path in df[0].path for coordinate in path]
        mean_x = np.mean([coordinate[0] for coordinate in coordinates])
        mean_y = np.mean([coordinate[1] for coordinate in coordinates])
    
        view_state = pdk.ViewState(
            longitude=mean_x,
            latitude=mean_y,
            zoom=12,
            pitch=0,
        )
    
        # Every layer provides only the rows applicable to its own objects.
        tooltip = {
            "html": "{_tooltip_html}",
            "style": {"color": "white", "background-color": "black", "padding": "10px"}
        }
    
        # Use OpenStreetMap street-style background (Voyager)
        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
        )
    
        # Save the map as an HTML file and add browser-side controls.
        if path_to_save:
            r.to_html(path_to_save, notebook_display=False)
            Plotter._inject_flow_metric_controls(path_to_save, metric_settings)
            logger.info(f"Map saved to {path_to_save}")
        else:
            return r
        

        
        
        
        
