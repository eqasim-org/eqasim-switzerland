#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:08:31 2025

@author: dabdelkader
"""

import numpy as np
import pandas as pd

def compute_bearing(geom):
    if geom is None or geom.is_empty:
        return np.nan
    # Get start and end coordinates
    start = geom.coords[0]
    end = geom.coords[-1]
    # Convert to radians
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    angle = np.arctan2(delta_x, delta_y)
    bearing = np.degrees(angle)
    # Normalize to [0, 360)
    return (bearing + 360) % 360


def angular_diff(a, b):
    """Calculate the smallest angular difference between two bearings."""
    return np.abs((a - b + 180) % 360 - 180)

def get_direction_based_on_bearing(bearing, d1, d2):
    if np.isnan(d1):
        return "direction2" if abs(angular_diff(bearing, d2))<90 else "direction1"
    
    if np.isnan(d2):
        return "direction1" if abs(angular_diff(bearing, d1))<90 else "direction2"  
      
    if not pd.isna(d1) and not pd.isna(d2):
        d = 1 if angular_diff(bearing, d1) <= angular_diff(bearing, d2) else 2
        return f"direction{d}"

def assign_bearing_direction(df):
    result_rows = []

    for _, group in df.groupby('id'):
        if len(group) == 1:
            row = group.iloc[0].copy()
            row['direction'] = get_direction_based_on_bearing(row['bearing'], row['bearing_direction1'], row['bearing_direction2'])
            result_rows.append(row)
                
        elif len(group) == 2:
            rows = group.to_dict(orient='records')
            
            b1 = rows[0]['bearing']
            d1_1 = rows[0]['bearing_direction1']
            d1_2 = rows[0]['bearing_direction2']
            
            if np.isnan(d1_1):
                b2 = rows[1]['bearing'] 
                dir1 = "direction2" if angular_diff(b1,d1_2)<angular_diff(b2,d1_2) else "direction1"
            elif np.isnan(d1_2):
                b2 = rows[1]['bearing'] 
                dir1 = "direction1" if angular_diff(b1,d1_1)<angular_diff(b2,d1_1) else "direction2"
            else:
                dir1 = get_direction_based_on_bearing(b1, d1_1, d1_2)
            
            
            dir2 = "direction2" if dir1=="direction1" else "direction1" 

            rows[0]['direction'] = dir1
            rows[1]['direction'] = dir2
            result_rows.extend(rows)

        # Skip groups with more than 2 rows
        else:
            continue

    return pd.DataFrame(result_rows)







    

def is_opposite_direction(b1, b2, tolerance=20):
    diff = abs(b1 - b2)
    diff = min(diff, 360 - diff)
    return abs(diff - 180) <= tolerance


def tag_opposite_directions(df_group):
    if len(df_group) != 2:
        return pd.Series({"opposite_direction": False})
    b1 = df_group.iloc[0]["bearing"]
    b2 = df_group.iloc[1]["bearing"]
    return pd.Series({"opposite_direction": is_opposite_direction(b1, b2)})


def get_best_opposite_pair(group):
    if len(group) < 2:
        return group

    pairs = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            b1, b2 = group.iloc[i]["bearing"], group.iloc[j]["bearing"]
            if is_opposite_direction(b1, b2):
                avg_dist = (group.iloc[i]["distance"] + group.iloc[j]["distance"]) / 2
                pairs.append((i, j, avg_dist))

    if not pairs:
        # No opposite pairs — return the single closest link
        return group.loc[[group["distance"].idxmin()]]

    # Pick the best (closest average-distance) opposite pair
    best_i, best_j, _ = min(pairs, key=lambda x: x[2])
    return group.iloc[[best_i, best_j]]

def get_minimum_distance_match(group):
    if len(group) < 2:
        return group
    if "angle" in group:
        group = group.copy()
        group["distance"] = (group["distance"] / 2).round() * 2
        group["angle_diff"] = np.abs(group.angle-group.bearing)        
        return group.nsmallest(1, ['distance', 'angle_diff'])
    
    return group.loc[[group["distance"].idxmin()]]





























###################### DISTANCES PART ############################
from shapely.geometry import LineString, MultiLineString
import numpy as np
from scipy.spatial.distance import cdist
from shapely import line_interpolate_point


def get_distances(line1, line2, num_steps=10):
    """
    Sample points along line1 and compute distance to line2.
    Supports LineString and MultiLineString.
    """
    def flatten_geometry(geom):
        if isinstance(geom, LineString):
            return [geom]
        elif isinstance(geom, MultiLineString):
            return list(geom.geoms)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geom)}")

    lines1 = flatten_geometry(line1)
    lines2 = flatten_geometry(line2)

    # Combine all geometries into a single collection for nearest neighbor search
    points = []
    for line in lines1:
        length = line.length
        if length == 0:
            continue
        for i in range(num_steps):
            point = line.interpolate(i / num_steps, normalized=True)
            points.append(point)

    if not points:
        return np.array([])

    # Compute distances one-by-one (no vectorization in Shapely 1.x)
    distances = [line2.distance(pt) for pt in points]
    return np.array(distances)


def asymmetric_min_hausdorff_geometry(A, B, num_steps=10):
    """
    Compute asymmetric min Hausdorff distance between two geometries.
    Returns mean of sampled distances in both directions and returns the minimum.
    """
    dist_A_to_B = get_distances(A, B, num_steps)
    dist_B_to_A = get_distances(B, A, num_steps)

    if len(dist_A_to_B) == 0 or len(dist_B_to_A) == 0:
        return float('inf')

    mean_A_to_B = dist_A_to_B.mean()
    mean_B_to_A = dist_B_to_A.mean()

    return min(mean_A_to_B, mean_B_to_A)


# Get distance
def unit_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def angle_between_vectors_deg(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)

def get_start_end_vector(geometry):
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        start = np.array(coords[0])
        end = np.array(coords[-1])
        return end - start
    
    elif isinstance(geometry, MultiLineString):
        if not geometry.geoms:
            raise ValueError("MultiLineString is empty")
        first_line = geometry.geoms[0]
        last_line = geometry.geoms[-1]
        start = np.array(first_line.coords[0])
        end = np.array(last_line.coords[-1])
        return end - start
    
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")
        
def is_parallel(line1, line2, tolerance = 20):
    #TODO: add the case when line1 is in V form
    vec1 = get_start_end_vector(line1)
    vec2 = get_start_end_vector(line2)

    angle_diff = angle_between_vectors_deg(vec1, vec2)

    if angle_diff<tolerance:
        return "same" 
    elif angle_diff>180-tolerance:
        return "opposite"
    else:
        return "not parallel"



