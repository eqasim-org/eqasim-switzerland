#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 11:16:39 2025

@author: dabdelkader
"""

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from typing import Union



###################### Orientation ############################
class GeometryOrientation:
    @staticmethod
    def is_opposite_direction(b1, b2, tolerance=15):
        # diff = abs(b1 - b2)
        # diff = min(diff, 360 - diff)
        # return abs(diff - 180) <= tolerance
        return GeometryOrientation.angular_diff(b1,b2)>(180-tolerance)

    @staticmethod
    def tag_opposite_directions(df_group):
        if len(df_group) != 2:
            return pd.Series({"opposite_direction": False})
        b1 = df_group.iloc[0]["bearing"]
        b2 = df_group.iloc[1]["bearing"]
        return pd.Series({"opposite_direction": GeometryOrientation.is_opposite_direction(b1, b2)})

    @staticmethod
    def get_best_opposite_pair(group):
        if len(group) < 2:
            return group

        pairs = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                b1, b2 = group.iloc[i]["bearing"], group.iloc[j]["bearing"]
                if GeometryOrientation.is_opposite_direction(b1, b2):
                    avg_dist = (group.iloc[i]["distance"] + group.iloc[j]["distance"]) / 2
                    pairs.append((i, j, avg_dist))

        if not pairs:
            # No opposite pairs — return the single closest link
            return group.loc[[group["distance"].idxmin()]]

        # Pick the best (closest average-distance) opposite pair
        best_i, best_j, _ = min(pairs, key=lambda x: x[2])
        return group.iloc[[best_i, best_j]]    
    
    @staticmethod
    def get_direction_based_on_bearing(bearing, d1, d2):
        if np.isnan(d1):
            return "direction2" if abs(GeometryOrientation.angular_diff(bearing, d2))<90 else "direction1"
        
        if np.isnan(d2):
            return "direction1" if abs(GeometryOrientation.angular_diff(bearing, d1))<90 else "direction2"  
          
        if not pd.isna(d1) and not pd.isna(d2):
            d = 1 if GeometryOrientation.angular_diff(bearing, d1) <= GeometryOrientation.angular_diff(bearing, d2) else 2
            return f"direction{d}"
    
    @staticmethod
    def assign_bearing_direction(df):
        result_rows = []

        for _, group in df.groupby('id'):
            if len(group) == 1:
                row = group.iloc[0].copy()
                row['direction'] = GeometryOrientation.get_direction_based_on_bearing(row['bearing'], row['bearing_direction1'], row['bearing_direction2'])
                result_rows.append(row)
                    
            elif len(group) == 2:
                rows = group.to_dict(orient='records')
                
                b1 = rows[0]['bearing']
                d1_1 = rows[0]['bearing_direction1']
                d1_2 = rows[0]['bearing_direction2']
                
                if np.isnan(d1_1):
                    b2 = rows[1]['bearing'] 
                    dir1 = "direction2" if GeometryOrientation.angular_diff(b1,d1_2)<GeometryOrientation.angular_diff(b2,d1_2) else "direction1"
                elif np.isnan(d1_2):
                    b2 = rows[1]['bearing'] 
                    dir1 = "direction1" if GeometryOrientation.angular_diff(b1,d1_1)<GeometryOrientation.angular_diff(b2,d1_1) else "direction2"
                else:
                    dir1 = GeometryOrientation.get_direction_based_on_bearing(b1, d1_1, d1_2)
                
                
                dir2 = "direction2" if dir1=="direction1" else "direction1" 

                rows[0]['direction'] = dir1
                rows[1]['direction'] = dir2
                result_rows.extend(rows)

            # Skip groups with more than 2 rows
            else:
                continue

        return pd.DataFrame(result_rows)
    
    @staticmethod
    def angular_diff(a, b):
        """Calculate the smallest angular difference between two bearings."""
        return np.abs((a - b + 180) % 360 - 180)
    
    @staticmethod
    def calculate_bearing(geom):
        if geom is None or geom.is_empty:
            return np.nan
        # Get start and end coordinates
        start = geom.coords[0]
        end = geom.coords[-1]
        angle = np.arctan2(end[0] - start[0], end[1] - start[1])
        return (np.degrees(angle) + 360) % 360
    
    @staticmethod
    def unit_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    @staticmethod
    def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float:
        v1_u = GeometryOrientation.unit_vector(v1)
        v2_u = GeometryOrientation.unit_vector(v2)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_rad = np.arccos(dot)
        return np.degrees(angle_rad)

    @staticmethod
    def get_vector(geometry: Union[LineString, MultiLineString]) -> np.ndarray:
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            if len(coords) < 2:
                raise ValueError("LineString must have at least two points.")
            return np.array(coords[-1]) - np.array(coords[0])

        elif isinstance(geometry, MultiLineString):
            if not geometry.geoms:
                raise ValueError("MultiLineString is empty.")
            first_line = geometry.geoms[0]
            last_line = geometry.geoms[-1]
            return np.array(last_line.coords[-1]) - np.array(first_line.coords[0])

        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    @staticmethod
    def is_parallel(
        line1: Union[LineString, MultiLineString],
        line2: Union[LineString, MultiLineString],
        tolerance: float = 20.0
    ) -> str:
        vec1 = GeometryOrientation.get_vector(line1)
        vec2 = GeometryOrientation.get_vector(line2)
        angle_diff = GeometryOrientation.angle_between_vectors_deg(vec1, vec2)

        if angle_diff < tolerance:
            return "same"
        elif angle_diff > 180 - tolerance:
            return "opposite"
        else:
            return "not parallel"




###################### DISTANCES PART ############################
class GeometryDistanceMetrics:
    @staticmethod
    def _flatten_geometry(geom: Union[LineString, MultiLineString]) -> list[LineString]:
        if isinstance(geom, LineString):
            return [geom]
        elif isinstance(geom, MultiLineString):
            return list(geom.geoms)
        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")

    @staticmethod
    def _sample_points(lines: list[LineString], num_steps: int) -> list[Point]:
        points = []
        for line in lines:
            length = line.length
            if length == 0:
                continue
            for i in range(num_steps):
                pt = line.interpolate(i / num_steps, normalized=True)
                points.append(pt)
        return points

    @staticmethod
    def get_distances(
        line1: Union[LineString, MultiLineString],
        line2: Union[LineString, MultiLineString],
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Sample points along line1 and compute distance to line2.
        """
        lines1 = GeometryDistanceMetrics._flatten_geometry(line1)
        points = GeometryDistanceMetrics._sample_points(lines1, num_steps)

        if not points:
            return np.array([])

        return np.array([line2.distance(pt) for pt in points])

    @staticmethod
    def asymmetric_min_hausdorff(
        geom1: Union[LineString, MultiLineString],
        geom2: Union[LineString, MultiLineString],
        num_steps: int = 10
    ) -> float:
        """
        Compute asymmetric minimum Hausdorff-like distance between two geometries.
        Returns the minimum mean sampled distance in either direction.
        """
        dists_1_to_2 = GeometryDistanceMetrics.get_distances(geom1, geom2, num_steps)
        dists_2_to_1 = GeometryDistanceMetrics.get_distances(geom2, geom1, num_steps)

        if len(dists_1_to_2) == 0 or len(dists_2_to_1) == 0:
            return float('inf')

        return min(dists_1_to_2.mean(), dists_2_to_1.mean())


    @staticmethod
    def get_minimum_distance_match(group):
        if len(group) < 2:
            return group
        if "angle" in group:
            group = group.copy()
            group["distance"] = (group["distance"] / 2).round() * 2
            group["angle_diff"] = GeometryOrientation.angular_diff(group.angle,group.bearing)   
            return group.nsmallest(1, ['distance', 'angle_diff'])
        
        return group.loc[[group["distance"].idxmin()]]









