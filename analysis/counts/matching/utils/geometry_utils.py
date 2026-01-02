#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:12:39 2025

@author: dabdelkader
"""

from shapely import wkt
from shapely.errors import ShapelyError

def safe_wkt_load(wkt_str):
    try:
        geom = wkt.loads(wkt_str)
        if geom.geom_type == "LineString" and len(geom.coords) > 1:
            return geom
    except (ShapelyError, AttributeError):
        pass
    return None
