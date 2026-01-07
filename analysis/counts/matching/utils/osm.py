"""
Osm is a class to handle OpenStreetMap data retrieval and caching.
It uses osmnx to fetch place geometries and caches results for efficiency.
"""
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
import pandas as pd
from joblib import Memory
import os
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory with default cache directory
memory = Memory(os.path.join(".cache", "osm_class_cache"), verbose=0)

class Osm:

    def __init__(self, places: dict=None, cache:str="cache", country:str="switzerland"):
        _cache_osm = os.path.join(cache, "osm_class_cache")
        self.path_to_cache = os.path.join(_cache_osm, "osm.places.pkl")        
        
        # have this initiated here to use the right cache dir
        global memory       
        memory = Memory(_cache_osm, verbose=0)
        
        # Load the cache if it exists
        if os.path.exists(self.path_to_cache):
            self._places = pd.read_pickle(self.path_to_cache)
        else:
            self._places = gpd.GeoDataFrame()
            
        if places is not None:
            _ = self.get_places(places)
        
        self.border = Osm.get_border(country)
        
    @property
    def places(self):
        return self._places
    
    def get_places(self, places: dict):
        places_geo = {}
        for old_name, place in places.items():
            assert isinstance(place, str), "Places should be a list of strings"
            lat, lon = self.get_one_place(place)  # Still static, but accessed through self
            
            if not (np.isnan(lat)|np.isnan(lon)):
                places_geo[place] = dict(
                    old_name=old_name,
                    lat=lat,
                    lon=lon,
                    geometry=Point((lon, lat))  # Correct coordinate order
                )

        places_geo = pd.DataFrame.from_dict(places_geo).T
        places_geo.index.name = "cities"
        places_geo = places_geo.reset_index()
        places_geo = gpd.GeoDataFrame(places_geo,
                                      geometry="geometry",
                                      crs="EPSG:4326")
        
        self.add_to_cache(places_geo)
        return places_geo
    
    def add_to_cache(self, df):
        self._places = pd.concat([self._places,df], ignore_index=True)
        self._places = self._places.drop_duplicates(["cities"])
        self._places = gpd.GeoDataFrame(self._places,
                                       geometry="geometry",
                                       crs="EPSG:4326")
        self._places.to_pickle(self.path_to_cache)
        
    def get_one_place(self, place: str):
        if len(self._places) and (place in self._places.cities):
            row = self._places[self._places.cities==place].iloc[0]            
            return  row["lat"], row["lon"]
        
        lat, lon = Osm.get_one_place_from_osm(place)
        return lat, lon
    
    @staticmethod
    @memory.cache
    def get_one_place_from_osm(place: str):
        try:
            lat, lon = ox.geocode(place)
        except Exception as e:
            logger.info(f"Failed to geocode '{place}': {e}")
            lat, lon = np.nan, np.nan
        return lat, lon
    
    @staticmethod
    @memory.cache
    def get_border(place: str):
        return ox.geocode_to_gdf(place)
        
