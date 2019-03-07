import pandas as pd
import numpy as np
import data.utils
import data.spatial.utils
import data.constants as c
import pyproj
import geopandas as gpd
import data.spatial.ov_guteklasse

def configure(context, require):
    require.stage("data.microcensus.persons")
    require.stage("data.microcensus.trips")
    require.stage("data.microcensus.transit")

def execute(context):
    df_persons = context.stage("data.microcensus.persons")
    df_trips = context.stage("data.microcensus.trips")
    df_transit = context.stage("data.microcensus.transit")

    df_persons.to_csv("%s/persons.csv" % context.cache_path, sep = ";", index = None)
    df_trips.to_csv("%s/trips.csv" % context.cache_path, sep = ";", index = None)
    df_transit.to_csv("%s/transit.csv" % context.cache_path, sep = ";", index = None)
