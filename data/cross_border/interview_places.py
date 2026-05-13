import geopandas as gpd
import pandas as pd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")
    crossings = f"{data_path}/crossborder/border_interview_places_final.gpkg"
    crossings = gpd.read_file(crossings)
    return crossings