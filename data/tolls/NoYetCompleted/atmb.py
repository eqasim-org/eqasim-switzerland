import re
import os
import pdfplumber
import pandas as pd
import geopandas as gpd

def configure(context):
    context.config("data_path")
    context.config("tolls_atmb_file", "flyer_tarifs-peage-26-1.pdf")
    context.config("tolls_atmb_geo_file", "atmb.csv")

def execute(context):
    # pdf_file = os.path.join(context.config("data_path"), "tolls", context.config("tolls_atmb_file"))
    # assert os.path.exists(pdf_file), f"File not found: {pdf_file}" # we do this mainly to check the data manually

    # df = pd.DataFrame(toll_tariffs)

    # this file already contains the prices
    geo = pd.read_csv(os.path.join(context.config("data_path"), "tolls", context.config("tolls_atmb_geo_file")), sep=",")
    geo = geo.rename(columns={"value":"price"})
    geo = geo.astype({"lat": float, "lon": float, "price": float})
    geo["geometry"] = gpd.points_from_xy(geo.lon, geo.lat)
    geo = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")

    return None, geo

                








#################### helper functions ####################
"""
IMPORTANT:

This file is different than the other pdf files. It doesn't contain text, but a picture, a photo of the table. Therefore, we cannot extract the data from it using pdfplumber.
"""

# fmt: off
toll_tariffs = [
    # Annemasse – Châtillon-en-Michaille
    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Viry", "destination": None, "price_class_1": 4.40, "price_class_2": 7.50, "price_class_3": 11.70, "price_class_4": 15.20, "price_class_5": 2.90},
    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Éloise", "destination": None, "price_class_1": 4.40, "price_class_2": 7.50, "price_class_3": 11.70, "price_class_4": 15.20, "price_class_5": 2.90},

    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Viry", "destination": None, "price_class_1": 5.80, "price_class_2": 10.00, "price_class_3": 15.10, "price_class_4": 20.20, "price_class_5": 3.90},
    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Bellegarde", "destination": None, "price_class_1": 5.80, "price_class_2": 10.00, "price_class_3": 15.10, "price_class_4": 20.20, "price_class_5": 3.90},

    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Éloise", "destination": None, "price_class_1": 1.40, "price_class_2": 2.50, "price_class_3": 3.40, "price_class_4": 5.00, "price_class_5": 1.00},
    {"section": "Annemasse–Châtillon-en-Michaille", "origin": "Bellegarde", "destination": None, "price_class_1": 1.40, "price_class_2": 2.50, "price_class_3": 3.40, "price_class_4": 5.00, "price_class_5": 1.00},

    # Le Fayet – Gaillard
    {"section": "Le Fayet–Gaillard", "origin": "Le Fayet", "destination": None, "price_class_1": 2.50, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},
    {"section": "Le Fayet–Gaillard", "origin": "Cluses", "destination": None, "price_class_1": 2.50, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},
    {"section": "Le Fayet–Gaillard", "origin": "Cluses amont", "destination": None, "price_class_1": 2.50, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},

    {"section": "Le Fayet–Gaillard", "origin": "Le Fayet", "destination": None, "price_class_1": 4.80, "price_class_2": 8.60, "price_class_3": 13.60, "price_class_4": 17.40, "price_class_5": 2.80},
    {"section": "Le Fayet–Gaillard", "origin": "Cluses Centre", "destination": None, "price_class_1": 4.80, "price_class_2": 8.60, "price_class_3": 13.60, "price_class_4": 17.40, "price_class_5": 2.80},
    
    {"section": "Le Fayet–Gaillard", "origin": "Cluses Centre", "destination": None, "price_class_1": 2.30, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},
    {"section": "Le Fayet–Gaillard", "origin": "Scionzier", "destination": None, "price_class_1": 2.30, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},
    {"section": "Le Fayet–Gaillard", "origin": "Cluses aval", "destination": None, "price_class_1": 2.30, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},

    {"section": "Le Fayet–Gaillard", "origin": "Scionzier", "destination":None, "price_class_1": 2.30, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},
    {"section": "Le Fayet–Gaillard", "origin": "Bonneville", "destination": None, "price_class_1": 2.30, "price_class_2": 4.30, "price_class_3": 6.80, "price_class_4": 8.70, "price_class_5": 1.40},

    {"section": "Le Fayet–Gaillard", "origin": "Bonneville Ouest", "destination": None, "price_class_1": 1.30, "price_class_2": 2.30, "price_class_3": 3.70, "price_class_4": 4.80, "price_class_5": 0.90},
    {"section": "Le Fayet–Gaillard", "origin": "Bonneville Est", "destination": None, "price_class_1": 1.30, "price_class_2": 2.30, "price_class_3": 3.70, "price_class_4": 4.80, "price_class_5": 0.90},
    
    {"section": "Le Fayet–Gaillard", "origin": "Bonneville Est", "destination":None, "price_class_1": 1.70, "price_class_2": 3.10, "price_class_3": 4.80, "price_class_4": 6.30, "price_class_5": 1.00},
    {"section": "Le Fayet–Gaillard", "origin": "Scientrier", "destination": None, "price_class_1": 1.70, "price_class_2": 3.10, "price_class_3": 4.80, "price_class_4": 6.30, "price_class_5": 1.00},

    {"section": "Le Fayet–Gaillard", "origin": "Scientrier", "destination": None, "price_class_1": 2.20, "price_class_2": 3.90, "price_class_3": 6.30, "price_class_4": 8.30, "price_class_5": 1.30},
    {"section": "Le Fayet–Gaillard", "origin": "Gaillard", "destination":None, "price_class_1": 2.20, "price_class_2": 3.90, "price_class_3": 6.30, "price_class_4": 8.30, "price_class_5": 1.30},
    {"section": "Le Fayet–Gaillard", "origin": "Nangy", "destination": None, "price_class_1": 2.20, "price_class_2": 3.90, "price_class_3": 6.30, "price_class_4": 8.30, "price_class_5": 1.30},
    
]