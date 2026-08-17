import re
import os
import pdfplumber
import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context):
    context.config("data_path")
    context.config("tolls_aprr_file", "TARIFS_APRR.pdf")
    context.config("tolls_aprr_geo_file", "aprr.csv")

def execute(context):
    pdf_file = os.path.join(context.config("data_path"), "tolls", context.config("tolls_aprr_file"))
    all_rows = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages[1:]:
            all_rows.extend(parse_page(page))

    df = pd.DataFrame(all_rows, columns=COLS)

    # Convert numeric columns from French "1 234,56" format to float
    for c in ["Distance", "Classe 1", "Classe 2", "Classe 3", "Classe 4", "Classe 5"]:
        df[c] = df[c].apply(to_float)

    # rename columns
    df = df.rename(columns={"Gare d'entrée": "origin",
                            "Gare de sortie": "destination",
                            "Distance": "distance",
                            "Classe 1": "price",
                            "Classe 2": "price_class_2",
                            "Classe 3": "price_class_3",
                            "Classe 4": "price_class_4",
                            "Classe 5": "price_class_5"})
    
    # read geometries
    geo = pd.read_csv(os.path.join(context.config("data_path"), "tolls", context.config("tolls_aprr_geo_file")))
    geo["price"] = np.nan
    geo["geometry"] = gpd.points_from_xy(geo.lon, geo.lat)
    geo = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")

    return df, geo








############### helper functions ############
COLS = ["Gare d'entrée", "Gare de sortie", "Distance",
        "Classe 1", "Classe 2", "Classe 3", "Classe 4", "Classe 5"]

# Midpoints between each column's header x0 position (in points).
# Words with x0 < BOUNDARIES[i] go to column i; last bucket catches the rest.
BOUNDARIES = [122.35, 222.75, 288.0, 331.2, 374.15, 417.1, 460.05]


def bucket(x0):
    for i, b in enumerate(BOUNDARIES):
        if x0 < b:
            return i
    return len(BOUNDARIES)


def parse_page(page):
    # Drop '€' symbols: they sit at a slightly different vertical baseline
    # than the number they belong to, which would otherwise split one
    # logical row into two clusters.
    words = [w for w in page.extract_words() if w["text"] != "€"]
    words.sort(key=lambda w: w["top"])

    # Cluster words into rows by vertical position (tolerance = 3pt)
    clusters = []
    for w in words:
        if clusters and abs(w["top"] - clusters[-1][0]["top"]) < 3:
            clusters[-1].append(w)
        else:
            clusters.append([w])

    rows = []
    for c in clusters:
        c.sort(key=lambda w: w["x0"])
        cells = [[] for _ in range(8)]
        for w in c:
            cells[bucket(w["x0"])].append(w["text"])
        cells = [" ".join(cell) for cell in cells]
        # Keep only real data rows: the Distance column must look numeric
        # (this filters out the title/header text that repeats on every page)
        if re.match(r"^\d[\d\s]*,\d+$", cells[2]):
            rows.append(cells)
    return rows


def to_float(x):
    return float(x.replace(" ", "").replace(",", "."))
