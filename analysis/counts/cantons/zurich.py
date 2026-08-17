import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os
import contextily as ctx
import unicodedata
import numpy as np

def configure(context):
    context.config("data_path")
    context.config("counts_path", default=os.path.join(context.config("data_path"),"traffic_counts"))


def execute(context):
    # Define paths
    data_path = context.config("counts_path")    
    input_path  = os.path.join(data_path,"Zurich","sid_dav_verkehrszaehlung_miv_od2031_2024.csv")
    # loc_path = os.path.join(data_path,"Zurich", "locations", "data", "data.gpkg")

    output_path = os.path.join(context.path(),"processed_data.gpkg")

    # Column mapping
    cols = {
        "MSID": "id2",
        "ZSID": "id",
        "EKoord": "x",
        "NKoord": "y",
        "AnzFahrzeuge": "count",
        "Richtung": "direction",
        "MessungDatZeit":"mesDate",
    }

    # Read data
    df = pd.read_csv(input_path, usecols=cols.keys()).rename(columns=cols)
    df['mesDate'] = pd.to_datetime(df['mesDate'])
    df["hour"] = df.mesDate.dt.hour
    df["day"] = df.mesDate.dt.day_of_year
    df["year"] = df.mesDate.dt.year
    df["is_weekend"] = df.mesDate.dt.day_name().isin(['Saturday', 'Sunday'])

    # remove nans
    df = df[df["count"].notna()]

    # keep last year
    df = df[df['year'] == df['year'].max()]

    # remove days with not enough observations
    hours_observation = df.groupby(["id","direction","day"])["count"].transform("size")
    df = df[hours_observation>=22] # remove where there are not enough observations

    # # Only keep two directional links
    # d = df.groupby("id")["direction"].transform("nunique")
    # df = df[d==2]

    # compute the flow
    df_flow = compute_flow(df, col_name="flow")
    df_flow_w = compute_flow(df[~df["is_weekend"]], col_name="flow_w")
    df = df_flow.merge(df_flow_w[['id','direction','flow_w']], on=["id","direction"], how="left")

    # # get the geometry from the locations file
    # loc = gpd.read_file(loc_path)[['zsid','geometry']].rename(columns={ 'zsid':'id'})
    # df = df[['id','flow', 'flow_w']].merge(loc, on="id", how="left")
    # df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:2056")
    # df["x"] = df.geometry.x
    # df["y"] = df.geometry.y
    df["geometry"] = gpd.points_from_xy(df["x"], df["y"])

    # merge withc matched locations
    df = merge_with_matched_stations(context, df)

    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    df.plot(ax=ax, color='red', markersize=5)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df.crs)
    _ = plt.axis("off")
    plt.savefig(os.path.join(context.path(),"fig.png"), dpi=300)
    plt.close()

    # save file    
    df.to_file(output_path)
    
    return output_path





##################################################
def compute_flow(df, col_name="flow"):
    df = df.copy()
    df = (
        df
        .dropna(subset=["count"]) 
        .groupby(["id","direction","day"], as_index=False)
        .agg({        
            "x": "first",
            "y": "first",
            "count": "sum"        
        })
        .groupby(["id","direction"], as_index=False)
        .agg({        
            "x": "first",
            "y": "first",
            "count": "mean",        
        })
    )
    df = df.rename(columns={"count":col_name})
    return df[['id','direction','x','y',col_name]]


def normalize_id(s: str) -> str:
    """Make IDs robust to unicode normalization differences (e.g. German umlauts)."""
    if pd.isna(s):
        return s
    s = str(s)
    # Canonical decomposition + strip accents/diacritics entirely
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


def merge_with_matched_stations(context, df):
    data_path  = context.config("counts_path")
    file_path  = os.path.join(data_path, "Zurich", "zurich_matched.csv")

    df_matched  = pd.read_csv(file_path, encoding="utf-8")[
        ["counter_id", "link_osm_id", "link_x_from_node", "link_y_from_node",
         "link_x_to_node", "link_y_to_node"]
    ]
    df_matched["angle"] = np.degrees(np.arctan2(
                                    df_matched["link_y_to_node"] - df_matched["link_y_from_node"],
                                    df_matched["link_x_to_node"] - df_matched["link_x_from_node"],
                                ))

    df_matched = df_matched[["counter_id", "link_osm_id", "angle"]]
    df_matched = df_matched.rename(columns={"link_osm_id": "osm_id"})

    # normalized merge key on the matched side (keep original counter_id too)
    df_matched["merge_key"] = df_matched["counter_id"].map(normalize_id)

    df = df.copy()
    df["counter_id"] = df["id"].astype(str) + "_" + df["direction"].astype(str).str[0:3]
    df["merge_key"] = df["counter_id"].map(normalize_id)

    # merge on the normalized key, but keep df's own counter_id as the id afterwards
    df = df.merge(
        df_matched.drop(columns=["counter_id"]),
        on="merge_key",
        how="left"
    ).drop(columns=["merge_key"])

    # we need to drop the id and use counter id instead
    df = df.drop(columns=["id"]).rename(columns={"counter_id": "id"})

    df = df[df["osm_id"].notna()].reset_index(drop=True)
    return gpd.GeoDataFrame(df, crs="EPSG:2056")



