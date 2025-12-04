import geopandas as gpd
import pandas as pd


def configure(context):
    context.config("uspat_cantons")
    context.config("data_path")

    context.stage("data.spatial.cantons")


canton_to_abbrev = {
    "Zürich": "ZH",
    "Bern": "BE",
    "Luzern": "LU",
    "Uri": "UR",
    "Schwyz": "SZ",
    "Obwalden": "OW",
    "Nidwalden": "NW",
    "Glarus": "GL",
    "Zug": "ZG",
    "Fribourg": "FR",
    "Solothurn": "SO",
    "Basel-Stadt": "BS",
    "Basel-Landschaft": "BL",
    "Schaffhausen": "SH",
    "Appenzell Ausserrhoden": "AR",
    "Appenzell Innerrhoden": "AI",
    "St. Gallen": "SG",
    "Graubünden": "GR",
    "Aargau": "AG",
    "Thurgau": "TG",
    "Ticino": "TI",
    "Vaud": "VD",
    "Valais": "VS",
    "Neuchâtel": "NE",
    "Genève": "GE",
    "Jura": "JU"
}
abbrev_to_canton = {v: k for k, v in canton_to_abbrev.items()}


def execute(context):
    data_path    = context.config("data_path")
    uspat_path   = f"{data_path}/spatial/USPAT/statistische-grundeinheiten_stufe1_2025-01-01_2056.gpkg"
    cantons_list = context.config("uspat_cantons")

    # UPSAT zones within the study area cantons
    uspat_zones = gpd.read_file(uspat_path)
    zones_keep = uspat_zones[uspat_zones["KT_ID"].isin(cantons_list)]
    zones_keep = zones_keep[["U1_ID", "geometry"]].rename(columns={"U1_ID": "zone_id"})

    # Outside the study area, use the cantons as zones
    cantons = context.stage("data.spatial.cantons").copy()[["canton_name", "geometry"]]
    cantons["canton_name"] = [canton_to_abbrev[name] for name in cantons["canton_name"]]
    cantons = cantons[~cantons["canton_name"].isin(cantons_list)]
    cantons["zone_id"] = range(1, len(cantons) + 1)
    cantons = cantons[["zone_id", "geometry"]]

    uspat_zones = pd.concat([zones_keep, cantons], ignore_index=True)

    return uspat_zones