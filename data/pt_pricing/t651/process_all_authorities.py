import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

import data.pt_pricing.t651.authorities.unireso
import data.pt_pricing.t651.authorities.zvv
import data.pt_pricing.t651.authorities.uniresoFR
import data.pt_pricing.t651.authorities.mobilis
import data.pt_pricing.t651.authorities.libero
import data.pt_pricing.t651.authorities.frimobil
import data.pt_pricing.t651.authorities.ondeverte
import data.pt_pricing.t651.authorities.passepartout
import data.pt_pricing.t651.authorities.awelle
import data.pt_pricing.t651.authorities.ostwind
import data.pt_pricing.t651.authorities.zvb
import data.pt_pricing.t651.authorities.tvsz
import data.pt_pricing.t651.authorities.arcobaleno
import data.pt_pricing.t651.authorities.klosters
import data.pt_pricing.t651.authorities.davos
import data.pt_pricing.t651.authorities.transreno
import data.pt_pricing.t651.authorities.ctju
import data.pt_pricing.t651.authorities.engadinmobil
import data.pt_pricing.t651.authorities.sion
import data.pt_pricing.t651.authorities.tnw


import data.pt_pricing.t651.utils as t651utils

import warnings
warnings.filterwarnings("ignore")


def process_gtfs_stops(gtfs_stops_path):
    stops = pd.read_csv(gtfs_stops_path)
    stops = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs="EPSG:4326"
        )
    stops_gdf = stops.to_crs(crs="EPSG:2056")

    stops_gdf = stops_gdf.drop(columns=["stop_lat", "stop_lon"])

    stops_gdf["stop_id"] = stops_gdf["stop_id"].astype(str)

    return stops_gdf


def match_stops(gtfs_stops, stops):

    stops = pd.merge(stops, gtfs_stops[["stop_id", "stop_name", "geometry"]], 
                    left_on = "stop", right_on = "stop_name",
                    how = "left")

    matched     = stops[stops["geometry"].notna()]

    return matched


def prefix_zones_with_network(row):
    network = row["tarif network"]
    zones   = row["zones"]  

    if not zones or pd.isna(network):
        return None
    
    return ", ".join(f"{network}:{zone}" for zone in zones)


def process_zvv(authorities_path, temp_path, spatial_zones):
    pdf_path  = Path(f"{authorities_path}/T651.zvv.stops.pdf")
    temp_path = f"{temp_path}/zvv"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    zones_path = Path(f"{authorities_path}/ZVVzonenplan/Tarifzonen_des_offentlichen_Verkehrs_-OGD/ZVV_TARIFZONEN_F.shp")

    stations      = data.pt_pricing.t651.authorities.zvv.create_stations(pdf_path, temp_path)
    spatial_zones = data.pt_pricing.t651.authorities.zvv.import_zones(spatial_zones, zones_path)

    return stations, spatial_zones


def process_unireso(authorities_path, temp_path, spatial_zones, cantons):

    pdf_path  = Path(f"{authorities_path}/T651.unireso.stops.pdf")
    temp_path = f"{temp_path}/unireso"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.unireso.create_stations(pdf_path, temp_path)
    spatial_zones = data.pt_pricing.t651.authorities.unireso.import_zones(spatial_zones, cantons)

    return stations, spatial_zones


def process_mobilis(authorities_path):
    excel_path  = Path(f"{authorities_path}/flph_Liste des arrêts Mobilis _dès le 15.12.2024 modif dès le 01.06.2025.xlsx")
    stations    = data.pt_pricing.t651.authorities.mobilis.create_stations(excel_path)

    return stations


def process_libero(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.libero.stops.pdf")
    temp_path = f"{temp_path}/libero"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.libero.create_stations(pdf_path, temp_path)

    return stations


def process_frimobil(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.frimobil.stops.pdf")
    temp_path = f"{temp_path}/frimobil"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.frimobil.create_stations(pdf_path, temp_path)

    return stations


def process_ondeverte(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.ondeverte.stops.pdf")
    temp_path = f"{temp_path}/ondeverte"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.ondeverte.create_stations(pdf_path, temp_path)

    return stations


def process_passepartout(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.passepartout.stops.pdf")
    temp_path = f"{temp_path}/passepartout"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.passepartout.create_stations(pdf_path, temp_path)

    return stations


def process_awelle(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.awelle.stops.pdf")
    temp_path = f"{temp_path}/awelle"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.awelle.create_stations(pdf_path, temp_path)

    return stations


def process_ostwind(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.ostwind.stops.pdf")
    temp_path = f"{temp_path}/ostwind"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.ostwind.create_stations(pdf_path, temp_path)

    return stations


def process_zvb(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.zvb.stops.pdf")
    temp_path = f"{temp_path}/zvb"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.zvb.create_stations(pdf_path, temp_path)

    return stations


def process_tvsz(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.tvsz.stops.pdf")
    temp_path = f"{temp_path}/tvsz"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.tvsz.create_stations(pdf_path, temp_path)

    return stations


def process_klosters(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.klosters.stops.pdf")
    temp_path = f"{temp_path}/klosters"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.klosters.create_stations(pdf_path, temp_path)

    return stations


def process_davos(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.davos.stops.pdf")
    temp_path = f"{temp_path}/davos"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.davos.create_stations(pdf_path, temp_path)

    return stations


def process_transreno(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.transreno.stops.pdf")
    temp_path = f"{temp_path}/transreno"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.transreno.create_stations(pdf_path, temp_path)

    return stations


def process_transreno(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.transreno.stops.pdf")
    temp_path = f"{temp_path}/transreno"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.transreno.create_stations(pdf_path, temp_path)

    return stations


def process_ctju(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.ctju.stops.pdf")
    temp_path = f"{temp_path}/ctju"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.ctju.create_stations(pdf_path, temp_path)

    return stations


def process_engadinmobil(authorities_path, temp_path):
    pdf_path  = Path(f"{authorities_path}/T651.engadinmobil.stops.pdf")
    temp_path = f"{temp_path}/engadinmobil"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    stations      = data.pt_pricing.t651.authorities.engadinmobil.create_stations(pdf_path, temp_path)

    return stations




def configure(context):
    context.config("data_path")
    context.config("gtfs_name")

    context.stage("data.spatial.cantons")


def execute(context):
    data_path = context.config("data_path")
    gtfs_name = context.config("gtfs_name")
    gtfs_stops_path = f"{data_path}/gtfs/{gtfs_name}/stops.txt"

    cantons = context.stage("data.spatial.cantons")

    gtfs = process_gtfs_stops(gtfs_stops_path)

    spatial_zones = {}
    networks      = []

    authorities_path = f"{data_path}/pt_pricing/t651"

    temp_path = f"{context.path()}/temp/t651"
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    # ZVV
    zvv, spatial_zones = process_zvv(authorities_path, temp_path, spatial_zones)
    zvv = pd.DataFrame(zvv, columns=["stop", "zones", "local network", "tarif network"])
    zvv = match_stops(gtfs, zvv)
    networks.append(zvv)
    print("ZVV imported")

    # Unireso
    tpg_stops   = pd.read_csv(f"{authorities_path}/tpg_Arrets.csv", encoding = "latin1", sep = ";")
    tpg_stops   = tpg_stops[["NomArret", "CodeDidoc"]].rename(columns = {"NomArret": "stop", "CodeDidoc": "stop_id"}).drop_duplicates()
    tpg_stops   = tpg_stops[tpg_stops["stop_id"].notna()]
    tpg_stops["stop_id"] = tpg_stops["stop_id"].astype(int).astype(str)

    unireso, spatial_zones = process_unireso(authorities_path, temp_path, spatial_zones, cantons)
    unireso  = pd.DataFrame(unireso, columns=["stop", "zones", "local network", "tarif network"])
    unireso  = data.pt_pricing.t651.authorities.unireso.match_stops(gtfs, unireso, tpg_stops)
    networks.append(unireso)
    print("Unireso imported")

    uniresoFR = data.pt_pricing.t651.authorities.uniresoFR.assign_zones_to_gtfs(gtfs)
    networks.append(uniresoFR)
    print("Unireso FR imported")

    # Lausanne / Canton Vaud
    mobilis = process_mobilis(authorities_path)
    mobilis = pd.DataFrame(mobilis, columns=["stop", "zones", "local network", "tarif network"])
    mobilis = match_stops(gtfs, mobilis)
    networks.append(mobilis)
    print("Mobilis imported")

    # Bern
    libero = process_libero(authorities_path, temp_path)
    libero = pd.DataFrame(libero, columns=["stop", "zones", "local network", "tarif network"])
    libero = match_stops(gtfs, libero)
    networks.append(libero)
    print("Libero imported")

    # Fribourg
    frimobil = process_frimobil(authorities_path, temp_path)
    frimobil = pd.DataFrame(frimobil, columns=["stop", "zones", "local network", "tarif network"])
    frimobil = match_stops(gtfs, frimobil)
    networks.append(frimobil)
    print("Frimobil imported")

    # Neuchâtel
    ondeverte = process_ondeverte(authorities_path, temp_path)
    ondeverte = pd.DataFrame(ondeverte, columns=["stop", "zones", "local network", "tarif network"])
    ondeverte = match_stops(gtfs, ondeverte)
    networks.append(ondeverte)
    print("Ondeverte imported")

    # Luzern
    passepartout = process_passepartout(authorities_path, temp_path)
    passepartout = pd.DataFrame(passepartout, columns=["stop", "zones", "local network", "tarif network"])
    passepartout = match_stops(gtfs, passepartout)
    networks.append(passepartout)
    print("Passepartout imported")

    # Awelle (Aargau)
    awelle = process_awelle(authorities_path, temp_path)
    awelle = pd.DataFrame(awelle, columns=["stop", "zones", "local network", "tarif network"])
    awelle = match_stops(gtfs, awelle)
    networks.append(awelle)
    print("A-Welle imported")

    # Ostwind (Sankt Gallen + Appenzell AR + Appenzell IR + LI)
    ostwind = process_ostwind(authorities_path, temp_path)
    ostwind = pd.DataFrame(ostwind, columns=["stop", "zones", "local network", "tarif network"])
    ostwind = match_stops(gtfs, ostwind)
    networks.append(ostwind)
    print("Ostwind imported")

    # Zug (ZVB)
    zvb = process_zvb(authorities_path, temp_path)
    zvb = pd.DataFrame(zvb, columns=["stop", "zones", "local network", "tarif network"])
    zvb = match_stops(gtfs, zvb)
    networks.append(zvb)
    print("ZVB imported")

    # TVSZ (Schwyz)
    tvsz = process_tvsz(authorities_path, temp_path)
    tvsz = pd.DataFrame(tvsz, columns=["stop", "zones", "local network", "tarif network"])
    tvsz = match_stops(gtfs, tvsz)
    networks.append(tvsz)
    print("TSVZ imported")

    # Klosters
    klosters = process_klosters(authorities_path, temp_path)
    klosters = pd.DataFrame(klosters, columns=["stop", "zones", "local network", "tarif network"])
    klosters = match_stops(gtfs, klosters)
    networks.append(klosters)
    print("Klosters imported")

    # Davos
    davos = process_davos(authorities_path, temp_path)
    davos = pd.DataFrame(davos, columns=["stop", "zones", "local network", "tarif network"])
    davos = match_stops(gtfs, davos)
    networks.append(davos)
    print("Davos imported")

    # Transreno
    transreno = process_transreno(authorities_path, temp_path)
    transreno = pd.DataFrame(transreno, columns=["stop", "zones", "local network", "tarif network"])
    transreno = match_stops(gtfs, transreno)
    networks.append(transreno)
    print("Transreno imported")

    # EngadinMobil (Skt Moritz)
    engadinmobil = process_engadinmobil(authorities_path, temp_path)
    engadinmobil = pd.DataFrame(engadinmobil, columns=["stop", "zones", "local network", "tarif network"])
    engadinmobil = match_stops(gtfs, engadinmobil)
    networks.append(engadinmobil)
    print("Engadin imported")

    # Arcobaleno (Ticino)
    arcobaleno = data.pt_pricing.t651.authorities.arcobaleno.assign_zones_to_gtfs(gtfs)
    networks.append(arcobaleno)
    print("Arcobaleno imported")

    # TNW (Nordwest Schweiz, around Basel)
    tnw = data.pt_pricing.t651.authorities.tnw.assign_zones_to_gtfs(gtfs)
    networks.append(tnw)
    print("TNW imported")

    # Sion
    sion = data.pt_pricing.t651.authorities.sion.assign_zones_to_gtfs(gtfs)
    networks.append(sion)
    print("Sion imported")


    ### DONE ###
    # Merge the constructed networks into GTFS
    networks = pd.concat(networks)
    networks = networks[networks["geometry"].notna()]
    networks = networks[["stop_id", "stop_name", "geometry", "tarif network", "local network", "zones"]]

    gtfs_networks = gtfs.merge(networks, on = ["stop_id", "stop_name", "geometry"], how = "left")

    gtfs_networks.loc[gtfs_networks["zones"].notna(), "zones"] = (
        gtfs_networks[gtfs_networks["zones"].notna()]
        .apply(prefix_zones_with_network, axis=1)
    )

    # Create shapes - from estimated polygon shapes and spatial_zones
    df_shapes = t651utils.create_shapes(gtfs_networks, spatial_zones)

    found   = gtfs_networks[gtfs_networks["zones"].notna()]
    missing = gtfs_networks[gtfs_networks["zones"].isna()]

    # Find the zones for the missing stops using the alpha shapes created before
    missing_spatial_merge = gpd.sjoin(missing[["stop_id", "stop_name", "geometry"]], df_shapes, predicate = "within", how = "left").drop(columns="index_right")
    
    has_zones = pd.concat([found, missing_spatial_merge[missing_spatial_merge["zones"].notna()]].copy())
    no_zones  = missing_spatial_merge[missing_spatial_merge["zones"].isna()]

    gtfs_networks = pd.concat([has_zones, no_zones])
    gtfs_networks = gtfs_networks[["stop_id", "stop_name", "tarif network", "local network", "zones", "geometry"]]
    gtfs_networks = (
        gtfs_networks
        .groupby(["stop_id", "stop_name", "geometry"])
        .agg({
            "tarif network": lambda x: list(sorted(set(x))), 
            "local network": lambda x: list(sorted(set(x))),
            "zones": lambda x: ", ".join(
                sorted(
                    set(
                        zone.strip()
                        for z in x if pd.notna(z) and z.strip() != ""  # Skip NaN/empty
                        for zone in z.split(",")
                    )
                )
            ) if any(pd.notna(z) and z.strip() != "" for z in x) else None  # Return None if all empty
        })
        .reset_index()
    )

    gtfs_networks.loc[gtfs_networks["stop_name"].str.startswith("Grenzach"), "zones"] = [[np.nan]]
    gtfs_networks.loc[gtfs_networks["stop_name"].str.startswith("Grenzach"), "tarif network"] =  [[np.nan]]
    gtfs_networks.loc[gtfs_networks["stop_name"].str.startswith("Grenzach"), "local network"] =  [[np.nan]]

    gtfs_networks = gpd.GeoDataFrame(gtfs_networks, crs = "EPSG:2056")

    # Save the outputs
    #gtfs_networks.to_file(f"{context.path()}/gtfs_with_zone_info.shp")

    del gtfs_networks["geometry"]

    gtfs_networks = pd.DataFrame(gtfs_networks)

    return gtfs_networks



