import pandas as pd
import geopandas as gpd
import numpy as np
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", "workday")


def resolve_days(day):
    if isinstance(day, str):
        if day == "workday":
            return sorted(WEEKDAYS)
        elif day == "weekend":
            return sorted(WEEKEND)
        elif day in ALL_DAYS:
            return [day]
        else:
            raise RuntimeError(f"Invalid day value: {day}. Expected a day name, workday, or weekend.")
        
    elif isinstance(day, (list, tuple)):
        day     = list(day)
        invalid = set(day) - ALL_DAYS
        if invalid:
            raise RuntimeError(f"Invalid day value(s): {invalid}. Expected day names from {ALL_DAYS}.")
        return day
    
    else:
        raise RuntimeError(f"Invalid type for day parameter: {type(day)}. Expected str or list.")
    

def filter_by_days(df, days):
    day_ints = [DAY_TO_INT[d] for d in days]
    return df[df["JOUR_SEM_DEPL"].isin(day_ints)]
    

WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday"}
WEEKEND  = {"saturday", "sunday"}
ALL_DAYS = WEEKDAYS | WEEKEND


DAY_TO_INT = {
    "monday": 1, "tuesday": 2, "wednesday": 3,
    "thursday": 4, "friday": 5
}


HOUSEHOLD_COLUMNS = {
    "NUM_MEN": str, # id
    "NB_AUTO": str, "NB_VELO": str, "NB_2RM": str,  # number_of_cars, number_of_bikes, number_of_motorbikes
    "POND_MEN": str # weights
}

PERSON_COLUMNS = {
    "NUM_MEN": str, "NUM_PERS": str, # id
    "DEPL_OUI_NON": str, # respondents of travel questionary section
    "SEXE": int, "AGE": int, # sex, age
    "STATUT_TRAVAIL": str, # employed, studies
    "PERMIS_CONDUIRE": str, "ABO_TC": str, # has_license, has_pt_subscription
    "PROFESSION": str, # socioprofessional_class
    "POND_PERS": str, # weights
    "JOUR_SEM_DEPL": int # day of the week, 1 = monday to 5 = friday
}


TRIP_COLUMNS = {
    "NUM_MEN_FR": str, "NUM_PERS_FR": str, "NUM_DEPL_FR_PROV": str, # id
    "D2A": str, "D5A": str, # preceding_purpose, following_purpose
    "LIEU_DEPART_FR": str, "LIEU_ARRIVEE_FR": str, # origin_zone, destination_zone
    "D4": int, "D8": int, # time_departure, time_arrival
    "MODP": int, "DOIB": int, "DIST": int # mode, euclidean_distance, routed_distance
}

MODES_MAP = {
    "car": [10, 13, 15, 21, 81], # 10 is (driving) an ambulance
    "car_passenger": [14, 16, 22, 82],
    "pt": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 51, 52, 53, 54, 61, 71, 91, 92, 94, 95],
    "bike": [11, 17, 12, 18, 93],
    "walk": [1, 2] # Actually, 2 is not really explained, but we assume it is walk
}


PURPOSE_TO_KEY = {"DOMICILE": 1,
                  "RÉSIDENCE SECONDAIRE, AUTRE DOMICILE": 2,
                  " TRAVAIL SUR LE LIEU D\x92EMPLOI DÉCLARÉ": 11,
                  " TRAVAIL SUR UN AUTRE LIEU - TÉLÉTRAVAIL": 12,
                  " TRAVAIL SUR UN AUTRE LIEU - HORS TÉLÉTRAVAIL": 13,
                  " ETRE GARDÉ (NOURRICE, CRÈCHE,,,)": 21,
                  " ETUDIER À L'ÉCOLE MATERNELLE ET PRIMAIRE (SUR LE LIEU DÉCLARÉ)": 22,
                  " ETUDIER AU COLLÈGE (SUR LE LIEU DÉCLARÉ)": 23,
                  " ETUDIER AU LYCÉE (SUR LE LIEU DÉCLARÉ) ": 24,
                  " ETUDIER À L'UNIVERSITÉ ET GRANDES ÉCOLES (SUR LE LIEU DÉCLARÉ) ": 25,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (ECOLE MATERNELLE ET PRIMAIRE)": 26,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (COLLÈGE) ": 27,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (LYCÉE)": 28,
                  " ETUDIER SUR UN AUTRE LIEU DÉCLARÉ (UNIVERSITÉ ET GRANDES ÉCOLES) ": 29,
                  " VISITE D\x92UN MAGASIN, D\x92UN CENTRE COMMERCIAL OU D\x92UN MARCHÉ DE PLEIN VENT SANS EFFECTUER D\x92ACHAT": 30,
                  " RÉALISER PLUSIEURS MOTIFS EN CENTRE COMMERCIAL": 31,
                  " FAIRE DES ACHATS EN GRAND MAGASIN, SUPERMARCHÉ, HYPERMARCHÉ ET LEURS GALERIES MARCHANDES": 32,
                  " FAIRE DES ACHATS EN PETIT ET MOYEN COMMERCE ET ": 33,
                  "34": 34,
                  "35": 35,
                  "41": 41,
                  "42": 42,
                  "43": 43,
                  "50": 51, 
                  "51": 51,
                  "52": 52,
                  "53": 53,
                  "54": 54,
                  "61": 61,
                  "62": 62,
                  "63": 63,
                  "64": 64,
                  "65": 63,
                  "66": 64, 
                  "67": 63, 
                  "68": 64, 
                  "72": 72,
                  "73": 73,
                  "76": 72, 
                  "77": 73, 
                  "81": 81,
                  "82": 82,
                  "91": 91}

def execute(context):
    data_path   = context.config("data_path")
    edgt_path   = f"{data_path}/hts_annemasse_2017_tpg/TPG"

    df_households = pd.read_csv(f"{edgt_path}/Menages_EMD.csv", sep = ";", usecols = list(HOUSEHOLD_COLUMNS.keys()), dtype = HOUSEHOLD_COLUMNS)
    df_persons    = pd.read_csv(f"{edgt_path}/Personnes_EMD.csv", sep = ";", usecols = list(PERSON_COLUMNS.keys()), dtype = PERSON_COLUMNS)
    df_trips      = pd.read_csv(f"{edgt_path}/deplacements_emd_full_OCT.csv", sep = ";", usecols = list(TRIP_COLUMNS.keys()), dtype = TRIP_COLUMNS, encoding = "latin1")

    df_trips["D2A"] = df_trips["D2A"].map(PURPOSE_TO_KEY).fillna(91).astype(int)
    df_trips["D5A"] = df_trips["D5A"].map(PURPOSE_TO_KEY).fillna(91).astype(int)

    spatial1 = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZF.TAB")
    spatial1 = spatial1.set_crs(epsg = 2154, allow_override=True)
    spatial1.columns = spatial1.columns.str.lower()
    spatial1 = spatial1[["geometry", "zonefine", "depcom"]]
    spatial1.columns = ["geometry", "zone_id", "departement"]
    spatial1["departement_id"] = spatial1["departement"].astype(str).str[:2]
    spatial1 = spatial1[["geometry", "zone_id", "departement_id"]]

    spatial2 = gpd.read_file(f"{edgt_path}/Doc/SIG/EDGTFVG2016_ZonesExternes.TAB")
    spatial2 = spatial2.set_crs(epsg = 2154, allow_override=True)
    spatial2.columns = spatial2.columns.str.lower()
    spatial2 = spatial2[["geometry", "num_zf", "nom_d10"]]
    spatial2.columns = ["geometry", "zone_id", "departement_id"]

    df_spatial = gpd.GeoDataFrame(pd.concat([spatial1, spatial2]), crs=spatial1.crs)

    # Merge departement into households
    df_spatial        = df_spatial[["zone_id", "departement_id"]].copy()
    df_spatial["ZFM"] = df_spatial["zone_id"].astype(str).str.pad(width = 8, side = "left", fillchar = "0")
    df_spatial        = df_spatial[["ZFM", "departement_id"]]

    df_households["ZFM"]            = df_households["NUM_MEN"].str[:-3]
    df_households                   = pd.merge(df_households, df_spatial, on = "ZFM", how = "left")
    df_households["departement_id"] = df_households["departement_id"].fillna("unknown")

    # Transform original IDs to integer (they are hierarchichal)
    df_households["edgt_household_id"] = df_households["NUM_MEN"].astype(int)
    df_persons["edgt_person_id"]       = df_persons["NUM_PERS"].str[-2:].astype(int)
    df_persons["edgt_household_id"]    = df_persons["NUM_MEN"].astype(int)
    df_trips["edgt_person_id"]         = df_trips["NUM_PERS_FR"].str[-2:].astype(int)
    df_trips["edgt_household_id"]      = df_trips["NUM_MEN_FR"].astype(int)
    df_trips["edgt_trip_id"]           = df_trips["NUM_DEPL_FR_PROV"].str[-2:].astype(int)

    df_households["household_id"] = np.arange(len(df_households))

    days = resolve_days(context.config("specific_day_scenario"))
    if any(d in WEEKEND for d in days):
        raise RuntimeError(
            f"Invalid day for EDGT Annemasse: {days}. "
            f"The Annemasse EDGT survey does not cover weekends. "
            f"Please select ENTD as HTS or choose another day."
        )    
    
    initial_length = len(df_persons)
    df_persons     = filter_by_days(df_persons, days)
    final_length   = len(df_persons)
    persdiff       = initial_length - final_length
    persreldiff    = persdiff / initial_length * 100

    hhl_length1   = len(df_households)
    df_households = df_households[df_households["edgt_household_id"].isin(df_persons["edgt_household_id"].values.tolist())]
    hhl_length2   = len(df_households)
    hhldiff       = hhl_length1 - hhl_length2
    hhlreldiff    = hhldiff / hhl_length1 * 100

    trips_length1 = len(df_trips)
    df_trips      = df_trips[df_trips["edgt_person_id"].isin(df_trips["edgt_person_id"].values.tolist())]
    trips_length2 = len(df_trips)
    tripsdiff     = trips_length1 - trips_length2
    tripsreldiff  = tripsdiff / trips_length1 * 100

    logger.info(f"Filtering data based on selected days. Removed {hhldiff} ({round(hhlreldiff, 2)}%) households, {persdiff} ({round(persreldiff, 2)}%) persons, and {tripsdiff} ({round(tripsreldiff, 2)}%) trips.")

    df_persons = pd.merge(
        df_persons, df_households[["edgt_household_id", "household_id", "departement_id"]],
        on = ["edgt_household_id"]
    ).sort_values(by = ["household_id", "edgt_person_id"])
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["edgt_person_id", "edgt_household_id", "person_id", "household_id"]],
        on = ["edgt_person_id", "edgt_household_id"]
    ).sort_values(by = ["household_id", "person_id", "edgt_trip_id"])
    df_trips["trip_id"] = np.arange(len(df_trips))

    for mode, values in MODES_MAP.items():
        df_trips.loc[df_trips["MODP"].isin(values), "mode"] = mode

    # Add weight to trips
    df_trips = pd.merge(
        df_trips, df_persons[["person_id", "POND_PERS"]], on = "person_id", how = "left"
    ).rename(columns = { "POND_PERS": "trip_weight" })

    df_trips["trip_weight"] = df_trips["trip_weight"].str.replace(",", ".").astype(float)


    return df_trips
