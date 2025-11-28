import pandas as pd
import geopandas as gpd
from collections import Counter


def transfer_pairs(modes):
    # Remove 'walk' and consecutive duplicates
    main_modes = [m for m in modes if m != 'walk']
    pairs = [(m1, m2) for m1, m2 in zip(main_modes, main_modes[1:])]
    # Represent pairs as "mode1->mode2"
    return Counter(f"{m1}->{m2}" for m1, m2 in pairs)


def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons")
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")
    trips = context.stage("data.microcensus.trips")[0].copy()
    trips_pt = trips[trips["mode"]=="pt"]

    del trips_pt["mode_detailed"]
    del trips_pt["network_distance"]
    del trips_pt["parking_cost"]
    del trips_pt["crowfly_distance"]
    del trips_pt["activity_duration"]
    del trips_pt["purpose"]

    trips_pt["trip_id"] = trips_pt["person_id"].astype(str) + "_" + trips_pt["trip_id"].astype(str)
    del trips_pt["person_id"]

    df_mz_stages = pd.read_csv(f"{data_path}/microcensus/etappen.csv", encoding = "latin1")
    stages       = df_mz_stages[["HHNR", "WEGNR", "ETNR", "f51300"]]
    stages.columns = ["person_id", "trip_id", "stage_id", "mode_detailed"]

    stages["trip_id"] = stages["person_id"].astype(str) + "_" + stages["trip_id"].astype(str)
    stages_pt         = stages[stages["trip_id"].isin(trips_pt["trip_id"].unique())]

    # First, adjust the modes
    stages_pt.loc[stages_pt["mode_detailed"] == -99, "mode_detailed"] = "other"
    stages_pt.loc[stages_pt["mode_detailed"] == -97, "mode_detailed"] = "other"
    stages_pt.loc[stages_pt["mode_detailed"] == 1,   "mode_detailed"] = "walk"    
    stages_pt.loc[stages_pt["mode_detailed"] == 2,   "mode_detailed"] = "bike"     
    stages_pt.loc[stages_pt["mode_detailed"] == 3,   "mode_detailed"] = "motorbike"     
    stages_pt.loc[stages_pt["mode_detailed"] == 4,   "mode_detailed"] = "motorbike"      
    stages_pt.loc[stages_pt["mode_detailed"] == 5,   "mode_detailed"] = "motorcycle"      
    stages_pt.loc[stages_pt["mode_detailed"] == 6,   "mode_detailed"] = "motorcycle"     
    stages_pt.loc[stages_pt["mode_detailed"] == 7,   "mode_detailed"] = "car"
    stages_pt.loc[stages_pt["mode_detailed"] == 8,   "mode_detailed"] = "car"     
    stages_pt.loc[stages_pt["mode_detailed"] == 9,   "mode_detailed"] = "rail"  
    stages_pt.loc[stages_pt["mode_detailed"] == 10,  "mode_detailed"] = "bus"   
    stages_pt.loc[stages_pt["mode_detailed"] == 11,  "mode_detailed"] = "bus"    
    stages_pt.loc[stages_pt["mode_detailed"] == 12,  "mode_detailed"] = "tram" # Also subway   
    stages_pt.loc[stages_pt["mode_detailed"] == 13,  "mode_detailed"] = "taxi"    
    stages_pt.loc[stages_pt["mode_detailed"] == 14,  "mode_detailed"] = "pt_other" #Reisecar?  
    stages_pt.loc[stages_pt["mode_detailed"] == 15,  "mode_detailed"] = "truck"  
    stages_pt.loc[stages_pt["mode_detailed"] == 16,  "mode_detailed"] = "ferry"    
    stages_pt.loc[stages_pt["mode_detailed"] == 17,  "mode_detailed"] = "plane" 
    stages_pt.loc[stages_pt["mode_detailed"] == 18,  "mode_detailed"] = "cable-car" 
    stages_pt.loc[stages_pt["mode_detailed"] == 19,  "mode_detailed"] = "other" 
    stages_pt.loc[stages_pt["mode_detailed"] == 20,  "mode_detailed"] = "ebike" 
    stages_pt.loc[stages_pt["mode_detailed"] == 21,  "mode_detailed"] = "ebike" 
    stages_pt.loc[stages_pt["mode_detailed"] == 95,  "mode_detailed"] = "other"

    stages_pt = stages_pt.sort_values(['trip_id', 'stage_id'])

    mode_counts = stages_pt.groupby(['trip_id', 'mode_detailed']).size().unstack(fill_value=0)
    result      = mode_counts

    mode_columns = [
        col for col in result.columns 
        if col not in ['index', 'trip_id', 'connections']
    ]

    # For each row, get the modes used (where count > 0), join with '-'
    result['modes_used'] = result[mode_columns].apply(
        lambda row: '-'.join([mode for mode in mode_columns if row[mode] > 0]), axis=1
    )

    # Only keep the main PT modes and remove any access/egress mode which is not walk
    result = result[~result["modes_used"].str.contains("car|motorcycle|truck|other|motorbike|plane|taxi|bike|ebike")]
    for mode in ["car", "motorcycle", "truck", "other", "motorbike", "plane", "taxi", "bike", "ebike"]:
        del result[mode]

    result = result.reset_index()

    stages_pt = stages_pt[stages_pt["trip_id"].isin(result["trip_id"])]

    # Get the list of modes per trip
    modes_per_trip = stages_pt.sort_values(['trip_id','stage_id']).groupby('trip_id')['mode_detailed'].apply(list)

    # Count transfer pairs per trip
    transfer_counts = modes_per_trip.apply(transfer_pairs)
    transfer_df     = pd.DataFrame(list(transfer_counts), index=transfer_counts.index).fillna(0).astype(int).reset_index()

    transfer_columns = [col for col in transfer_df.columns if col != "trip_id"]
    transfer_df["transfers"] = transfer_df[transfer_columns].sum(axis=1)
    
    result = result.merge(transfer_df, on = "trip_id", how="left")

    # Merge with trips
    trips_pt = trips_pt[trips_pt["trip_id"].isin(result["trip_id"])]
    trips_pt = trips_pt.merge(result, on = "trip_id")

    for mode in ["rail", "bus", "tram", "ferry"]:
        trips_pt = trips_pt.rename(columns = {mode: "legs_" + mode})

    df_mz_persons = pd.read_csv(f"{data_path}/microcensus/zielpersonen.csv", sep = ",", encoding = "latin1", parse_dates = ["USTag"])
    df_mz_persons = df_mz_persons[["HHNR", "WP", "tag"]]
    df_mz_persons.columns = ["person_id", "weight", "day_of_the_week"]

    trips_pt.loc[:, "person_id"] = trips_pt["trip_id"].str.split("_").str[0].astype(int)

    trips_pt = trips_pt.merge(df_mz_persons, on = "person_id")
    del trips_pt["person_id"]

    cantons = context.stage("data.spatial.cantons")

    origins      = gpd.GeoSeries.from_xy(trips_pt["origin_x"], trips_pt["origin_y"])
    destinations = gpd.GeoSeries.from_xy(trips_pt["destination_x"], trips_pt["destination_y"])

    origins      = gpd.GeoDataFrame(geometry=origins, crs=cantons.crs)
    destinations = gpd.GeoDataFrame(geometry=destinations, crs=cantons.crs)

    origins_with_canton      = gpd.sjoin(origins, cantons, how="left", predicate="within")
    destinations_with_canton = gpd.sjoin(destinations, cantons, how="left", predicate="within")

    trips_pt["origin_canton"]      = origins_with_canton["canton_name"]
    trips_pt["destination_canton"] = destinations_with_canton["canton_name"]

    trips_pt.to_csv(f"{context.path()}/reference_MZ2015_with_cantons.csv", index = False)

    return trips_pt

