import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import random

def configure(context):
    context.config("data_path")
    context.config("weekend", default = False)

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.swiss_border")


def sample_rows_by_weight(df, weight_col='weight'):
    df = df.copy()
    
    # Separate integer and fractional parts
    df['int_part'] = df[weight_col].astype(int)
    df['frac_part'] = df[weight_col] - df['int_part']
    
    # Repeat rows according to integer part
    repeated = df.loc[df.index.repeat(df['int_part'])].drop(columns=['int_part', 'frac_part'])
    
    # Handle fractional part with Bernoulli sampling
    fractional_mask = np.random.rand(len(df)) < df['frac_part']
    fractional = df[fractional_mask].drop(columns=['int_part', 'frac_part'])
    
    # Combine both
    sampled_df = pd.concat([repeated, fractional], ignore_index=True)
    return sampled_df


def sample_points_in_polygon(polygon, n):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < n:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            points.append(p)
    return points


def execute(context):
    # Load municipalities
    df_municipalities, _ = context.stage("data.spatial.municipalities")

    # Load data
    # We are using the 2021 release because the 2015 one doesn't provide reliable destination coordinates.
    # Obviously there are some covid-related biases but this is the best we currently have.
    data_path = context.config("data_path")
    data_path = f"{data_path}/crossborder/AuGQPV_2021/AGQPV21_finale_Auswertungsdatenbank.csv"

    df2021 = pd.read_csv(data_path, encoding="latin1", sep = ";")

    df2021 = df2021[["INTERVIEWID", "BEFRAGUNGSORTID", "BEFRAGUNGSORT", "GRENZABSCHNITT", "TAGESTYP", "VERKEHRSTRAEGER", "UEBERGANGSART", "FAHRZEUGTYP", "GRUPPENGROESSE",
                 "WOHNORTLANDISO", "WOHNORT_GISCO_ID", "STARTORTLANDISO", "STARTORT_GISCO_ID", "ZIELORTLANDISO", "ZIELORT_GISCO_ID",
                 "STARTORTORTLATITUDE", "STARTORTORTLONGITUDE",
                 "ZIELORTORTLATITUDE", "ZIELORTORTLONGITUDE",
                 "FAHRTZWECK", "ANZAHLUEBERNACHTUNGEN", "AUFENTHALTSLAND1ISO", "AUFENTHALTSLAND2ISO", "AUFENTHALTSLAND3ISO",
                 "ZUGTYP", "FAHRTRICHTUNGGU", "FAHRTRICHTUNGAU", "VERKEHRSART", "GEWICHT_Personen"]]

    df2021.columns = ["interview_id", "interview_place_id", "interview_place", "neighbor_country", "day_cat", "road_type", "crossing_cat", "vehicle_type", "nb_passengers",
                  "residence_country", "residence_place", "origin_country", "origin_place", "destination_country", "destination_place",
                  "start_y", "start_x", "end_y", "end_x", 
                  "trip_purpose", "nb_nights", "country1", "country2", "country3",
                  "train_type", "direction_crossing", "direction_alps", "travel_cat", "weight"]

    # Process the columns
    # 1. Rename countries
    swiss_neighbors = ["CH", "FR", "DE", "IT", "AT", "LI"]
    for column in ["residence_country", "origin_country", "destination_country"]:
        df2021.loc[:, column] = df2021[column].apply(lambda x: x if x in swiss_neighbors else "other")
    
    # 2. Separate road and rail observations
    df2021["road_type"] = df2021["road_type"].astype(str)
    df2021.loc[df2021["road_type"]=="1", "road_type"] = "road"
    df2021.loc[df2021["road_type"]=="2", "road_type"] = "rail"
    
    # 3. Identify vehicles
    df2021["vehicle_type"] = df2021["vehicle_type"].astype(str)
    df2021.loc[df2021["vehicle_type"]=="1", "vehicle_type"] = "MIV"
    df2021.loc[df2021["vehicle_type"]=="2", "vehicle_type"] = "car" #"motorcycle"
    df2021.loc[df2021["vehicle_type"]=="3", "vehicle_type"] = "pt"  #"long distance bus"
    
    # 4. Identify trip purpose
    df2021["trip_purpose"] = df2021["trip_purpose"].astype(str)
    df2021.loc[df2021["trip_purpose"]=="1", "trip_purpose"] = "work"
    df2021.loc[df2021["trip_purpose"]=="2", "trip_purpose"] = "education"
    df2021.loc[df2021["trip_purpose"]=="3", "trip_purpose"] = "shop"
    df2021.loc[df2021["trip_purpose"]=="4", "trip_purpose"] = "work_secondary"
    df2021.loc[df2021["trip_purpose"]=="5", "trip_purpose"] = "leisure"
    df2021.loc[df2021["trip_purpose"]=="6", "trip_purpose"] = "other"
    df2021.loc[df2021["trip_purpose"]=="7", "trip_purpose"] = "freight"
    
    # 5. Is the respondent entering or leaving CH?
    df2021["direction_crossing"] = df2021["direction_crossing"].astype(str)
    df2021.loc[df2021["direction_crossing"]=="1", "direction_crossing"] = "entering CH"
    df2021.loc[df2021["direction_crossing"]=="2", "direction_crossing"] = "leaving CH"
    
    # 6. For the Alps crossing, not used currently
    df2021["direction_alps"] = df2021["direction_alps"].astype(str)
    df2021.loc[df2021["direction_alps"]=="1", "direction_alps"] = "North"
    df2021.loc[df2021["direction_alps"]=="2", "direction_alps"] = "South"
    
    # 7. More detailed compared to direction_crossing, but obviously there are inconsistencies betweeen these two columns
    df2021["travel_cat"] = df2021["travel_cat"].astype(str)
    df2021.loc[df2021["travel_cat"]=="1", "travel_cat"] = "Within CH"
    df2021.loc[df2021["travel_cat"]=="2", "travel_cat"] = "Through CH"
    df2021.loc[df2021["travel_cat"]=="3", "travel_cat"] = "From CH"
    df2021.loc[df2021["travel_cat"]=="4", "travel_cat"] = "To CH"
    
    # 8. Identify the observation day
    df2021.loc[df2021["day_cat"]=="Werktag", "day_cat"]    = "Mo-Fr"
    df2021.loc[df2021["day_cat"]=="Samstag", "day_cat"]    = "WE"
    df2021.loc[df2021["day_cat"]=="Sonntag", "day_cat"]    = "WE"
    df2021.loc[df2021["day_cat"]=="Wochenende", "day_cat"] = "WE"
    
    # 9. Identify the train category for rail observations
    df2021.loc[df2021["train_type"]=="1", "train_type"] = "pt" #"long distance train"
    df2021.loc[df2021["train_type"]=="2", "train_type"] = "pt" #"regional train"

    # 10. Aggregate vehicle_type and train_type
    df2021["trip_mode"] = np.where(df2021["vehicle_type"].str.strip() != '', df2021["vehicle_type"], df2021["train_type"])
    
    del df2021["road_type"]
    del df2021["vehicle_type"]
    del df2021["train_type"]

    # 11. Adjust weight
    # Not sure how we should include the number of passengers here. Is the weight corresponding to the entire group? To one single person?
    # Deciding that the weight weighs the entire group leads to the best approximation of daily border crossings - 2.2 persons according to https://www.bazg.admin.ch/bazg/de/home/das-bazg/fakten-und-zahlen/ein-tag-an-der-grenze.html 
    df2021["weight"] = df2021["weight"] #/ df2021["nb_passengers"]
    df2021["weight"] = df2021["weight"] / 2 # Because the persons entering the country have to leave it too

    # 12. Only select border crossing data - remove Alps crossing data
    borders = df2021[df2021["crossing_cat"]==1]
    alps    = df2021[df2021["crossing_cat"]==2]
    
    del borders["crossing_cat"]
    del alps["crossing_cat"]
    del borders["direction_alps"]

    # 13. Remove Swiss residents, their mobility should be covered in the Microcensus
    residents_ch_mask = borders["residence_country"] == "CH"

    borders = borders[~ residents_ch_mask]

    # 76.5% are entering CH, 3.1% are leaving CH, 20.3% are going through CH,
    # 0.1% is traffic within CH.
    # Not sure why there is this dissimetry between people entering and leaving the country.
    # Differences in where the data was collected? Wrong categorization of responses? 
    # Unclear question formulation?
    # We decided to thus group together people entering and leaving CH, as long as they are not CH residents.
    # Here we will start by focusing on traffic from/to CH, we will add traffic through CH later.
    # And we do not include traffic within CH.
    #print(borders.groupby(["travel_cat"])["weight"].sum() / np.sum(borders["weight"]) * 100)

    # TODO process people crossing Switzerland
    through = borders[borders["travel_cat"]=="Through CH"]

    # Now process the trips
    trips = borders[borders["travel_cat"].isin(["From CH", "To CH"])]    

    # 1. Remove "through" trips that were not classified properly
    trips    = trips[(trips["origin_country"]=="CH") | (trips["destination_country"]=="CH")]
    trips_od = trips[["origin_country", "destination_country", "start_x", "start_y", "end_x", "end_y", "trip_mode", "day_cat", "trip_purpose", "weight", "nb_passengers"]]

    # 2. Remove trips with missing information on start or end point
    mask_missing_start = (trips_od["start_x"].str.strip() == "") # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = (trips_od["end_x"].str.strip() == "")   # Same with destinations
    
    trips_od_with_info = trips_od[~(mask_missing_start) & ~(mask_missing_end)]

    # This removes 1.49% of the records
    # print(trips_od_with_info["weight"].sum() / trips_od["weight"].sum() * 100)

    # 3. Adjust the weight
    df                  = trips_od_with_info[trips_od_with_info["day_cat"]=="Mo-Fr"].copy()
    df.loc[:, "weight"] = df["weight"] / (5*52) # Approximation of the number of weekend days per year

    if context.config("weekend"):
        df                  = trips_od_with_info[trips_od_with_info["day_cat"]=="WE"]
        df.loc[:, "weight"] = df["weight"] / (2*52) # Approximation of the number of weekend days per year

    # Because the number of trips from/to CH is assymetric, reorder start and end so that all trips end in CH
    mask = df["origin_country"] == "CH"
    df.loc[mask, ["origin_country", "destination_country"]] = df.loc[mask, ["destination_country", "origin_country"]].values
    df.loc[mask, ["start_x", "end_x"]] = df.loc[mask, ["end_x", "start_x"]].values
    df.loc[mask, ["start_y", "end_y"]] = df.loc[mask, ["end_y", "start_y"]].values

    # Prepare to sample points from destination municipality
    destinations = df.apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326")
    destinations = destinations.to_crs("EPSG:2056")

    joined = gpd.sjoin(gpd.GeoDataFrame(geometry = destinations), df_municipalities, how='left', predicate='within')

    df.loc[:, "destination_municipality"] = joined["municipality_id"].values

    # In 23 cases, corresponding mostly to people going to Liechtenstein or to points exactly on the border
    # in le Locle or Saint-Gingolph, the municipality cannot be found. 
    # Let's remove these observations.
    df = df[df["destination_municipality"].notna()]

    # Adjust trip mode when sharing the vehicle
    df_expanded = df.loc[df.index.repeat(df["nb_passengers"])]
    df_expanded["passenger_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]==1), "trip_mode"] = "car"
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]>1), "trip_mode"]  = "car_passenger"
    
    del df_expanded["nb_passengers"]
    del df_expanded["passenger_index"]

    # Sample

    df_sampled = sample_rows_by_weight(df_expanded, weight_col = "weight")

    del df_sampled["end_x"]
    del df_sampled["end_y"]

    # Sample random point in destination municipality. First join the geometry, then sample 100 points per municipality
    # and select randomly one of them as the destination
    df_sampled['destination_municipality'] = df_sampled['destination_municipality'].astype(int)
    df_municipalities['municipality_id']   = df_municipalities['municipality_id'].astype(int)
    
    df_sampled = df_sampled.merge(df_municipalities[['municipality_id', 'geometry']],
                  left_on='destination_municipality',
                  right_on='municipality_id',
                  how='left')

    samples_per_municipality = 100
    municipality_points      = {}
    
    for _, row in df_municipalities.iterrows():
        code = row['municipality_id']
        geom = row['geometry']
        municipality_points[code] = sample_points_in_polygon(geom, samples_per_municipality)

    def assign_random_point(muni_code):
        candidates = municipality_points.get(muni_code, [])
        if candidates:
            return random.choice(candidates)
        else:
            return None  
    
    df_sampled['destination_point'] = df_sampled['destination_municipality'].apply(assign_random_point)

    del df_sampled["geometry"]
    del df_sampled["day_cat"]
    del df_sampled["municipality_id"]
    del df_sampled["destination_municipality"]
    del df_sampled["weight"]

    df = df_sampled.copy().reset_index()

    # Fix the origins
    origins = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["start_x"], df["start_y"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2056")

    origins["record"] = range(len(origins))
    ch_borders        = context.stage("data.spatial.swiss_border")[0]
    ch_borders_simple = ch_borders.simplify(50)

    origins["dist_to_ch"] =  origins.geometry.apply(lambda g: g.distance(ch_borders_simple)) / 1000

    close_mask = origins["dist_to_ch"] < 20
    far_mask = ~close_mask

    far_points            = origins[far_mask]
    close_points          = origins[close_mask]
    close_points_registry = close_points.copy().drop_duplicates(subset = ["geometry"], keep = "first")
    merging_aux_df        = close_points_registry.copy().rename(columns = {"geometry": "close_point_geometry"})
    del merging_aux_df["dist_to_ch"]    

    nearest = far_points.sjoin_nearest(close_points_registry[["geometry", "record"]], how="left")
    del nearest["index_right"]  
    nearest = pd.merge(nearest, merging_aux_df, left_on = "record_right", right_on = "record", how = "left")
    nearest = nearest[["record_left", "dist_to_ch", "close_point_geometry", "geometry"]]
    nearest.columns = ["record", "dist_to_ch", "geometry", "geometry_before_projection"]
    nearest["origin_purpose"] = "other"

    close_points["geometry_before_projection"] = close_points["geometry"]
    close_points["origin_purpose"]             = "home"

    origins = pd.concat([nearest, close_points])
    origins = origins.sort_values(by="record")

    df["origin_point"]                          = origins["geometry"].values
    df["origin_before_projection_to_ch_border"] = origins["geometry_before_projection"].values
    df["origin_purpose"]                        = origins["origin_purpose"].values

    del df["start_x"]
    del df["start_y"]

    df["cross_border_person_id"] = range(len(df))
    df["cross_border_person_id"] = "CBS_" + df["cross_border_person_id"].astype(str)

    df["origin_x"] =  df["origin_point"].apply(lambda p: p.x)
    df["origin_y"] =  df["origin_point"].apply(lambda p: p.y)

    df["destination_x"] =  df["destination_point"].apply(lambda p: p.x)
    df["destination_y"] =  df["destination_point"].apply(lambda p: p.y)

    df["residence_x"] =  df["origin_before_projection_to_ch_border"].apply(lambda p: p.x)
    df["residence_y"] =  df["origin_before_projection_to_ch_border"].apply(lambda p: p.y)

    df = df[["cross_border_person_id",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose"]]

    return df
