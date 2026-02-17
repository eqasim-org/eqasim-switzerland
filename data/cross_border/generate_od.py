import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import random
from pathlib import Path

def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.swiss_border")

    context.config("cross_border_countries", default = "All")
    context.config("cross_border_exclude_shapefiles", default=None)


def sample_rows_by_weight(df2, weight_col="weight"):
    df = df2.copy()
    
    # Separate integer and fractional parts
    df["int_part"]  = df[weight_col].astype(int)
    df["frac_part"] = df[weight_col] - df["int_part"]
    
    # Repeat rows according to integer part
    repeated = df.iloc[np.repeat(np.arange(len(df)), df["int_part"])].copy().drop(columns=["int_part", "frac_part"])
    
    # Handle fractional part with Bernoulli sampling
    fractional_mask = np.random.rand(len(df)) < df["frac_part"]
    fractional = df[fractional_mask].drop(columns=["int_part", "frac_part"])
    
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


def project_point_series_close_to_border(df, x, y, distance_threshold, default_purpose, projected_purpose, column_name, context):
    df = df.copy()
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df[x], df[y]), crs = "EPSG:4326").to_crs("EPSG:2056")
    points["record"] = range(len(points))

    ch_borders        = context.stage("data.spatial.swiss_border").copy()[0]
    ch_borders_simple = ch_borders.simplify(50)

    points["dist_to_border"] = points.geometry.apply(lambda g: g.distance(ch_borders_simple)) / 1000
    close_mask               = points["dist_to_border"] < distance_threshold
    far_mask                 = ~close_mask

    far_points   = points[far_mask].copy()
    close_points = points[close_mask].copy()

    close_points_registry = close_points.copy().drop_duplicates(subset = ["geometry"], keep = "first")
    merging_aux_df        = close_points_registry.copy().rename(columns = {"geometry": "close_point_geometry"})
    del merging_aux_df["dist_to_border"]     

    nearest = far_points.sjoin_nearest(close_points_registry[["geometry", "record"]], how="left")
    del nearest["index_right"]  
    nearest = pd.merge(nearest, merging_aux_df, left_on = "record_right", right_on = "record", how = "left")
    nearest = nearest[["record_left", "dist_to_border", "close_point_geometry", "geometry"]]
    nearest.columns = ["record", "dist_to_border", "geometry", "geometry_before_projection"]
    nearest["purpose"] = projected_purpose

    close_points["geometry_before_projection"] = close_points["geometry"]
    close_points["purpose"]             = default_purpose

    points = pd.concat([nearest, close_points])
    points = points.sort_values(by = "record")

    df[column_name + "_point"]             = points["geometry"].values
    df[column_name + "_before_projection"] = points["geometry_before_projection"].values
    df[column_name + "_purpose"]           = points["purpose"].values
    
    del df[x]
    del df[y]

    df[column_name +  "_x"] = df[column_name + "_point"].apply(lambda p : p.x)
    df[column_name +  "_y"] = df[column_name + "_point"].apply(lambda p : p.y)

    return df


def expand_and_sample(df, expand_column, weight_column):
    df = df.copy()

    # Expand
    df_expanded = df.loc[df.index.repeat(df[expand_column])].copy()
    df_expanded["passenger_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]==1), "trip_mode"] = "car"
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]>1), "trip_mode"]  = "car_passenger"
    
    del df_expanded[expand_column]
    del df_expanded["passenger_index"]

    # Sample
    df_sampled = sample_rows_by_weight(df_expanded, weight_col = weight_column)
    del df_sampled[weight_column]

    return df_sampled.copy().reset_index()


def process_from_to_trips(df_trips, context):
    # Load municipalities
    df_municipalities, _ = context.stage("data.spatial.municipalities")

    # 1. Remove "through" trips that were not classified properly
    trips    = df_trips[(df_trips["origin_country"]=="CH") | (df_trips["destination_country"]=="CH")].copy()
    trips_od = trips[["origin_country", "destination_country", "start_x", "start_y", "end_x", "end_y", "trip_mode", "trip_purpose", "weight", "nb_passengers"]].copy()

    # 2. Remove trips with missing information on start or end point
    mask_missing_start = (trips_od["start_x"].str.strip() == "") # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = (trips_od["end_x"].str.strip() == "")   # Same with destinations
    
     # This removes 1.49% of the records
    df = trips_od[~(mask_missing_start) & ~(mask_missing_end)].copy()

    # Reorder start and end so that all trips end in CH
    mask = df["origin_country"] == "CH"
    df.loc[mask, ["origin_country", "destination_country"]] = df.loc[mask, ["destination_country", "origin_country"]].values
    df.loc[mask, ["start_x", "end_x"]] = df.loc[mask, ["end_x", "start_x"]].values
    df.loc[mask, ["start_y", "end_y"]] = df.loc[mask, ["end_y", "start_y"]].values

    # Prepare to sample points from destination municipality
    destinations = df.copy().apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326").to_crs("EPSG:2056")

    joined = gpd.sjoin(gpd.GeoDataFrame(geometry = destinations), df_municipalities, how = "left", predicate = "within")

    df["destination_municipality"] = joined["municipality_id"].values

    # In 23 cases, corresponding mostly to people going to Liechtenstein or to points exactly on the border
    # in le Locle or Saint-Gingolph, the municipality cannot be found. 
    # Let's remove these observations.
    df = df[df["destination_municipality"].notna()].copy()

    df = expand_and_sample(df.copy(), "nb_passengers", "weight")

    # Fix the origins
    df = project_point_series_close_to_border(df.copy(), "start_x", "start_y", 20, "home", "other", "origin", context)

    # Re-create the destinations
    destinations = df.copy().apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326").to_crs("EPSG:2056")

    df["destination_x"] = destinations.apply(lambda p : p.x)
    df["destination_y"] = destinations.apply(lambda p : p.y)

    df["cross_border_person_id"] = range(len(df))
    df["cross_border_person_id"] = "CBS_" + df["cross_border_person_id"].astype(str)

    df["residence_x"] =  df["origin_before_projection"].apply(lambda p: p.x)
    df["residence_y"] =  df["origin_before_projection"].apply(lambda p: p.y)

    df["label"] = "From-To"

    df = df[["cross_border_person_id", "label",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose"]]
    
    return df


def process_through_trips(through_trips, N, context):
    through_od = through_trips[
        ["origin_country", "destination_country", "start_x", "start_y", "end_x", "end_y", "trip_mode", "trip_purpose", "weight", "nb_passengers"]
    ]

    mask_missing_start = (through_od["start_x"].str.strip() == "") # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = (through_od["end_x"].str.strip() == "")    

    df = through_od[~(mask_missing_start) & ~(mask_missing_end)].copy() # This removes 9% / 11.3% of the records (unweighted/weighted)

    df_expanded = df.loc[df.index.repeat(df["nb_passengers"])].copy()
    df_expanded["passenger_index"] = df_expanded.groupby(df_expanded.index).cumcount() + 1
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]==1), "trip_mode"] = "car"
    df_expanded.loc[(df_expanded["trip_mode"]=="MIV") & (df_expanded["passenger_index"]>1), "trip_mode"]  = "car_passenger"

    del df_expanded["nb_passengers"]
    del df_expanded["passenger_index"]

    df_sampled = sample_rows_by_weight(df_expanded, weight_col = "weight")
    del df_sampled["weight"]

    df = df_sampled.copy().reset_index()

    df = project_point_series_close_to_border(df.copy(), "start_x", "start_y", 20, "other", "other", "origin", context)
    df = project_point_series_close_to_border(df.copy(), "end_x", "end_y", 20, "other", "other", "destination", context)

    df["cross_border_person_id"] = range(N, N + len(df))
    df["cross_border_person_id"] = "CBS_" + df["cross_border_person_id"].astype(str)

    df["residence_x"] =  df["origin_before_projection"].apply(lambda p: p.x)
    df["residence_y"] =  df["origin_before_projection"].apply(lambda p: p.y)

    df["label"] = "Through"

    df = df[["cross_border_person_id", "label",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose"]]

    return df


def execute(context):
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
    days    = {"Mo-Fr": 5, "WE": 2}

    day = context.config("specific_day_scenario")

    if day == "weekend":
        day_key = "WE"
    elif day == "workday":
        day_key = "Mo-Fr"
    elif day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        day_key = "Mo-FR"
    elif day in ["Saturday", "Sunday"]:
        day_key = "WE"

    day_value = days[day_key]
    df_days   = df2021[df2021["day_cat"]==day_key].copy()
    df_days["weight"] = df_days["weight"] / (52 * day_value)
    df_days["weight"] = df_days["weight"] / 2 # Because the persons entering the country have to leave it too
    del df_days["day_cat"]

    # 12. Only select border crossing data - remove Alps crossing data
    borders = df_days[df_days["crossing_cat"]==1].copy()
    
    del borders["crossing_cat"]
    del borders["direction_alps"]

    # 13. Remove Swiss residents, their mobility should be covered in the Microcensus
    residents_ch_mask = borders["residence_country"] == "CH"
    borders = borders[~ residents_ch_mask].copy()

    # 14. Selector by origin country
    allowed_countries  = ["FR", "DE", "AT", "LI", "IT"]
    selected_countries = context.config("cross_border_countries")

    if selected_countries != "All":
        if isinstance(selected_countries, list):
            selected_countries = [c for c in selected_countries if c in allowed_countries]
            if not selected_countries:
                raise ValueError(
                    f"No valid countries in selection. Must be within {allowed_countries}."
                )
            
            borders = borders.loc[borders["origin_country"].within(selected_countries)].copy()

        elif isinstance(selected_countries, str):
            if selected_countries not in allowed_countries:
                raise ValueError(
                    f"Invalid country code '{selected_countries}'. Must be one of {allowed_countries}."
                )
            borders = borders.loc[borders["origin_country"] == selected_countries].copy()

        else:
            raise TypeError("cross_border_countries must be a list, string, or 'All'.")

    # 15. Now process the trips
    trips = borders[borders["travel_cat"].isin(["From CH", "To CH"])]   
    from_to_trips = process_from_to_trips(trips, context)

    through = borders[borders["travel_cat"]=="Through CH"]
    through_trips = process_through_trips(through, len(from_to_trips), context)

    df = pd.concat([from_to_trips, through_trips])
        
    # 16. Remove trips starting in the spatial file to be excluded
    exclude_file = context.config("cross_border_exclude_shapefiles")

    if not exclude_file is None:
        if isinstance(exclude_file, (str, Path)):
            exclude_file = [exclude_file]

        if not isinstance(exclude_file, (list, tuple)):
            raise TypeError(
                "cross_border_exclude_shapefiles must be a path or a list of paths."
            )

        gdfs = []   

        for f in exclude_file:
            suffix = Path(f).suffix.lower()
            if suffix not in {".gpkg", ".shp"}:
                raise TypeError(
                    f"{f} is not a .gpkg or .shp file."
                )

            gdf = gpd.read_file(f).to_crs("EPSG:2056")
            gdfs.append(gdf)

        exclude_region = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2056")
        origins        = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["origin_x"], df["origin_y"]), crs="EPSG:2056")

        joined           = gpd.sjoin(origins, exclude_region[["geometry"]], how = "left", predicate = "within")
        is_within_region = joined["index_right"].notna()
        df["exclude"]    = is_within_region.values

        excluded_ids = df.loc[df["exclude"], "cross_border_person_id"].unique()
        df = df[~df["cross_border_person_id"].isin(excluded_ids)].copy()

        print(df.head())
             

    return df
