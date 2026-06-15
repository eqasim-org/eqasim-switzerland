import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import geopandas as gpd


def configure(context):
    context.config("data_path")
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.swiss_border")
    context.stage("data.cross_border.interview_places")

    context.config("cross_border_countries", default = "All")
    context.config("cross_border_exclude_shapefiles", default=None)

    context.config("output_path")


def has_point(x):
    return pd.notna(x)


def make_segment_1(row):
    if has_point(row["interview_geometry_point"]):
        return LineString([
            Point(row["origin_x"], row["origin_y"]),
            row["interview_geometry_point"]
        ])
    return LineString([
        Point(row["origin_x"], row["origin_y"]),
        Point(row["destination_x"], row["destination_y"])
    ])


def make_segment_2(row):
    if has_point(row["interview_geometry_point"]):
        return LineString([
            row["interview_geometry_point"],
            Point(row["destination_x"], row["destination_y"])
        ])
    return None


def process_from_to_trips(df_trips, context):
    # Load municipalities
    df_municipalities, _ = context.stage("data.spatial.municipalities")

    # 1. Remove "through" trips that were not classified properly
    trips    = df_trips[(df_trips["origin_country"]=="CH") | (df_trips["destination_country"]=="CH")].copy()
    trips_od = trips[["origin_country", "destination_country", "start_x", "start_y", "end_x", "end_y", "trip_mode", "trip_purpose", "weight", "nb_passengers",
        "interview_place", "interview_point_id", "interview_geometry_point", "group_weight"]].copy()
    
    # 2. Remove trips with missing information on start or end point
    mask_missing_start = pd.to_numeric(trips_od["start_x"], errors="coerce").isna() # If start_x is missing, so is origin_place, so we cannot use one value to compensate the absence of the other.
    mask_missing_end   = pd.to_numeric(trips_od["end_x"], errors="coerce").isna()   # Same with destinations
    
    df = trips_od[~(mask_missing_start) & ~(mask_missing_end)].copy()

    # Reorder start and end so that all trips start in CH
    mask = df["destination_country"] == "CH"
    df.loc[mask, ["origin_country", "destination_country"]] = df.loc[mask, ["destination_country", "origin_country"]].values
    df.loc[mask, ["start_x", "end_x"]] = df.loc[mask, ["end_x", "start_x"]].values
    df.loc[mask, ["start_y", "end_y"]] = df.loc[mask, ["end_y", "start_y"]].values

    # Prepare to sample points from destination municipality
    origins = df.copy().apply(lambda row: Point(row["start_x"], row["start_y"]), axis = 1)
    origins = gpd.GeoSeries(origins, crs = "EPSG:4326").to_crs("EPSG:2056")

    joined = gpd.sjoin(gpd.GeoDataFrame(geometry = origins), df_municipalities, how = "left", predicate = "within")

    df["origin_municipality"] = joined["municipality_id"].values

    # In 23 cases, corresponding mostly to people going to Liechtenstein or to points exactly on the border
    # in le Locle or Saint-Gingolph, the municipality cannot be found. 
    # Let's remove these observations.
    df = df[df["origin_municipality"].notna()].copy()

    # Re-create the destinations
    destinations = df.copy().apply(lambda row: Point(row["end_x"], row["end_y"]), axis = 1)
    destinations = gpd.GeoSeries(destinations, crs = "EPSG:4326").to_crs("EPSG:2056")

    df["destination_x"] = destinations.apply(lambda p : p.x)
    df["destination_y"] = destinations.apply(lambda p : p.y)

    # Re-create the origins
    origins = df.copy().apply(lambda row: Point(row["start_x"], row["start_y"]), axis = 1)
    origins = gpd.GeoSeries(origins, crs = "EPSG:4326").to_crs("EPSG:2056")

    df["origin_x"] = origins.apply(lambda p : p.x)
    df["origin_y"] = origins.apply(lambda p : p.y)

    df["cross_border_person_id"] = range(len(df))
    df["cross_border_person_id"] = "CBS_CH_" + df["cross_border_person_id"].astype(str)

    df["residence_x"] =  df["origin_x"]
    df["residence_y"] =  df["origin_y"]

    df["label"] = "From-To"

    df = df[["cross_border_person_id", "label", "weight",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "residence_x", "residence_y",
        "trip_mode", "trip_purpose",
        "interview_place", "interview_point_id", "interview_geometry_point"]]
    
    seg1 = df.copy()
    seg1["geometry"] = seg1.apply(make_segment_1, axis=1)
    seg1["segment"] = "origin_to_destination"  # or keep "origin_to_interview" if you prefer

    seg2 = df.copy()
    seg2["geometry"] = seg2.apply(make_segment_2, axis=1)
    seg2["segment"] = "interview_to_destination"
    seg2 = seg2[seg2["geometry"].notna()]

    result = gpd.GeoDataFrame(
        pd.concat([seg1, seg2], ignore_index=True),
        geometry="geometry",
        crs="EPSG:2056"
    )

    output_path = context.config("output_path")
    result.to_file(f"{output_path}/swiss_residents_border_crossing.gpkg", driver = "gpkg")
    
    return df


def read_2021_data(context):
    # Load data
    # We are using the 2021 release because the 2015 one doesn't provide reliable destination coordinates.

    data_path = context.config("data_path")
    data_path = f"{data_path}/crossborder/AuGQPV_2021/AGQPV21_finale_Auswertungsdatenbank.csv"

    df2021 = pd.read_csv(data_path, encoding="latin1", sep = ";")

    df2021 = df2021[["INTERVIEWID", "BEFRAGUNGSORTID", "BEFRAGUNGSORT", "GRENZABSCHNITT", "TAGESTYP", "VERKEHRSTRAEGER", "UEBERGANGSART", "FAHRZEUGTYP", "GRUPPENGROESSE",
                 "WOHNORTLANDISO", "WOHNORT_GISCO_ID", "STARTORTLANDISO", "STARTORT_GISCO_ID", "ZIELORTLANDISO", "ZIELORT_GISCO_ID",
                 "STARTORTORTLATITUDE", "STARTORTORTLONGITUDE",
                 "ZIELORTORTLATITUDE", "ZIELORTORTLONGITUDE",
                 "FAHRTZWECK", "ANZAHLUEBERNACHTUNGEN", "AUFENTHALTSLAND1ISO", "AUFENTHALTSLAND2ISO", "AUFENTHALTSLAND3ISO",
                 "ZUGTYP", "FAHRTRICHTUNGGU", "FAHRTRICHTUNGAU", "VERKEHRSART", "GEWICHT_Personen", "GEWICHT_Fahrzeuge"]]

    df2021.columns = ["interview_id", "interview_place_id", "interview_place", "neighbor_country", "day_cat", "road_type", "crossing_cat", "vehicle_type", "nb_passengers",
                  "residence_country", "residence_place", "origin_country", "origin_place", "destination_country", "destination_place",
                  "start_y", "start_x", "end_y", "end_x", 
                  "trip_purpose", "nb_nights", "country1", "country2", "country3",
                  "train_type", "direction_crossing", "direction_alps", "travel_cat", "weight", "weight_vehicles"]
    
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
    df2021.loc[df2021["vehicle_type"]=="1", "vehicle_type"] = "car"
    df2021.loc[df2021["vehicle_type"]=="2", "vehicle_type"] = "car" #"motorcycle"
    df2021.loc[df2021["vehicle_type"]=="3", "vehicle_type"] = "pt"  #"long distance bus"
    
    # 4. Identify trip purpose
    df2021["trip_purpose"] = df2021["trip_purpose"].astype(str)
    df2021.loc[df2021["trip_purpose"]=="1", "trip_purpose"] = "work"
    df2021.loc[df2021["trip_purpose"]=="2", "trip_purpose"] = "education"
    df2021.loc[df2021["trip_purpose"]=="3", "trip_purpose"] = "shop"
    df2021.loc[df2021["trip_purpose"]=="4", "trip_purpose"] = "work"
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

    df2021["weight"]          = df2021["weight"].astype(float)
    df2021["weight_vehicles"] = df2021["weight_vehicles"].replace("", 0).replace(" ", 0).fillna(0).astype(float)
    df2021.loc[df2021["road_type"]=="road", "weight"] = 2 * df2021[df2021["road_type"]=="road"]["weight_vehicles"] 
    
    df2021.loc[:, "group_weight"] = df2021["weight"]
    df2021.loc[df2021["road_type"]=="road", "group_weight"] = df2021[df2021["road_type"]=="road"]["weight"] * df2021[df2021["road_type"]=="road"]["nb_passengers"]
    df2021["group_weight"] = df2021["group_weight"].astype(float)
    
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
    df_days["group_weight"] = df_days["group_weight"] / (52 * day_value)
    df_days["group_weight"] = df_days["group_weight"] / 2
    del df_days["day_cat"]

    # 12. Only select border crossing data - remove Alps crossing data
    borders = df_days[df_days["crossing_cat"]==1].copy()
    
    del borders["crossing_cat"]
    del borders["direction_alps"]

    # 13. Only keep Swiss residents
    residents_ch_mask = borders["residence_country"] == "CH"
    borders = borders[residents_ch_mask].copy()

    for col in ["start_x", "start_y", "end_x", "end_y"]:
        borders[col] = pd.to_numeric(borders[col].str.strip(), errors="coerce")

    borders = borders[borders["start_x"].notna() & borders["start_y"].notna()]
    borders = borders[borders["end_x"].notna() & borders["end_y"].notna()]

    # 14. Match to an interview point    
    points = context.stage("data.cross_border.interview_places").copy()[["interview_place", "border_crossing_point_id", "geometry", "importance"]]
    points["interview_point_id"] = points["border_crossing_point_id"]

    grouped_points = {k: v for k, v in points.groupby("interview_place")}

    def sample_point(row):
        candidates = grouped_points.get(row["interview_place"])
        if candidates is not None and "importance" in candidates.columns:
            sampled = candidates.sample(n=1, weights=candidates["importance"])
            return sampled.iloc[0][["geometry", "importance", "interview_point_id"]]
        
        # Fall back to closest point based on origin coordinates
        origin    = Point(row["start_x"], row["start_y"])
        distances = points["geometry"].apply(lambda geom: origin.distance(geom))
        closest   = points.loc[distances.idxmin()]
        return closest[["geometry", "importance", "interview_point_id"]]

    result = borders[borders["trip_mode"] == "car"][["interview_place", "start_x", "start_y"]].apply(sample_point, axis=1)
    borders[["interview_geometry_point", "candidate_label", "interview_point_id"]] = result

    return borders


def read_2015_data(context):
    data_path = context.config("data_path")
    data_path = f"{data_path}/crossborder/AuGQPV_2015/Finale_Auswertungsdatenbank_AGQPV2015_V2.csv"

    df2015   = pd.read_csv(data_path, encoding = "latin1", sep = ",")

    df2015   = df2015[["INTERVIEWID", "BEFRAGUNGSORTID", "BEFRAGUNGSORT", "GRENZABSCHNITT", "TAGESTYP", 
                    "VERKEHRSTRAEGER", "UEBERGANGSART", "FAHRZEUGTYP", "GRUPPENGROESSE", 
                    "WOHNORTLANDISO", "STARTORTLANDISO", "ZIELORTLANDISO", 
                    "STARTORTORTLATITUDE", "STARTORTORTLONGITUDE",
                    "ZIELORTORTLATITUDE","ZIELORTORTLONGITUDE",
                    "FAHRTZWECK", "ANZAHLUEBERNACHTUNGEN",
                    "AUFENTHALTSLAND1ISO", "AUFENTHALTSLAND2ISO", "AUFENTHALTSLAND3ISO",
                    "ZUGTYP", "FAHRTRICHTUNGGU", "FAHRTRICHTUNGAU", "VERKEHRSART", "GEWICHT"]]
    
    df2015.columns = ["interview_id", "interview_place_id", "interview_place", "neighbor_country", "day_cat", 
                  "road_type", "crossing_cat", "vehicle_type", "nb_passengers",
                  "residence_country", "origin_country", "destination_country", 
                  "start_y", "start_x", 
                  "end_y", "end_x", 
                  "trip_purpose", "nb_nights", 
                  "country1", "country2", "country3",
                  "train_type", "direction_crossing", "direction_alps", "travel_cat", "weight"]
    
    swiss_neighbors = ['CH', 'FR', 'DE', 'IT', 'AT', 'LI']
    for column in ["residence_country", "origin_country", "destination_country"]:
        df2015.loc[:, column] = df2015[column].apply(lambda x: x if x in swiss_neighbors else "other")

    df2015["road_type"] = df2015["road_type"].astype(str)
    df2015.loc[df2015["road_type"]=="1", "road_type"] = "road"
    df2015.loc[df2015["road_type"]=="2", "road_type"] = "rail"

    df2015["vehicle_type"] = df2015["vehicle_type"].astype(str)
    df2015.loc[df2015["vehicle_type"]=="1", "vehicle_type"] = "car"
    df2015.loc[df2015["vehicle_type"]=="3", "vehicle_type"] = "car"
    df2015.loc[df2015["vehicle_type"]=="4", "vehicle_type"] = "pt"

    df2015["trip_purpose"] = df2015["trip_purpose"].astype(str)
    df2015.loc[df2015["trip_purpose"]=="1", "trip_purpose"] = "work"
    df2015.loc[df2015["trip_purpose"]=="2", "trip_purpose"] = "education"
    df2015.loc[df2015["trip_purpose"]=="3", "trip_purpose"] = "shop"
    df2015.loc[df2015["trip_purpose"]=="4", "trip_purpose"] = "work"
    df2015.loc[df2015["trip_purpose"]=="5", "trip_purpose"] = "leisure"
    df2015.loc[df2015["trip_purpose"]=="6", "trip_purpose"] = "other"
    df2015.loc[df2015["trip_purpose"]=="7", "trip_purpose"] = "freight"

    df2015["direction_crossing"] = df2015["direction_crossing"].astype(str)
    df2015.loc[df2015["direction_crossing"]=="1", "direction_crossing"] = "entering CH"
    df2015.loc[df2015["direction_crossing"]=="2", "direction_crossing"] = "leaving CH"

    df2015["direction_alps"] = df2015["direction_alps"].astype(str)
    df2015.loc[df2015["direction_alps"]=="1", "direction_alps"] = "North"
    df2015.loc[df2015["direction_alps"]=="2", "direction_alps"] = "South"

    df2015["travel_cat"] = df2015["travel_cat"].astype(str)
    df2015.loc[df2015["travel_cat"]=="1", "travel_cat"] = "Within CH"
    df2015.loc[df2015["travel_cat"]=="2", "travel_cat"] = "Through CH"
    df2015.loc[df2015["travel_cat"]=="3", "travel_cat"] = "From CH"
    df2015.loc[df2015["travel_cat"]=="4", "travel_cat"] = "To CH"
    df2015.loc[~df2015["travel_cat"].isin(["Within CH", "Through CH", "From CH", "To CH"]), "travel_cat"]  = "Unknown"

    df2015.loc[df2015["day_cat"]=="Werktag", "day_cat"]    = "Mo-Fr"
    df2015.loc[df2015["day_cat"]=="Samstag", "day_cat"]    = "WE"
    df2015.loc[df2015["day_cat"]=="Sonntag", "day_cat"]    = "WE"
    df2015.loc[df2015["day_cat"]=="Wochenende", "day_cat"] = "WE"

    df2015.loc[df2015["train_type"]=="FV", "train_type"] = "pt"
    df2015.loc[df2015["train_type"]=="RV", "train_type"] = "pt"

    df2015["trip_mode"] = np.where(df2015["vehicle_type"].str.strip() != '', df2015["vehicle_type"], df2015["train_type"])

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
    df_days   = df2015[df2015["day_cat"]==day_key].copy()
    df_days["weight"] = df_days["weight"] / (52 * day_value)
    df_days["weight"] = df_days["weight"] / 2 # Because the persons entering the country have to leave it too
    del df_days["day_cat"]

    df2015 = df_days.copy()

    df2015.loc[df2015["road_type"]=="road", "weight"] = df2015[df2015["road_type"]=="road"]["weight"] * 2
    df2015.loc[:, "group_weight"] = df2015["weight"] 
    df2015.loc[df2015["road_type"]=="road", "group_weight"] = df2015[df2015["road_type"]=="road"]["weight"] * df2015[df2015["road_type"]=="road"]["nb_passengers"]

    del df2015["vehicle_type"]
    del df2015["train_type"]

    borders2015 = df2015[df2015["crossing_cat"]==1]

    del borders2015["crossing_cat"]
    del borders2015["direction_alps"]

    residents_ch_mask = borders2015["residence_country"] == "CH"
    borders2015 = borders2015[residents_ch_mask].copy()

    return borders2015


def execute(context):
    borders2021 = read_2021_data(context)    
    borders2015 = read_2015_data(context)

    grouped2021 = borders2021.groupby(["trip_mode", "trip_purpose", "origin_country", "destination_country"], as_index = False)["group_weight"].sum().rename(columns = {"group_weight": "group_weight_2021"})
    grouped2015 = borders2015.groupby(["trip_mode", "trip_purpose", "origin_country", "destination_country"], as_index = False)["group_weight"].sum().rename(columns = {"group_weight": "group_weight_2015"})

    grouped = grouped2021.merge(grouped2015, how = "left", on = ["trip_mode", "trip_purpose", "origin_country", "destination_country"])

    grouped.loc[grouped["group_weight_2015"].isna(), "scaling_factor"]  = 1
    grouped.loc[~grouped["group_weight_2015"].isna(), "scaling_factor"] = grouped[~grouped["group_weight_2015"].isna()]["group_weight_2015"] / grouped[~grouped["group_weight_2015"].isna()]["group_weight_2021"]
    grouped = grouped[["trip_mode", "trip_purpose", "origin_country", "destination_country", "scaling_factor"]]

    borders = borders2021.merge(
        grouped,
        on=["trip_mode", "trip_purpose", "origin_country", "destination_country"],
        how="left"
    )

    borders["weight"] = (
        borders["weight"] *
        borders["scaling_factor"]
    )

    borders["group_weight"] = (
        borders["group_weight"] *
        borders["scaling_factor"]
    )

    # Now process the trips
    trips = borders[borders["travel_cat"].isin(["From CH", "To CH"])]   
    from_to_trips = process_from_to_trips(trips, context)

    return from_to_trips
