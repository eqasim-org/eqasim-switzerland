import re
import pandas as pd
import geopandas as gpd
from shapely import LineString


MISSING_STATION_IDS = [
    {"page" : 29, "name" : "Le Crêt-du-Locle", "new_id" : "04315"},
    {"page" : 26, "name" : "La Neuveville", "new_id" : "04226"},
    {"page" : 33, "name" : "Fribourg/Freiburg Poya", "new_id" : "19133"},
    {"page" : 60, "name" : "Niederurnen, Ziegelbrückstr.", "new_id" : "87509"}
]

IS_NUMBER_PATTERN       = re.compile("^[0-9]+$")


def visualize_triangles(df_triangles, gtfs_stops, output_dir):

    gtfs_stops["stop_id"] = gtfs_stops["stop_id"].astype(int)
    
    df_triangles = df_triangles.merge(gtfs_stops[["stop_id", "stop_name", "geometry"]].rename(columns = {"geometry": "origin_geometry", "stop_name": "origin_name"}),
                                      how = "left",
                                      left_on = "origin_id", right_on = "stop_id")
    
    df_triangles = df_triangles.merge(gtfs_stops[["stop_id", "stop_name", "geometry"]].rename(columns = {"geometry": "destination_geometry", "stop_name": "destination_name"}),
                                      how = "left",
                                      left_on = "destination_id", right_on = "stop_id")
    
    del df_triangles["stop_id_x"]
    del df_triangles["stop_id_y"]

    df_triangles = df_triangles[(df_triangles["origin_geometry"].notna()) & (df_triangles["destination_geometry"].notna())]

    df_triangles = df_triangles[df_triangles["origin_id"] != df_triangles["destination_id"]]

    origins      = gpd.GeoSeries(df_triangles["origin_geometry"].values, crs = "EPSG:2056")
    destinations = gpd.GeoSeries(df_triangles["destination_geometry"].values, crs = "EPSG:2056")

    lines = gpd.GeoSeries(
                [LineString([orig, dest]) for orig, dest in zip(origins, destinations)],
                crs="EPSG:2056"
            )
    
    origins      = df_triangles["origin_geometry"].values
    destinations = df_triangles["destination_geometry"].values
    
    del df_triangles["origin_geometry"]
    del df_triangles["destination_geometry"]

    df_triangles["geometry"] = lines.values

    df_triangles = gpd.GeoDataFrame(df_triangles, crs = "EPSG:2056")
    df_triangles.to_file(f"{output_dir}/t603/triangles.shp")

    all_points    = list(origins) + list(destinations)
    unique_points = list({pt.wkt: pt for pt in all_points}.values())
    gdf_points    = gpd.GeoDataFrame(geometry=unique_points, crs="EPSG:2056")
    gdf_points.to_file(f"{output_dir}/shp/triangle_stations.shp")


def process_gtfs_stops(gtfs_stops_path):
    stops = pd.read_csv(gtfs_stops_path)
    stops = gpd.GeoDataFrame(
            stops,
            geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs="EPSG:4326"
        )
    stops_gdf = stops.to_crs(crs="EPSG:2056")

    # Filter for stations (location_type not null or 1)
    stations_gdf = stops_gdf[stops_gdf["location_type"].notna()]
    stations_gdf.loc[:, "stop_id"] = stations_gdf["stop_id"].str.split("Parent").str[-1]
    stations_gdf = stations_gdf.drop(columns=["stop_lat", "stop_lon"])

    return stations_gdf


def fix_line(line, page_number):
    if "Oberdorf SO 264" in line:
        #print("Fixing Oberdorf on page %d" % page_number)
        line = line.replace("Oberdorf SO 264", "Oberdorf SO 0264")

    for fix in MISSING_STATION_IDS:
        if fix["page"] == page_number and line.endswith(fix["name"]):
            #print("Fixing %s on page %d to id '%s'" % (fix["name"], fix["page"], fix["new_id"]))
            line += " " + fix["new_id"]

    return line


def fix_station_line(line, page_number):
    for i in range(len(line)):
        if page_number == 29 and (line[i] == "29a)" or line[i] == "44a)"):
            line[i] = line[i][:2]
        elif page_number == 20 and line[i] == "*":
            line[i] = "13"
        elif page_number == 34 and line[i] == "40a)":
            line[i] = "40"

    return line


def read_station(line, direction):
    # Some stations have spaces in the name. Here we count until we find the
    # first number. Everything before belongs to the name.
    number_count = 0

    while re.match(IS_NUMBER_PATTERN, line[number_count]):
        number_count += 1

    station_name = " ".join(line[number_count:][:-1])
    station_id = line[-1]
    distances = list(map(int, line[:number_count]))

    if station_name == "Iselle transito":
        station_id = "1300003"

    if station_name == "Pino transito":
        station_id = "1300209"

    if station_name == "Delle":
        station_id = "1402427"

    if station_name == "Grandgourt":
        station_id = "82074"

    if station_name == "Waldshut":
        station_id = "8014474"

    if station_name == "Riehen Niederholz":
        station_id = "89473"

    if station_name == "Riehen":
        station_id = "14439"

    if station_id == "18549":
        station_id = "18459"
    
    if station_id == "16355":
        station_id = "75620"

    if station_id == "09415":
        station_id = "95374"

    if station_id == "04210":
        station_id = "04318"

    if station_id == "04319":
        station_id = "96101"
    
    if station_id == "01102":
        station_id = "79293"

    if "Bözingerfeld" in station_name:
        station_name = station_name.replace("Bözingerfeld", "Bözingenfeld")

    if station_name == "Linthal *":
        station_name = "Linthal"

    if station_id == "18542":
        station_id = "18452"

    if len(station_id) < 5:
        station_id = ("0" * (5 - len(station_id))) + station_id

    return {
        "name" : station_name, "id" : station_id, "distances" : distances,
        "direction" : direction
    }