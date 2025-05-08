import pandas as pd
import geopandas as gpd
from .add_cantons import add_canton_name

def configure(context):
    context.config("output_path")

    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("synthesis.output")

def inspect_gpkg_columns(file_path):
    """
    Exploration of gpkg files
    - Prints out columns and example values of the columns
    """
    # Load the GeoPackage file
    gdf = gpd.read_file(file_path)
    
    print("=== FILE OVERVIEW ===")
    print(f"File: {file_path}")
    print(f"Number of records: {len(gdf):,}")
    print(f"Geometry type: {gdf.geom_type.unique()[0]}")
    print(f"CRS: {gdf.crs}\n")
    
    print("=== COLUMN DETAILS ===")
    for column in gdf.columns:
        dtype = gdf[column].dtype
        unique_count = gdf[column].nunique()
        
        print(f"\nColumn: {column}")
        print(f"Type: {dtype}")
        print(f"Unique values: {unique_count}")
        
        if unique_count <= 10:
            print("Unique values:", gdf[column].unique())
        elif pd.api.types.is_numeric_dtype(dtype):
            print("Stats:", gdf[column].describe()[['min', 'max', 'mean']].to_dict())
        else:
            sample_values = gdf[column].dropna().sample(min(5, len(gdf)), random_state=1).unique()
            print("Sample values:", sample_values)

        if column == 'start_time' or column == 'end_time':
            print("Time range:", gdf[column].min(), "to", gdf[column].max())


def add_activities_coordinates(gpkg_path, csv_path, add_coordinates=False):
    """
    Optionally adds coordinate information to the activity dataset
    - the gpkg path contains the coordinates
    - the csv file to which the coordinates should be added
    """
    print("Reading activities CSV...")
    activities = pd.read_csv(
        csv_path,
        sep=';',
    )
    print(f"Loaded {len(activities):,} activity records")
    
    if not add_coordinates:
        return activities

    print("Reading coordinates from GPKG...")
    coords = gpd.read_file(
        gpkg_path,
        include_fields=['person_id', 'activity_index'],
    )
    
    coords['easting'] = coords.geometry.x
    coords['northing'] = coords.geometry.y
    coords = coords[['person_id', 'activity_index', 'easting', 'northing']]    
    print(f"Extracted {len(coords):,} coordinate records")

    print("Merging datasets...")
    merged = pd.merge(
        activities,
        coords,
        on=['person_id', 'activity_index'],
        how='left'
    )
    
    print("Missing coordinates count:", merged['easting'].isna().sum())
    
    return merged


def add_trip_coordinates(trips_gpkg_path, trips_csv_path, add_coordinates=False):
    """
    Optionally adds the trip origin and destination coordinates to the dataframe
    """    
    print("Reading trips CSV...")
    trips = pd.read_csv(
        trips_csv_path,
        sep=';',
        dtype={
            'person_id': 'int32',
            'trip_index': 'int8',
            'preceding_activity_index': 'int8',
            'following_activity_index': 'int8'
        }
    )

    # Coordinates are not always needed
    if not add_coordinates:
        return trips
    
    print("Reading trip geometries from GPKG...")
    trips_gdf = gpd.read_file(
        trips_gpkg_path,
        include_fields=['person_id', 'trip_index']
    )
    
    print("Extracting origin/destination coordinates...")
    trips_gdf['origin_easting'] = trips_gdf.geometry.apply(lambda g: g.coords[0][0])
    trips_gdf['origin_northing'] = trips_gdf.geometry.apply(lambda g: g.coords[0][1])
    trips_gdf['dest_easting'] = trips_gdf.geometry.apply(lambda g: g.coords[-1][0])
    trips_gdf['dest_northing'] = trips_gdf.geometry.apply(lambda g: g.coords[-1][1])
    
    trip_coords = trips_gdf[[
        'person_id', 'trip_index',
        'origin_easting', 'origin_northing',
        'dest_easting', 'dest_northing'
    ]]
    
    print("Merging trip data with coordinates...")
    trips_with_coords = pd.merge(
        trips,
        trip_coords,
        on=['person_id', 'trip_index'],
        how='left'
    )
    
    print("Missing coordinates count:")
    print("- Origin:", trips_with_coords['origin_easting'].isna().sum())
    print("- Destination:", trips_with_coords['dest_easting'].isna().sum())
    
    return trips_with_coords


def add_home_coordinates(homes_gpkg_path, persons_csv_path, households_csv_path):
    """
    Adds home coordinates to the persons and household dataset
    """
    print("Loading home locations...")
    homes = gpd.read_file(homes_gpkg_path)
    homes['home_easting'] = homes.geometry.x
    homes['home_northing'] = homes.geometry.y
    home_coords = homes[['household_id', 'home_easting', 'home_northing']]
    
    print("Processing households data...")
    households = pd.read_csv(households_csv_path, sep=';')
    households_with_coords = pd.merge(
        households,
        home_coords,
        on='household_id',
        how='left'
    )
    
    print("Processing persons data...")
    persons = pd.read_csv(persons_csv_path, sep=';')
    persons_with_coords = pd.merge(
        persons,
        home_coords,
        on='household_id',
        how='left'
    )
    
    print("=== VALIDATION ===")
    print(f"Households with coordinates: {households_with_coords['home_easting'].notna().sum()}/{len(households_with_coords)}")
    print(f"Persons with coordinates: {persons_with_coords['home_easting'].notna().sum()}/{len(persons_with_coords)}")
    print('columns of households', list(households_with_coords.columns))
    return households_with_coords, persons_with_coords


def modify_persons_households(person, household):
    """
    Augments the persons dataset using households:
    - adds weight 1 to each person
    - adds household income and number of cars class to persons
    """
    person['person_weight'] = 1
    person = person.merge(household[['household_id', 'income', 'number_of_cars_class']], on='household_id', how='left')
    person = person.rename(columns={'income': 'household_income'})
    return person


def filter_activities_dataset(persons, activity):
    """
    Modifies the activities dataset:
    - adds the weight 1 to each activity
    - removes all individuals below 6
    """
    activity['person_weight'] = 1
    filtered_persons = persons[persons['age'] > 6]
    activity_below_6 = activity[activity['person_id'].isin(filtered_persons['person_id'])]
    return activity_below_6


def add_home_canton(persons, trips, activities):
    """
    Adds the home canton of the person associated with the trip/activity
    - this is added because activities/trips should be assigned based on the person's home canton
    """
    trips = trips.merge(persons[['person_id', 'canton_name']], on='person_id', how='left')
    activities = activities.merge(persons[['person_id', 'canton_name']], on='person_id', how='left')

    return activities, trips

def preprocess_synthetic_data(directory, save_directory=None):
    """
    Preprocesses the synthetic data for analysis. 

    - directory: The directory where the synthetic data is stored
    - save_directory: The directory where the processed data is stored (optional). Can be passed to next stage directly
    """
    # Adds coordinates of activities
    activities = add_activities_coordinates(
        gpkg_path=f"{directory}/switzerland_activities.gpkg",
        csv_path=f"{directory}/switzerland_activities.csv",
    )

    # Adds coordinates of trips (origin & destination)
    trips = add_trip_coordinates(
        trips_gpkg_path=f"{directory}/switzerland_trips.gpkg",
        trips_csv_path=f"{directory}/switzerland_trips.csv",
    )

    # Adds coordinates to persons and households based on home coordinates
    households, persons = add_home_coordinates(
        homes_gpkg_path=f"{directory}/switzerland_homes.gpkg",
        persons_csv_path=f"{directory}/switzerland_persons.csv",
        households_csv_path=f"{directory}/switzerland_households.csv",
    )

    # Add cantons to each data point based on their household canton
    persons = add_canton_name(persons, x_col='home_easting', y_col='home_northing')
    households = add_canton_name(households, x_col='home_easting', y_col='home_northing')
    persons = modify_persons_households(persons, households)
    activities = filter_activities_dataset(persons, activities)
    activities, trips = add_home_canton(persons, trips, activities)
    
    if save_directory is not None:
        # Write all data
        persons.to_csv(f'{save_directory}/switzerland_persons_geo.csv', index=False, sep=',')
        households.to_csv(f'{save_directory}/switzerland_households_geo.csv', index=False, sep=',')
        trips.to_csv(f'{save_directory}/switzerland_trips_geo.csv', index=False, sep=',')
        activities.to_csv(f'{save_directory}/switzerland_activities_geo.csv', index=False, sep=',')

    # Return data for the next stage
    return persons, households, trips, activities

def execute(context):
    directory = context.config("output_path")

    persons, households, trips, activities = preprocess_synthetic_data(directory=directory)
    return persons, households, trips, activities

if __name__ == '__main__':

    # directory = input("Enter directory name where the synthetic data lies:")
    # save_directory = input("Enter directory where the processed data should be stored:")

    directory = '/cluster/project/cmdp/chaoch/switzerland_data/output'
    save_directory = '/cluster/project/cmdp/chaoch/switzerland_data/output_test'

    persons, households, trips, activities = preprocess_synthetic_data(directory=directory, save_directory=save_directory)
