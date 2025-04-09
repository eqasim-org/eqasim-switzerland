import os
import pandas as pd
import geopandas as gpd
import time


def modify_persons_households(person, household, path):
    """
    Augments the persons dataset using households:
    - adds weight 1 to each person
    - adds household income and number of cars class to persons
    """
    person['person_weight'] = 1
    person = person.merge(household[['household_id', 'income', 'number_of_cars_class']], on='household_id', how='left')
    person = person.rename(columns={'income': 'household_income'})
    person.to_csv(path, sep=';', index=False, lineterminator='\n')

def modify_activities_dataset(persons, activity, path):
    """
    Modifies the activities dataset:
    - adds the weight 1 to each activity
    - removes all individuals below 6
    """
    activity['person_weight'] = 1
    print(list(activity.columns))
    print(list(persons.columns))
    filtered_persons = persons[persons['age'] > 6]
    activity_below_6 = activity[activity['person_id'].isin(filtered_persons['person_id'])]
    activity_below_6.to_csv(path, sep=';', index=False, lineterminator='\n')


def add_home_canton(persons, trips, activities, prefix):
    """
    Adds the home canton of the person associated with the trip/activity
    - this is added because activities/trips should be assigned based on the person's home canton
    """
    persons = persons.rename(columns={'canton_name': 'home_canton'})
    trips = trips.merge(persons[['person_id', 'home_canton']], on='person_id', how='left')
    activities = activities.merge(persons[['person_id', 'home_canton']], on='person_id', how='left')

    activities.to_csv(f'{prefix}switzerland_activities_geo.csv', sep=',', index=False, lineterminator='\n')
    trips.to_csv(f'{prefix}switzerland_trips_geo.csv', sep=',', index=False, lineterminator='\n')


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


def join_activities_with_coordinates(gpkg_path, csv_path, output_path=None):
    """
    Adds coordinate information to the activity dataset
    - the gpkg path contains the coordinates
    - the csv file to which the coordinates should be added
    """
    print("Reading coordinates from GPKG...")
    coords = gpd.read_file(
        gpkg_path,
        include_fields=['person_id', 'activity_index'],
    )
    
    coords['easting'] = coords.geometry.x
    coords['northing'] = coords.geometry.y
    coords = coords[['person_id', 'activity_index', 'easting', 'northing']]
    
    print(f"Extracted {len(coords):,} coordinate records")
    
    print("Reading activities CSV...")
    activities = pd.read_csv(
        csv_path,
        sep=';',
    )

    print(f"Loaded {len(activities):,} activity records")
    
    print("Merging datasets...")
    merged = pd.merge(
        activities,
        coords,
        on=['person_id', 'activity_index'],
        how='left'
    )
    
    print("Missing coordinates count:", merged['easting'].isna().sum())
    
    if output_path:
        merged.to_csv(output_path, index=False)
        print(f"Saved merged data to {output_path}")
    
    return merged


def add_trip_coordinates(trips_gpkg_path, trips_csv_path, output_path=None):
    """
    Adds the trip origin and destination coordinates to the dataframe
    """    
    print("Reading trip geometries from GPKG...")
    trips_gdf = gpd.read_file(
        trips_gpkg_path,
        include_fields=['person_id', 'trip_index']  # Only load necessary attributes
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
    
    if output_path:
        trips_with_coords.to_csv(output_path, index=False, sep=';')
        print(f"Saved trip data with coordinates to {output_path}")
    
    return trips_with_coords


def add_home_coordinates(homes_gpkg_path, persons_csv_path, households_csv_path, output_dir=None):
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
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        households_output = os.path.join(output_dir, "switzerland_households_coords.csv")
        households_with_coords.to_csv(households_output, index=False, sep=';')
        print(f"Saved households with coordinates to {households_output}")
        
        persons_output = os.path.join(output_dir, "switzerland_persons_coords.csv")
        persons_with_coords.to_csv(persons_output, index=False, sep=';')
        print(f"Saved persons with coordinates to {persons_output}")
    
    return households_with_coords, persons_with_coords


if __name__ == '__main__':

    # person = pd.read_csv(f'/cluster/home/chaoch/ch/switzerland_data/output/switzerland_persons_filt.csv', sep=';', header=0)
    # household = pd.read_csv(f'/cluster/home/chaoch/ch/switzerland_data/output/switzerland_households.csv', sep=';', header=0)

    prefix = '/cluster/project/cmdp/chaoch/switzerland_data/output/'
    add_coords_activities = False
    add_coords_to_trips = False
    add_coords_to_persons = False

    persons = pd.read_csv(f'{prefix}switzerland_persons_geo.csv', sep=';', header=0)
    # households = pd.read_csv(f'{prefix}switzerland_households_geo.csv', sep=',', header=0)
    activities = pd.read_csv(f'{prefix}switzerland_activities_geo.csv', sep=';', header=0)
    trips = pd.read_csv(f'{prefix}switzerland_trips_geo.csv', sep=',', header=0)
    print(persons.columns)
    print(activities.columns)
    print(trips.columns)
    add_home_canton(persons, trips, activities, prefix)

    # Adds coordinates of activities
    if add_coords_activities:
        merged_data = join_activities_with_coordinates(
            gpkg_path=f"{prefix}switzerland_activities.gpkg",
            csv_path=f"{prefix}switzerland_activities.csv",
            output_path=f"{prefix}activities_with_coordinates.csv"
        )

    # Adds coordinates of trips (origin & destination)
    if add_coords_to_trips:
        trips_with_coords = add_trip_coordinates(
            trips_gpkg_path=f"{prefix}switzerland_trips.gpkg",
            trips_csv_path=f"{prefix}switzerland_trips.csv",
            output_path=f"{prefix}trips_with_coordinates.csv"
        )

    # Adds coordinates to persons and households based on home coordinates
    if add_coords_to_persons:
        households, persons = add_home_coordinates(
            homes_gpkg_path=f"{prefix}switzerland_homes.gpkg",
            persons_csv_path=f"{prefix}switzerland_persons.csv",
            households_csv_path=f"{prefix}switzerland_households.csv",
            output_dir=f"{prefix}"
        )
   