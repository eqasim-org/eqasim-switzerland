import pickle
import os 
import numpy as np
import pandas as pd
import geopandas as gpd
from analysis.webmap_export import assign_cantons, clean_geo_name

def process_filename(file_path):
    temp = file_path.split(".")[2:] # skip synthesis.population
    temp[-2] = temp[-2].split("__")[0] # skip the hash the end
    result = "_".join(temp[:-1]) # skip the .p ending
    return result

def get_pkl_data(directory, prefix):
    """
    Reads in the .pkl files and simplifies their names
    """
    files = dict()
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.p'):
            with open(directory + '/' + filename, 'rb') as file:
                data = pickle.load(file)
                print("filename", filename)
                processed_name = process_filename(filename)
                files[processed_name] = data
    return files

def create_activities(trips, persons):
    """
    Creates the activities dataset given the trips and the person
    - Assumes the first activity is home for all individuals
    - The first activity is the following purpose of the first trip
    - Individuals who never had a trip will have one home activity
    - Each activity is weighed by the person's weight
    """
    activities = []

    # Group by person_id to handle each person's trips individually
    for person_id, group in trips.groupby('person_id'):
        group = group.sort_values(by='trip_index')
        
        # add the first activity as "home"
        activities.append({
            'person_id': person_id,
            'activity_index': 1,
            'preceding_trip_index': -1,
            'following_trip_index': 1,
            'purpose': 'home',
            'start_time': np.nan,
            'end_time': group['departure_time'].iloc[0],
            'is_last': False,
            'canton_name': group['canton_name'].iloc[0] 
        })

        for i, row in group.iterrows():
            activity_index = row['trip_index'] + 1
            purpose = row['following_purpose']
            preceding_trip_index = row['trip_index']
            following_trip_index = row['trip_index'] + 1
            start_time = group.loc[i, 'arrival_time']
            canton = group.loc[i, 'canton_name']
            end_time = group.loc[i + 1, 'departure_time'] if i + 1 in group.index else np.nan
            is_last = i == group.index[-1]
            
            activities.append({
                'person_id': person_id,
                'activity_index': activity_index,
                'preceding_trip_index': preceding_trip_index,
                'following_trip_index': following_trip_index,
                'purpose': purpose,
                'start_time': start_time,
                'end_time': end_time,
                'canton_name': canton,
                'is_last': is_last
            })

    # for individuals who never made a trip, add one activity called "home"
    one_activity_persons = persons[~persons['person_id'].isin(trips['person_id'].drop_duplicates())]
    filtered_persons = one_activity_persons[one_activity_persons['age'] > 6]
    person_ids = zip(filtered_persons['person_id'].tolist(), filtered_persons['canton_name'].tolist())

    for id, canton in person_ids:
        activities.append({
            'person_id': id,
            'activity_index': 1,
            'preceding_trip_index': -1,
            'following_trip_index': 6,
            'purpose': 'home',
            'start_time': np.nan,
            'end_time': np.nan,
            'canton_name': canton,
            'is_last': False
        })

    activities_df = pd.DataFrame(activities)

    # Add the person weight
    weights = persons[['person_id', 'person_weight']]
    activities_df = activities_df.merge(weights, left_on='person_id', right_on='person_id', how='left')
    activities_df = activities_df.dropna(subset=['person_weight'])

    return activities_df

def convert_persons(persons, households):
    """
    Adds household income and renames some column to match synthetic data
    """
    persons = persons.merge(households[['person_id', 'income', 'canton_name']], on='person_id', how='left')
    persons = persons.rename(columns={
        'driving_license': 'has_driving_license',
        'income': 'household_income',
    })
    return persons

def convert_trips(trips, persons):
    """
    Renames some columns. Adds the preceding purpose of a trip, the weight, and
    the home canton of person performing the trip.
    """
    # Update the purpose of trips
    trips = trips.rename(columns={'purpose': 'following_purpose', 'trip_id': 'trip_index'})
    trips['preceding_activity_index'] = None
    trips['preceding_purpose'] = trips['following_purpose'].shift(1)
    trips.loc[trips['trip_index'] == 1, 'preceding_purpose'] = 'home'

    # Add person weight and home canton to trips
    weights = persons[['person_id', 'person_weight', 'canton_name']]
    trips = trips.merge(weights, left_on='person_id', right_on='person_id', how='left')
    trips = trips.dropna(subset=['person_weight'])

    return trips

def convert_households(data, canton_boundaries):
    data = data.rename(columns={'income_class': 'income'})
    
    # Use assign_cantons with the same canton_boundaries
    data = assign_cantons(data, canton_boundaries, x_col='home_x', y_col='home_y')
    data['canton_name'] = data['canton_name'].transform(clean_geo_name)
    return data

def preprocess_microcensus_data(directory, canton_boundaries, save_directory=None, prefix=''):
    print("Reading the .pkl files...")
    files = get_pkl_data(directory, prefix='data.microcensus')
    
    households = None
    trips = None
    persons = None
    activities = None

    for name, file in files.items():
        print(name, type(file), len(file))
        if name == 'households':
            households = file
        elif name == 'persons':
            persons = file
        elif name == 'trips':
            trips = file[0]

    print("Preprocessing the data files...")
    households = convert_households(households, canton_boundaries)
    persons = convert_persons(persons, households)
    trips = convert_trips(trips, persons)
    activities = create_activities(trips, persons)

    if save_directory is not None: 
        print("Writing the data files...")
        households.to_csv(f'{save_directory}/microcensus_households.csv', sep=',', index=False, lineterminator='\n')
        persons.to_csv(f'{save_directory}/microcensus_persons.csv', sep=',', index=False, lineterminator='\n')
        trips.to_csv(f'{save_directory}/microcensus_trips.csv', sep=',', index=False, lineterminator='\n')
        activities.to_csv(f'{save_directory}/microcensus_activities.csv', sep=',', index=False, lineterminator='\n')

    return persons, households, trips, activities

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
    trips = pd.read_csv(trips_csv_path, sep=';')

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
    activities = activities.merge(persons[['person_id', 'canton_name']], on='person_id', how='left')
    trips = trips.merge(persons[['person_id', 'canton_name']], on='person_id', how='left')
    return activities, trips

def preprocess_synthetic_data(directory, canton_boundaries, save_directory=None):
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
    persons = assign_cantons(persons, canton_boundaries, x_col='home_easting', y_col='home_northing')
    households = assign_cantons(households, canton_boundaries, x_col='home_easting', y_col='home_northing')
    persons['canton_name'] = persons['canton_name'].transform(clean_geo_name)
    households['canton_name'] = households['canton_name'].transform(clean_geo_name)
    
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
