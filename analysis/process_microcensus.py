import pickle
import os 
import numpy as np
import pandas as pd
from .add_cantons import add_canton_name

def configure(context):
    context.config("output_path")
    context.config("working_directory")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("synthesis.output")

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

def convert_households(data):
    data = data.rename(columns={'income_class': 'income'})
    data = add_canton_name(data, x_col='home_x', y_col='home_y')
    return data

def preprocess_microcensus_data(directory, save_directory=None, prefix='data.microcensus'):
    print("Reading the .pkl files...")
    files = get_pkl_data(directory, prefix)
    
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
    households = convert_households(households)
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

def execute(context):
    directory = context.config('working_directory')

    persons, households, trips, activities = preprocess_microcensus_data(directory=directory)
    return persons, households, trips, activities

if __name__ == '__main__':
    # Get the directories for reading and writing
    # directory = input("Enter directory name where the microcensus data lies:")
    # save_directory = input("Enter directory where the processed data should be stored:")

    directory = '/cluster/project/cmdp/chaoch/switzerland_data/cache/'
    save_directory = '/cluster/project/cmdp/chaoch/microcensus_data_test'
    prefix = 'data.microcensus'

    preprocess_microcensus_data(directory, save_directory, prefix=prefix)
