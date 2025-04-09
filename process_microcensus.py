import pickle
import os 
import numpy as np
import pandas as pd

def process_filename(file_path):
    temp = file_path.split(".")[2:] # skip synthesis.population
    temp[-2] = temp[-2].split("__")[0] # skip the hash the end
    result = "_".join(temp[:-1]) # skip the .p ending
    return result

def get_pkl_data(directory, prefix):
    files = dict()
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.p'):
            with open(directory + filename, 'rb') as file:
                data = pickle.load(file)
                processed_name = process_filename(filename)
                files[processed_name] = data
    return files

def convert_trips(data):
    data = data.rename(columns={'purpose': 'following_purpose', 'trip_id': 'trip_index'})
    data['preceding_activity_index'] = None
    data['preceding_purpose'] = data['following_purpose'].shift(1)
    data.loc[data['trip_index'] == 1, 'preceding_purpose'] = 'home'

    data.to_csv('microcensus_data/microcensus_trips.csv', sep=';', index=False, lineterminator='\n')

def convert_persons(data):
    data = data.rename(columns={'driving_license': 'has_driving_license'})    
    data.to_csv('microcensus_data/microcensus_persons.csv', sep=';', index=False, lineterminator='\n')

def convert_households(data):
    data = data.rename(columns={'income_class': 'income'})
    data.to_csv('microcensus_data/microcensus_households.csv', sep=';', index=False, lineterminator='\n')


def create_activities(trips, persons):
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

    activities_df.to_csv('microcensus_data/microcensus_activities.csv', sep=';', index=False, lineterminator='\n')

def augment_activities(activities, persons):
    # add the weights of individuals
    weights = persons[['person_id', 'person_weight']]
    activities_df = activities.merge(weights, left_on='person_id', right_on='person_id', how='left')

    activities_df = activities_df.dropna(subset=['person_weight'])

    activities_df.to_csv('microcensus_data/microcensus_act_weighted.csv', sep=';', index=False, lineterminator='\n')

def augment_trips(trips, persons):
    # add the weights of individuals
    weights = persons[['person_id', 'person_weight']]
    trips_df = trips.merge(weights, left_on='person_id', right_on='person_id', how='left')

    trips_df = trips_df.dropna(subset=['person_weight'])

    trips_df.to_csv('microcensus_data/microcensus_trips_weighted.csv', sep=';', index=False, lineterminator='\n')

def add_household_income(persons, households):
    persons = persons.merge(households[['person_id', 'income']], on='person_id', how='left')
    persons = persons.rename(columns={'income': 'household_income'})
    persons.to_csv('microcensus_data/microcensus_persons.csv', sep=';', index=False, lineterminator='\n')

def add_origin_canton(persons, trips, activities):
    persons = persons.rename(columns={'canton_name': 'home_canton'})
    trips = trips.merge(persons[['person_id', 'home_canton']], on='person_id', how='left')
    activities = activities.merge(persons[['person_id', 'home_canton']], on='person_id', how='left')
    trips.to_csv('microcensus_data/microcensus_trips_geo.csv', sep=',', index=False, lineterminator='\n')
    activities.to_csv('microcensus_data/microcensus_act_geo.csv', sep=';', index=False, lineterminator='\n')

if __name__ == '__main__':
    # directory = '/cluster/home/chaoch/ch/switzerland_data/cache/'
    # prefix = 'data.microcensus'
    # files = get_pkl_data(directory, prefix)
    # for name, file in files.items():
    #     print(name, type(file), len(file))
    
    trips = pd.read_csv('microcensus_data/microcensus_trips_geo.csv', sep=',', header=0)
    persons = pd.read_csv('microcensus_data/microcensus_persons_geo.csv', sep=',', header=0)
    activities = pd.read_csv('microcensus_data/microcensus_act_geo.csv', sep=';', header=0)
    # households = pd.read_csv('microcensus_data/microcensus_households.csv', sep=';', header=0)

    add_origin_canton(persons, trips, activities)