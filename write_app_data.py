import dash
import pandas as pd
from app_utils import *
import json 

# Initialize the Dash app
app = dash.Dash(__name__)

activities = pd.read_csv('microcensus_data/microcensus_act_geo.csv', sep=';', header=0)
trips = pd.read_csv('microcensus_data/microcensus_trips_geo.csv', sep=',', header=0)
households = pd.read_csv('microcensus_data/microcensus_households_geo.csv', sep=',', header=0)
persons = pd.read_csv('microcensus_data/microcensus_persons_geo.csv', sep=',', header=0)
persons['home_canton'] = persons['canton_name']
households['home_canton'] = households['canton_name']

output_activities = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_activities_geo.csv', sep=',', header=0)
output_trips = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_trips_geo.csv', sep=',', header=0)
output_households = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_households_geo.csv', sep=',', header=0)
output_persons = pd.read_csv('/cluster/project/cmdp/chaoch/switzerland_data/output/switzerland_persons_geo.csv', sep=';', header=0)
output_households['household_weight'] = 1
output_trips['person_weight'] = 1
output_persons['home_canton'] = output_persons['canton_name']
output_households['home_canton'] = output_households['canton_name']

cantons = [
    'Zürich', 'Basel-Stadt', 'St. Gallen', 'Bern', 'Fribourg', 'Vaud', 
    'Ticino', 'Aargau', 'Genève', 'Solothurn', 'Jura', 'Valais', 
    'Luzern', 'Basel-Landschaft', 'Neuchâtel', 'Thurgau', 'Uri', 
    'Schwyz', 'Nidwalden', 'Glarus', 'Graubünden', 'Schaffhausen', 
    'Zug', 'Obwalden', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden'
]

def write_non_category_data(micro, synthetic, func):
    """
    Writes data to a specific JSON format. 

    :func: The function to compute the data to write
    """
    write_data = dict()
    for canton in cantons:
        write_data[canton] = dict()

        micro_split = micro.query(f"home_canton == '{canton}'")
        synthetic_split = synthetic.query(f"home_canton == '{canton}'")

        act_micro, freq_micro = func(micro_split)
        act_out, freq_out = func(synthetic_split)

        # insert synthetic & microcensus data
        write_data[canton]['synthetic'] = dict()
        write_data[canton]['microcensus'] = dict()
        for act, freq in zip(act_micro, freq_micro):
            write_data[canton]['microcensus'][act] = freq
        for act, freq in zip(act_out, freq_out):
            write_data[canton]['synthetic'][act] = freq
    return write_data


def write_category_data(micro, synthetic, category_name, category_options, func, feature, bins=False):
    """
    Writes data that will be filtered by a specific category.

    :category_name: the category we want to filter by (name to marginalize the data by) (e.g. purpose)
    :category_options: the different values the category contains (e.g. home, shopping, education)
    :func: function to apply per category
    :feature: feature of interest in the analysis (e.g. duration)
    """
    write_data = dict()
    for canton in cantons:
        write_data[canton] = dict()
        
        # select by canton
        micro_split = micro.query(f"home_canton == '{canton}'")
        synthetic_split = synthetic.query(f"home_canton == '{canton}'")

        # insert synthetic & microcensus data
        write_data[canton]['synthetic'] = dict()
        write_data[canton]['microcensus'] = dict()

        for cat in category_options:
            write_data[canton]['microcensus'][cat] = dict()
            write_data[canton]['synthetic'][cat] = dict()

            # filter by category
            if cat == 'All': 
                micro_filtered = micro_split
                synthetic_filtered = synthetic_split
            else:
                micro_filtered = micro.loc[micro[category_name] == cat]
                synthetic_filtered = synthetic.loc[synthetic[category_name] == cat]

            bin_edges = None
            if bins:
                min_dist = np.percentile(micro_filtered[feature], 5)
                max_dist = np.percentile(micro_filtered[feature], 95)
                bin_edges = np.linspace(min_dist, max_dist, num=40)  

            bins_micro, hist_micro = func(micro_filtered, feature=feature, bins=bin_edges)
            bins_syn, hist_syn = func(synthetic_filtered, feature=feature, bins=bin_edges)

            for val, freq in zip(bins_micro, hist_micro):
                write_data[canton]['microcensus'][cat][val] = freq
            for val, freq in zip(bins_syn, hist_syn):
                write_data[canton]['synthetic'][cat][val] = freq
    return write_data


def write_num_activities(activities, output_activities):
    data = write_non_category_data(activities, output_activities, get_weighted_num_activities)

    with open("plot_data/num_activities.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_frequent_sequences(activities, output_activities):
    data = write_non_category_data(activities, output_activities, frequent_weighted_sequences)
    
    with open("plot_data/frequent_sequences.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_out_of_home(activities, output_activities):
    data = write_non_category_data(activities, output_activities, frequent_out_of_home_activities)
   
    with open("plot_data/out_of_home.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_available_cars_general(households, output_households):
    data = write_non_category_data(households, output_households, get_car_availability)
    with open("plot_data/car_availability.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_pt_subscriptions_general(persons, output_persons):
    data = write_non_category_data(persons, output_persons, get_subscription_proportions)
    with open("plot_data/pt_subscriptions.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

def write_trip_crowfly_distance(trips, output_trips):
    options = ['All', 'home', 'work', 'leisure', 'shop', 'other', 'education']
    output_trips['person_weight'] = 1

    data = write_category_data(trips, output_trips, 
                        category_name='following_purpose',
                        category_options=options, 
                        func=get_histogram,
                        feature='crowfly_distance',
                        bins=True)
    with open("plot_data/trip_distance.json", "w") as json_file:
        json.dump(data, json_file, indent=4) 

def write_activity_durations(activities, output_activities):
    for df in [activities, output_activities]:
        df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
        df["duration"] = df["end_time"] - df["start_time"]
    
    activities = activities.dropna(subset=["duration"])
    output_activities = output_activities.dropna(subset=["duration"])

    activity_types = sorted(set(activities["purpose"].unique()).union(output_activities["purpose"].unique()))
    activity_types.append('All')

    data = write_category_data(activities, output_activities, 
                        category_name='purpose',
                        category_options=activity_types,
                        func=get_histogram_time,
                        feature='duration',)
    
    with open("plot_data/activity_durations.json", "w") as json_file:
        json.dump(data, json_file, indent=4)  

def write_departure_times(trips, output_trips):
    output_trips['weight'] = 1
    purpose_types = sorted(trips['following_purpose'].unique())
    purpose_types.append('All')
    
    data = write_category_data(trips, output_trips, 
                        category_name='following_purpose',
                        category_options=purpose_types, 
                        func=get_histogram_time,
                        feature='departure_time')
    with open("plot_data/departure_times.json", "w") as json_file:
        json.dump(data, json_file, indent=4) 


if __name__ == '__main__':

    # write_frequent_sequences(activities, output_activities)
    # write_out_of_home(activities, output_activities)
    print(list(persons.columns))
    write_pt_subscriptions_general(persons, output_persons)
    write_available_cars_general(households, output_households)
