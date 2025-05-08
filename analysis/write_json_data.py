import pandas as pd
from .app_utils import *
import json 
import numbers
import pickle
import os

# TODO these characters do not render well on the app
cantons = [
    'Zürich', 'Basel-Stadt', 'St. Gallen', 'Bern', 'Fribourg', 'Vaud', 
    'Ticino', 'Aargau', 'Genève', 'Solothurn', 'Jura', 'Valais', 
    'Luzern', 'Basel-Landschaft', 'Neuchâtel', 'Thurgau', 'Uri', 
    'Schwyz', 'Nidwalden', 'Glarus', 'Graubünden', 'Schaffhausen', 
    'Zug', 'Obwalden', 'Appenzell Ausserrhoden', 'Appenzell Innerrhoden', 'All'
]

def write_non_category_data(micro, synthetic, func):
    """
    Writes data to a specific JSON format. 

    - func: The function to compute the data to write
    """
    print("Synthetic columns:", synthetic.columns)
    print("Micro columns:", micro.columns)
    write_data = dict()
    for canton in cantons:
        write_data[canton] = dict()

        if canton == 'All':
            micro_split = micro
            synthetic_split = synthetic
        else:
            micro_split = micro.query(f"canton_name == '{canton}'")
            synthetic_split = synthetic.query(f"canton_name == '{canton}'")

        act_micro, freq_micro = func(micro_split)
        act_out, freq_out = func(synthetic_split)

        # insert synthetic & microcensus data
        write_data[canton]['Synthetic'] = dict()
        write_data[canton]['Microcensus'] = dict()
        for act, freq in zip(act_micro, freq_micro):
            write_data[canton]['Microcensus'][act] = freq
        for act, freq in zip(act_out, freq_out):
            write_data[canton]['Synthetic'][act] = freq
    return write_data


def write_category_data(micro, synthetic, category_name, category_options, func, feature, bins=False):
    """
    Writes data that will be filtered by a specific category.

    - category_name: the category we want to filter by (name to marginalize the data by) (e.g. purpose)
    - category_options: the different values the category contains (e.g. home, shopping, education)
    - func: function to apply per category (to obtain the statistics)
    - feature: feature of interest in the analysis (e.g. duration)
    """
    write_data = dict()
    for canton in cantons:
        write_data[canton] = dict()
        
        # select by canton
        if canton == 'All':
            micro_split = micro
            synthetic_split = synthetic
        else:
            micro_split = micro.query(f"canton_name == '{canton}'")
            synthetic_split = synthetic.query(f"canton_name == '{canton}'")

        # insert synthetic & microcensus data
        write_data[canton]['Synthetic'] = dict()
        write_data[canton]['Microcensus'] = dict()

        for cat in category_options:

            if isinstance(cat, numbers.Number):
                cat = int(cat)
            
            write_data[canton]['Microcensus'][cat] = dict()
            write_data[canton]['Synthetic'][cat] = dict()

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

            # Apply the function that computes the statistics
            bins_micro, hist_micro = func(micro_filtered, feature=feature, bins=bin_edges)
            bins_syn, hist_syn = func(synthetic_filtered, feature=feature, bins=bin_edges)

            for val, freq in zip(bins_micro, hist_micro):
                write_data[canton]['Microcensus'][cat][val] = round(freq, 6)
            for val, freq in zip(bins_syn, hist_syn):
                write_data[canton]['Synthetic'][cat][val] = round(freq, 6)
    return write_data


def write_num_activities(activities, output_activities, save_directory):
    data = write_non_category_data(activities, output_activities, get_weighted_num_activities)

    with open(f"{save_directory}/num_activities.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_frequent_sequences(activities, output_activities, save_directory):
    data = write_non_category_data(activities, output_activities, frequent_weighted_sequences)
    
    with open(f"{save_directory}/frequent_sequences.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_out_of_home(activities, output_activities, save_directory):
    data = write_non_category_data(activities, output_activities, frequent_out_of_home_activities)
   
    with open(f"{save_directory}/out_of_home.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_available_cars_general(households, output_households, save_directory):
    data = write_non_category_data(households, output_households, get_car_availability)
    with open(f"{save_directory}/car_availability.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_pt_subscriptions_general(persons, output_persons, save_directory):
    data = write_non_category_data(persons, output_persons, get_subscription_proportions)
    with open(f"{save_directory}/pt_subscriptions.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_trip_crowfly_distance(trips, output_trips, save_directory):
    options = ['All', 'home', 'work', 'leisure', 'shop', 'other', 'education']
    output_trips['person_weight'] = 1

    data = write_category_data(trips, output_trips, 
                        category_name='following_purpose',
                        category_options=options, 
                        func=get_histogram,
                        feature='crowfly_distance',
                        bins=True)
    
    with open(f"{save_directory}/trip_distance.json", "w") as json_file:
        json.dump(data, json_file, indent=4) 

def write_activity_durations(activities, output_activities, save_directory):
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
    
    with open(f"{save_directory}/activity_durations.json", "w") as json_file:
        json.dump(data, json_file, indent=4)  


def write_departure_times(trips, output_trips, save_directory):
    output_trips['weight'] = 1
    purpose_types = sorted(trips['following_purpose'].unique())
    purpose_types.append('All')
    
    data = write_category_data(trips, output_trips, 
                        category_name='following_purpose',
                        category_options=purpose_types, 
                        func=get_histogram_time,
                        feature='departure_time')
    
    with open(f"{save_directory}/departure_times.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_num_cars_income(persons, output_persons, save_directory):
    incomes = sorted(persons[persons['household_income'] >= 0]['household_income'].unique())

    data = write_category_data(persons, output_persons, 
                               category_name='household_income', 
                               category_options=incomes, 
                               func=get_individual_car_class,
                               feature=None)
    
    with open(f"{save_directory}/num_cars_income.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_num_cars_gender(persons, output_persons, save_directory):
    genders = [0, 1]

    data = write_category_data(persons, output_persons, 
                               category_name='sex', 
                               category_options=genders, 
                               func=get_individual_car_class,
                               feature=None)
    
    with open(f"{save_directory}/num_cars_gender.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_num_cars_age(persons, output_persons, save_directory):
    ages = [6, 15, 18, 24, 30, 45, 65, 80]
    labels = ['[6, 15)', '[15, 18)', '[18, 24)', '[24, 30)', '[30, 45)', '[45, 65)', '[65, 80)']
    persons['age_group'] = pd.cut(persons['age'], bins=ages, labels=labels, right=False)
    output_persons['age_group'] = pd.cut(output_persons['age'], bins=ages, labels=labels, right=False)
    
    data = write_category_data(persons, output_persons, 
                               category_name='age_group', 
                               category_options=labels, 
                               func=get_individual_car_class,
                               feature=None)
    
    with open(f"{save_directory}/num_cars_age.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_pt_sub_income(persons, output_persons, save_directory):
    # NOTE using household weight doesn't solve the issue
    incomes = sorted(persons[persons['household_income'] >= 0]['household_income'].unique())

    data = write_category_data(persons, output_persons, 
                                category_name='household_income', 
                                category_options=incomes, 
                                func=get_subscription_proportions,
                                feature=None)

    with open(f"{save_directory}/pt_sub_income.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_pt_sub_gender(persons, output_persons, save_directory):
    genders = [0, 1]
    
    data = write_category_data(persons, output_persons, 
                               category_name='sex', 
                               category_options=genders, 
                               func=get_subscription_proportions,
                               feature=None)
    with open(f"{save_directory}/pt_sub_gender.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_pt_sub_age(persons, output_persons, save_directory):
    ages = [6, 15, 18, 24, 30, 45, 65, 80]
    labels = ['[6, 15)', '[15, 18)', '[18, 24)', '[24, 30)', '[30, 45)', '[45, 65)', '[65, 80)']
    persons['age_group'] = pd.cut(persons['age'], bins=ages, labels=labels, right=False)
    output_persons['age_group'] = pd.cut(output_persons['age'], bins=ages, labels=labels, right=False)
    
    data = write_category_data(persons, output_persons, 
                               category_name='age_group', 
                               category_options=labels, 
                               func=get_subscription_proportions,
                               feature=None)
    
    with open(f"{save_directory}/pt_sub_age.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def write_all_application_data(microcensus, synthetic, save_directory):
    """
    Writes all the data for the application, 
    - microcensus_prefix: the full name of where the microcensus_data folder lies
    - synthetic_prefix: the full name of where the switzterland_data folder lies
    - save_directory: where the application data should be stored
    """    
    activities = microcensus['activities']
    trips = microcensus['trips']
    households = microcensus['households']
    persons = microcensus['persons']

    output_activities = synthetic['activities']
    output_trips = synthetic['trips']
    output_households = synthetic['households']
    output_persons = synthetic['persons']
    output_households['household_weight'] = 1
    output_trips['person_weight'] = 1

    # Write general distribution
    write_num_activities(activities, output_activities, save_directory)
    write_frequent_sequences(activities, output_activities, save_directory)
    write_out_of_home(activities, output_activities, save_directory)
    write_available_cars_general(households, output_households, save_directory)
    write_pt_subscriptions_general(persons, output_persons, save_directory)
    
    # Write marginal distribution
    write_trip_crowfly_distance(trips, output_trips, save_directory)
    write_activity_durations(activities, output_activities, save_directory)
    write_pt_sub_age(persons, output_persons, save_directory)
    write_pt_sub_income(persons, output_persons, save_directory)
    write_pt_sub_gender(persons, output_persons, save_directory)
    write_num_cars_age(persons, output_persons, save_directory)
    write_num_cars_gender(persons, output_persons, save_directory)
    write_num_cars_income(persons, output_persons, save_directory)


def load_dataframes(microcensus_prefix, synthetic_prefix):
    activities = pd.read_csv(f'{microcensus_prefix}/microcensus_activities.csv', sep=',', header=0)
    trips = pd.read_csv(f'{microcensus_prefix}/microcensus_trips.csv', sep=',', header=0)
    households = pd.read_csv(f'{microcensus_prefix}/microcensus_households.csv', sep=',', header=0)
    persons = pd.read_csv(f'{microcensus_prefix}/microcensus_persons.csv', sep=',', header=0)

    output_activities = pd.read_csv(f'{synthetic_prefix}/switzerland_activities_geo.csv', sep=',', header=0)
    output_trips = pd.read_csv(f'{synthetic_prefix}/switzerland_trips_geo.csv', sep=',', header=0)
    output_households = pd.read_csv(f'{synthetic_prefix}/switzerland_households_geo.csv', sep=',', header=0)
    output_persons = pd.read_csv(f'{synthetic_prefix}/switzerland_persons_geo.csv', sep=',', header=0)
    output_households['household_weight'] = 1
    output_trips['person_weight'] = 1

    microcensus = {'activities': activities, 'trips': trips, 'households': households, 'persons': persons}
    synthetic = {'activities': output_activities, 'trips': output_trips, 'households': output_households, 'persons': output_persons}
    return microcensus, synthetic


def load_cache_data(directory, prefix):
    files = dict()
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.p'):
            with open(directory + '/' + filename, 'rb') as file:
                data = pickle.load(file)
                files[filename] = data
    return files


def configure(context):
    context.stage("analysis.process_output")
    context.stage("analysis.process_microcensus")

    context.config("working_directory")
    context.config("analysis_path")


def execute(context):
    prefix = 'analysis'
    directory = context.config('working_directory')
    save_directory = context.config("analysis_path")

    # Read in PKL data
    microcensus = dict()
    synthetic = dict()
    files = load_cache_data(directory, prefix)
    for filename, data in files.items():
        if 'microcensus' in filename:
            microcensus['persons'] = data[0]
            microcensus['households'] = data[1]
            microcensus['trips'] = data[2]
            microcensus['activities'] = data[3]
        if 'output' in filename: 
            synthetic['persons'] = data[0]
            synthetic['households'] = data[1]
            synthetic['trips'] = data[2]
            synthetic['activities'] = data[3]
    
    write_all_application_data(microcensus=microcensus,
                               synthetic=synthetic,
                               save_directory=save_directory)


if __name__ == '__main__':
    """ 
    Paths for testing purposes
    microcensus_prefix = '/cluster/project/cmdp/chaoch/microcensus_data_test'
    synthetic_prefix = '/cluster/project/cmdp/chaoch/switzerland_data/output_test'
    save_directory = '/cluster/home/chaoch/ch/ch-zh-synpop/plot_data'
    """
    microcensus_prefix = input('Enter the directory where the _processed_ microcensus data is stored:').strip()
    synthetic_prefix = input('Enter the directory where the _processed_ synthetic data is stored:').strip()
    save_directory= input('Enter the directory where plotting data should be stored').strip()
    
    microcensus, synthetic = load_dataframes(microcensus_prefix, synthetic_prefix)

    write_all_application_data(
        microcensus,
        synthetic,
        save_directory
    )

# process_output = {
#     "configure": configure,
#     "execute": execute
# }