import pandas as pd
from data.microcensus.shares import load_clean_trips
import os
import numpy as np
import logging

logger = logging.getLogger("synpp")

class ModeShareAnalyzer:
    distance_bins = [0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000] # in meters  
    age_bins = [0, 17, 25, 35, 50, 65, 100]

    def __init__(self, context, from_matsim = False):
        self.from_matsim = from_matsim
        if from_matsim:
            self.load_matsim_data(context)
        else:
            self.load_microcensus_data(context)        

    def load_microcensus_data(self, context):
        trips = load_clean_trips(context)

        # Ensure correct data types
        trips["canton_id"] = trips["canton_id"].astype(int)
        trips["income_class"] = trips["income_class"].astype(int)        
        trips["euclidean_distance_km"] = trips["crowfly_distance"] / 1000

        # Define distance bins for mode shares distributions    
        trips['distance_bin'] = self.get_distance_bins(trips)

        # Define age bins for mode shares distributions
        trips['age_class'] = self.get_age_bins(trips)                
        self.trips = trips

    def get_paths_matsim(self, context):
        output_path = context.config("output_path")
        output_id = context.config("output_id")
        simulation_directory = context.config("simulation_directory")          
        trips_path_matsim = f"{output_path}/{output_id}/{simulation_directory}/output_trips.csv.gz"
        persons_path_matsim = f"{output_path}/{output_id}/{simulation_directory}/output_persons.csv.gz"
        assert os.path.exists(trips_path_matsim), f"MATSIM trips file not found at {trips_path_matsim}"
        assert os.path.exists(persons_path_matsim), f"MATSIM persons file not found at {persons_path_matsim}"
        return trips_path_matsim, persons_path_matsim
    
    def get_distance_labels(self):
        distance_bins = np.array(ModeShareAnalyzer.distance_bins) / 1000
        distance_labels = [
            f"{i}-({distance_bins[i]}km-{distance_bins[i+1]}km)"
            if distance_bins[i+1] < distance_bins[-1]
            else f"{i}-({distance_bins[i]}km+)"
            for i in range(len(distance_bins)-1)
        ]
        return distance_labels

    def get_distance_bins(self, df):
        distance_bins = np.array(ModeShareAnalyzer.distance_bins) / 1000
        distance_labels = self.get_distance_labels()
        return pd.cut( df['euclidean_distance_km'], 
                       bins=distance_bins, 
                       labels=distance_labels, 
                       include_lowest=True, 
                       ordered=True)

    def get_age_labels(self):
        age_bins = ModeShareAnalyzer.age_bins
        age_labels = [
            f"{age_bins[i]+1}-{age_bins[i+1]}" if age_bins[i+1]<age_bins[-1] else f"{age_bins[i]}+"
            for i in range(len(age_bins)-1)
        ]
        return age_labels

    def get_age_bins(self, df):
        age_bins = ModeShareAnalyzer.age_bins.copy()
        age_labels = self.get_age_labels()
        return pd.cut( df['age'], 
                       bins=age_bins, 
                       labels=age_labels, 
                       include_lowest=True, 
                       ordered=True)
    
    def get_matsim_canton_id(self, context, df_trips):        
        df = df_trips[["person"]].copy()
        # canton level mode shares
        df_persons = context.stage("synthesis.population.enriched")
        df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name_en"]]
        
        # merge canton names
        df_cantons = df_cantons.rename(columns={"canton_name_en":"canton_name"})
        df_persons = pd.merge(df_persons, df_cantons, on="canton_id", how="left")
        df_persons = df_persons[["person_id", "canton_name","canton_id"]].astype({"person_id":str})    

        df = pd.merge(df, df_persons, right_on = "person_id",
                            left_on="person", how="left")
        df = df[df.canton_id.notna()].reset_index(drop=True)
        
        if len(df) != len(df_trips):
            logger.warning(f"{len(df_trips) - len(df)} trips could not be assigned a canton_id.")
            logger.warning(f"These trips will be ignored for canton level mode share computations.")

        return df[["person", "canton_id","canton_name"]]
    
    def get_income_class(self, context, df):
        df = df[["person"]].copy()
        df_persons = context.stage("synthesis.population.enriched")[["person_id","income_class"]].astype({"person_id":str})
        df = pd.merge(df, df_persons, right_on = "person_id",
                            left_on="person", how="left")
        df = df[df.income_class.notna()].reset_index(drop=True)
        return df[["person", "income_class"]]

    def load_matsim_data(self, context):
        trips_file, persons_file = self.get_paths_matsim(context)
        
        # Load trips and persons data
        persons = pd.read_csv(persons_file, dtype={0: str}, sep=";", usecols=["person","subpopulation","age","sex", "cantonId","cantonName"])
        persons = persons.astype({"person":str})
        external_persons = persons.subpopulation.isin(['crossborder', 'freight'])
        logger.info("Excluding %d external persons over %d persons from MATSIM data.", external_persons.sum(), len(persons))
        persons = persons[~external_persons]

        trips = pd.read_csv(trips_file, sep=";", dtype={0: str, 1: str}, usecols=["trip_id", "person", "main_mode", "euclidean_distance"])
        trips = trips.astype({"person":str})
        trips = trips[trips.person.isin(persons.person)]
        trips = trips[trips.main_mode.isin(["car","car_passenger","pt","bike","walk"])]

        # distance based filter
        trips["euclidean_distance_km"] = trips["euclidean_distance"] / 1000
        trips = trips[trips.euclidean_distance_km>1e-3].reset_index(drop=True)

        # assign income class
        income_class_info = self.get_income_class(context, persons)
        trips = trips.merge(income_class_info, on='person', how='left')

        # Define distance bins for mode shares distributions    
        trips['distance_bin'] = self.get_distance_bins(trips)

        # Define age bins for mode shares distributions        
        persons['age_class'] = self.get_age_bins(persons)                
        trips = trips.merge(persons[['person','age_class','sex','cantonId','cantonName']], on='person', how='left')
        
        # nans filter
        #trips = trips[(trips.cantonId.notna()) & (trips.income_class.notna()) & (trips.sex.notna())].reset_index(drop=True)

        # to be able to use the same functions for mode shares, we use a person_weight of 1
        trips['person_weight'] = 1.0

        # rename columns to match microcensus naming
        trips = trips.rename(columns={"main_mode":"mode",
                                      "cantonId":"canton_id",
                                      "cantonName":"canton_name"})
        sex_dict = {"m":0,"f":1}
        trips["sex"] = trips["sex"].str.lower().apply(lambda x: sex_dict.get(x, x))
        trips = trips[["trip_id","person","mode","euclidean_distance_km",
                       "distance_bin","canton_id","canton_name",
                       "income_class","age_class","sex","person_weight"]]

        self.trips = trips

    def compute_mode_shares(self):
        total_person_weight = self.trips['person_weight'].sum()
        mode_share_person = (
            self.trips.groupby('mode')
            .apply(lambda x: x['person_weight'].sum())
            .reset_index(name='mode_share')
        )
        mode_share_person['mode_share'] /= total_person_weight
        mode_share_person = mode_share_person.set_index("mode")             
        return mode_share_person[['mode_share']]

    def compute_mode_shares_by(self, by = "canton_id"):
        mode_share =  (self.trips
                        .groupby([by, "mode"], observed=False)["person_weight"]
                        .sum()
                        .groupby(level=by, observed = False)
                        .transform(lambda x: x / x.sum())                
                        .rename("mode_share")
                        .reset_index()                
                        .pivot(index=by, columns="mode", values="mode_share")
                        .fillna(0)
                        .sort_values(by=by))
                   
        return mode_share

    def compute_mode_distribution_by(self, by = "distance_bin"):
        mode_share =  (self.trips
                        .groupby([by, "mode"], observed=False)["person_weight"]
                        .sum()
                        .groupby(level="mode", observed = False)
                        .transform(lambda x: x / x.sum())                
                        .rename("distribution")
                        .reset_index()                
                        .pivot(index=by, columns="mode", values="distribution")
                        .fillna(0)
                        .sort_values(by=by))
                            
        return mode_share