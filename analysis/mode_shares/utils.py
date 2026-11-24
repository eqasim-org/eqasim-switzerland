import pandas as pd
from data.microcensus.shares import load_clean_trips
import os
import numpy as np
import logging
from matsim.readers import houshold_reader
from pathlib import Path
from analysis.counts.paths import get_simulation_path
from matsim.config.tools import resolve_output_file, glob_output_files

logger = logging.getLogger("synpp")

# Optional grouping used locally by synthesis/population/matching/matched_v1.py.
# Survey and exported Swiss population incomes otherwise retain classes 0-8.
INCOME_CLASSIFICATION = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}


class ModeShareAnalyzer:
    distance_bins = [0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000] # in meters  
    age_bins      = [0, 17, 25, 35, 50, 65, 100]
    
    @staticmethod
    def set_distance_bins(bins):
        ModeShareAnalyzer.distance_bins = bins

    @staticmethod
    def set_age_bins(bins):
        ModeShareAnalyzer.age_bins = bins


    def __init__(self, context, from_matsim=False, *, group_income_classes=False):
        """Load trips, optionally grouping incomes into low/medium/high (1-3).

        Grouping applies to either data source; enable it on both analyzers when
        comparing survey and simulation data. By default, retain classes 0-8.
        """
        self.from_matsim = from_matsim
        self.ex_cst      = context.stage("data.external_population.constants")

        if from_matsim:
            self.load_matsim_data(context)

        else:
            self.load_microcensus_data(context)

        if group_income_classes:
            self.trips["income_class"] = self.trips["income_class"].map(INCOME_CLASSIFICATION)


    def load_microcensus_data(self, context):
        trips = load_clean_trips(context)

        # Preserve residence for the interactive resident-based spatial selector.

        homes = context.stage("data.microcensus.persons")[["person_id", "home_x", "home_y"]].copy()
        if not {"home_x", "home_y"}.issubset(trips):
            trips = trips.merge(homes, on="person_id", how="left", validate="many_to_one")

        # Ensure correct data types
        trips["canton_id"]             = trips["canton_id"].astype(int)
        trips["income_class"]          = trips["income_class"].astype(int)
        trips["euclidean_distance_km"] = trips["crowfly_distance"] / 1000

        # Define distance bins for mode shares distributions    
        trips['distance_bin'] = self.get_distance_bins(trips)

        # Define age bins for mode shares distributions
        trips['age_class'] = self.get_age_bins(trips)  

        # correct purposes
        trips["purpose"] = trips["purpose"].str.replace("_secondary", "", regex=False)        
        self.trips = trips


    def get_paths_matsim(self, context):
        simulation_path = get_simulation_path(context)
        # Never take population files from another run when a specific
        # simulation directory was requested. Keep the legacy fallback otherwise.
        population_search_path = (
            simulation_path if context.config("analysis.counts.simulation_path")
            else os.path.join(context.config("output_path"), context.config("output_id"))
        )
        
        # get the trips file, if simulation not finished, get one from the intermediate iterations
        trips_path_matsim = resolve_output_file(os.path.join(simulation_path, "output_trips.csv.gz"), required=False)

        if not os.path.exists(trips_path_matsim):
            candidate_files = glob_output_files(os.path.join(simulation_path, "ITERS", "**", "*.trips.csv.gz"))

            if candidate_files:
                trips_path_matsim = max(candidate_files, key=os.path.getctime)  # Get the most recently created file
                logger.warning(f"MATSIM trips file not found as an output file. Using {trips_path_matsim} instead.")

            else:
                raise FileNotFoundError(f"MATSIM trips file not found as an output file and no candidate files found in {simulation_path}/ITERS/")
        
        # get the persons file
        persons_path_matsim = resolve_output_file(os.path.join(simulation_path, "output_persons.csv.gz"), required=False)

        if not os.path.exists(persons_path_matsim):
            candidate_files = glob_output_files(os.path.join(population_search_path, "**", "output_persons.csv.gz"))

            if candidate_files:
                persons_path_matsim = max(candidate_files, key=os.path.getctime)  # Get the most recently created file
                logger.warning(f"MATSIM persons file not found as an output file. Using {persons_path_matsim} instead.")

            else:
                raise FileNotFoundError(f"MATSIM persons file not found as an output file and no candidate files found in {population_search_path}")

        # get the households file
        households_path_matsim = resolve_output_file(os.path.join(simulation_path, "output_households.xml.gz"), required=False)

        if not os.path.exists(households_path_matsim):
            candidate_files = glob_output_files(os.path.join(population_search_path, "**", "output_households.xml.gz"))

            if candidate_files:
                households_path_matsim = max(candidate_files, key=os.path.getctime)  # Get the most recently created file
                logger.warning(f"MATSIM households file not found as an output file. Using {households_path_matsim} instead.")

            else:
                raise FileNotFoundError(f"MATSIM households file not found as an output file and no candidate files found in {population_search_path}")
            
        assert os.path.exists(trips_path_matsim), f"MATSIM trips file not found at {trips_path_matsim}"
        assert os.path.exists(persons_path_matsim), f"MATSIM persons file not found at {persons_path_matsim}"
        assert os.path.exists(households_path_matsim), f"MATSIM households file not found at {households_path_matsim}"
        return trips_path_matsim, persons_path_matsim, households_path_matsim

    
    def get_distance_labels(self):
        distance_bins   = np.round(np.array(ModeShareAnalyzer.distance_bins) / 1000,1)
        distance_labels = [
            f"{i}-({distance_bins[i]}km-{distance_bins[i+1]}km)"
            if distance_bins[i+1] < distance_bins[-1]
            else f"{i}-({distance_bins[i]}km+)"
            for i in range(len(distance_bins)-1)
        ]

        return distance_labels


    def get_distance_bins(self, df):
        distance_bins   = np.array(ModeShareAnalyzer.distance_bins) / 1000
        distance_labels = self.get_distance_labels()
        return pd.cut( df["euclidean_distance_km"], 
                       bins           = distance_bins, 
                       labels         = distance_labels, 
                       include_lowest = True, 
                       ordered        = True)

    def get_age_labels(self):
        age_bins   = ModeShareAnalyzer.age_bins
        age_labels = [
            f"{age_bins[i]+1}-{age_bins[i+1]}" if age_bins[i+1]<age_bins[-1] else f"{age_bins[i]}+"
            for i in range(len(age_bins)-1)
        ]

        return age_labels

    def get_age_bins(self, df):
        age_bins   = ModeShareAnalyzer.age_bins.copy()
        age_labels = self.get_age_labels()
        return pd.cut( df["age"], 
                       bins           = age_bins, 
                       labels         = age_labels, 
                       include_lowest = True, 
                       ordered        = True)
    
    
    def get_income_class(self, df, households):
        df     = df[["person"]].copy()
        income = households.households[["members", "incomeClass"]].explode("members").rename(columns = {"members": "person", 
                                                                                                        "incomeClass": "income_class"})
        income = income.astype({"person": str, "income_class": int})
        
        df = pd.merge(df, income, on = "person", how = "left")
        logger.info(f"There are {df.income_class.isna().sum()} persons with no income class assigned over {len(df)} persons.")

        df = df[df.income_class.notna()].reset_index(drop = True)

        return df[["person", "income_class"]]


    def load_matsim_data(self, context):
        trips_file, persons_file, households_file = self.get_paths_matsim(context)
        
        # Load trips and persons data
        person_columns = ["person", "subpopulation", "age", "sex", "cantonId", "cantonName",
                          "home_coordinate_x", "home_coordinate_y", "home_x", "home_y",
                          "first_act_x", "first_act_y", "first_act_type"]
        # Resolve optional columns explicitly. Pandas' chunked C parser can fail
        # with callable usecols when MATSim attributes mix strings and numbers.
        available = pd.read_csv(persons_file, sep=";", nrows=0).columns
        persons = pd.read_csv(persons_file, dtype={"person": str}, sep=";", low_memory=False,
                              usecols=[column for column in person_columns if column in available])
        persons = persons.astype({"person": str})

        # Prefer explicit residential coordinates; only home first activities are a fallback.

        for axis in ("x", "y"):
            home = pd.Series(np.nan, index=persons.index)
            for column in ("home_coordinate_" + axis, "home_" + axis):
                if column in persons:
                    value = pd.to_numeric(persons[column], errors="coerce")
                    home = home.fillna(value.where(np.isfinite(value)))
            if "first_act_" + axis in persons and "first_act_type" in persons:
                value = pd.to_numeric(persons["first_act_" + axis], errors="coerce")
                home = home.fillna(value.where(persons.first_act_type.eq("home") & np.isfinite(value)))
            persons["home_" + axis] = home

        external_persons = persons.subpopulation.isin(["crossborder", "freight"])
        logger.info("Excluding %d external persons over %d persons from MATSIM data." %(external_persons.sum(), len(persons)))
        persons = persons[~external_persons]

        trips = pd.read_csv(trips_file, sep=";", dtype={"person": str, "trip_id": str},
                            usecols=["trip_id", "person", "main_mode", "euclidean_distance",
                                     "start_activity_type", "end_activity_type", "start_x", "start_y", "end_x", "end_y"])
        trips = trips.astype({"person": str})

        # Airport activities are outside this comparison, at either trip endpoint.

        airport = (trips.start_activity_type.astype("string").str.strip().str.lower().eq("airport")
                   | trips.end_activity_type.astype("string").str.strip().str.lower().eq("airport")).fillna(False)
        logger.info("Excluding %d MATSim trips with an airport origin or destination", int(airport.sum()))
        trips = trips.loc[~airport].copy()
        trips = trips[trips.person.isin(persons.person)]
        trips = trips[trips.main_mode.isin(["car","car_passenger","pt","bike","walk"])]

        # distance based filter
        trips["euclidean_distance_km"] = trips["euclidean_distance"] / 1000
        trips = trips[trips.euclidean_distance_km > 1e-3].reset_index(drop = True)

        # read households
        households = houshold_reader(households_file, convert_dataframes_types=False)
        if ("incomeClass" not in households.households
                or households.households["incomeClass"].isna().all()):
            input_file = resolve_output_file(Path(get_simulation_path(context)).parent / (
                context.config("output_prefix") + "households.xml.gz"))
            logger.info("Output households lack income classes; reading %s", input_file)
            households = houshold_reader(str(input_file), convert_dataframes_types=False)

        # assign income class from households
        income_class_info = self.get_income_class(persons, households)
        trips = trips.merge(income_class_info, on = "person", how = "left")

        # Define distance bins for mode shares distributions    
        trips["distance_bin"] = self.get_distance_bins(trips)

        # Define age bins for mode shares distributions        
        persons["age_class"] = self.get_age_bins(persons)                
        trips = trips.merge(persons[["person", "age_class", "sex", "cantonId", "cantonName", "home_x", "home_y"]],
                            on="person", how="left", validate="many_to_one")
        
        # purposes
        trips            = trips.rename(columns = {"end_activity_type": "purpose"})
        trips["purpose"] = trips["purpose"].str.replace("_secondary", "", regex = False)
        trips            = trips[trips["purpose"].isin(["work", "home", "other", "shop", "education", "leisure"])].reset_index(drop = True)

        # nans filter
        #trips = trips[(trips.cantonId.notna()) & (trips.income_class.notna()) & (trips.sex.notna())].reset_index(drop=True)

        # to be able to use the same functions for mode shares, we use a person_weight of 1
        trips["person_weight"] = 1.0

        # rename columns to match microcensus naming
        trips = trips.rename(columns={"main_mode":  "mode",
                                      "cantonId":   "canton_id",
                                      "cantonName": "canton_name"})
        trips = trips.rename(columns={"start_x": "origin_x", "start_y": "origin_y",
                                      "end_x": "destination_x", "end_y": "destination_y"})
        sex_dict = {"m": 0, "f": 1}
        trips["sex"] = trips["sex"].str.lower().apply(lambda x: sex_dict.get(x, x))
        trips = trips[["trip_id", "person", "mode", "euclidean_distance_km",
                       "distance_bin", "canton_id", "canton_name", "purpose",
                       "income_class", "age_class", "sex", "person_weight",
                       "origin_x", "origin_y", "destination_x", "destination_y", "home_x", "home_y"]]

        self.trips = trips


    def spatial_trips(self, consider_external_population=False):
        """Return the same filtered, weighted cohort used for global mode shares.

        One row is one retained trip. Residence is only a spatial selector: the
        denominator remains trips, with a respondent's person_weight repeated
        for each of their survey trips. Cached inputs are never modified.
        """
        frame = self.trips
        if not consider_external_population:
            frame = frame.loc[frame.canton_id != self.ex_cst.canton_id]
        person_column = "person" if self.from_matsim else "person_id"
        columns = [person_column, "trip_id", "mode", "person_weight", "origin_x", "origin_y",
                   "destination_x", "destination_y", "home_x", "home_y"]
        result = frame[columns].rename(columns={person_column: "person_id", "person_weight": "weight"}).copy()
        result["person_id"] = result["person_id"].astype(str)
        return result.reset_index(drop=True)


    def compute_mode_shares(self, consider_external_population: bool = True):
        df = self.trips.copy()

        if not consider_external_population:
            df = df[df.canton_id != self.ex_cst.canton_id].reset_index(drop = True)

        total_person_weight = df["person_weight"].sum()
        mode_share_person   = (
            df.groupby("mode")
            .apply(lambda x: x["person_weight"].sum())
            .reset_index(name = "mode_share")
        )
        mode_share_person["mode_share"] /= total_person_weight
        mode_share_person = mode_share_person.set_index("mode")             
        return mode_share_person[["mode_share"]]


    def compute_mode_shares_by(self, by = "canton_id", consider_external_population:bool = True):
        df = self.trips.copy()

        if not consider_external_population:
            df = df[df.canton_id != self.ex_cst.canton_id].reset_index(drop = True)

        mode_share =  (df
                        .groupby([by, "mode"], observed=False)["person_weight"]
                        .sum()
                        .groupby(level = by, observed = False)
                        .transform(lambda x: x / x.sum())                
                        .rename("mode_share")
                        .reset_index()                
                        .pivot(index = by, columns = "mode", values = "mode_share")
                        .fillna(0)
                        .sort_values(by = by))
                   
        return mode_share


    def compute_mode_distribution_by(self, by = "distance_bin", consider_external_population:bool = True):
        df = self.trips.copy()

        if not consider_external_population:
            df = df[df.canton_id!=self.ex_cst.canton_id].reset_index(drop=True)

        mode_share =  (df
                        .groupby([by, "mode"], observed = False)["person_weight"]
                        .sum()
                        .groupby(level="mode", observed = False)
                        .transform(lambda x: x / x.sum())                
                        .rename("distribution")
                        .reset_index()                
                        .pivot(index = by, columns = "mode", values = "distribution")
                        .fillna(0)
                        .sort_values(by = by))
                            
        return mode_share

    def compute_distance_by_mode(self, df = None, consider_external_population:bool = True):
        if df is None:
            df = self.trips.copy()

        if not consider_external_population:
            df = df[df.canton_id != self.ex_cst.canton_id].reset_index(drop = True)

        modes = df["mode"].unique()
        distance_distribution = dict()

        for mode in modes:
            mode_trips = df[df["mode"] == mode]
            if self.from_matsim:
                distance_distribution[mode] =  mode_trips["euclidean_distance_km"].values
            else:
                distance_distribution[mode] = { "distances": mode_trips["euclidean_distance_km"].values,
                                                "weights":   mode_trips["person_weight"].values}

        return distance_distribution

    
    def compute_distance_by_mode_and_purpose(self, df = None, consider_external_population:bool = True):
        if df is None:
            df = self.trips.copy()
        
        if not consider_external_population:
            df = df[df.canton_id != self.ex_cst.canton_id].reset_index(drop = True)

        purposes = df["purpose"].unique()
        distance_distribution = dict()

        for purpose in purposes:
            distance_distribution[purpose] = self.compute_distance_by_mode(df[df["purpose"] == purpose], consider_external_population = consider_external_population)

        return distance_distribution
