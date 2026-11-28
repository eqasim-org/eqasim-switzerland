import pandas as pd
import geopandas as gpd
import shapely
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


FRENCH_MODES = ["car", "pt", "walk", "bike", "car_passenger"]
FRENCH_COHORTS = {
    "all_fr_resident_trips": "French residents: all trips",
    "switzerland_involved": "French residents: origin or destination in Switzerland",
}
MODE_LABELS = {"car": "Car driver", "car_passenger": "Car passenger", "pt": "Public transport",
               "walk": "Walking", "bike": "Cycling"}


class ModeShareAnalyzer:
    distance_bins = [0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000] # in meters  
    age_bins      = [0, 17, 25, 35, 50, 65, 100]
    
    @staticmethod
    def set_distance_bins(bins):
        ModeShareAnalyzer.distance_bins = bins

    @staticmethod
    def set_age_bins(bins):
        ModeShareAnalyzer.age_bins = bins


    def __init__(self, context, from_matsim=False, *, group_income_classes=False, source=None):
        """Load trips, optionally grouping incomes into low/medium/high (1-3).

        ``source`` selects microcensus, matsim or eqasim. The legacy from_matsim
        argument remains supported. Grouping applies to all sources; enable it
        on both analyzers when comparing results. By default, retain classes 0-8.
        """
        self.source = source or ('matsim' if from_matsim else 'microcensus')
        if self.source not in ('matsim', 'microcensus', 'eqasim'):
            raise ValueError('Mode-share source must be matsim, microcensus or eqasim')
        self.from_matsim = self.source != 'microcensus'  # population data use unit weights
        self.ex_cst      = context.stage("data.external_population.constants")

        if self.source == 'eqasim':
            self.load_eqasim_data(context)
        elif self.source == 'matsim':
            self.load_matsim_data(context)

        else:
            self.load_microcensus_data(context)

        if group_income_classes:
            self.trips["income_class"] = self.trips["income_class"].map(INCOME_CLASSIFICATION)


    def load_eqasim_data(self, context):
        from .inputs import prepared_trips
        self.trips = prepared_trips(context.stage('analysis_population'),
                                    context.stage('synthesis.prepared.households'))
        self.trips['distance_bin'] = self.get_distance_bins(self.trips)
        self.trips['age_class'] = self.get_age_bins(self.trips)


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
                   "destination_x", "destination_y", "home_x", "home_y", "euclidean_distance_km"]
        result = frame[columns].rename(columns={person_column: "person_id", "person_weight": "weight",
                                                "euclidean_distance_km": "distance_km"}).copy()
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
                        .reindex(columns=FRENCH_MODES, fill_value=0)
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
                        .reindex(columns=FRENCH_MODES, fill_value=0)
                        .sort_values(by = by))
                            
        return mode_share

    def compute_distance_by_mode(self, df = None, consider_external_population:bool = True):
        if df is None:
            df = self.trips.copy()

        if not consider_external_population:
            df = df[df.canton_id != self.ex_cst.canton_id].reset_index(drop = True)

        distance_distribution = dict()

        for mode in FRENCH_MODES:
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


def load_french_target(context):
    """Reuse the survey underlying data.microcensus.shares without modifying its cache."""
    raw = context.stage("data.external_population.hts_trips.trips")
    target = raw.rename(columns={"person_id": "person", "mode": "main_mode"}).copy()
    valid = (target.main_mode.isin(FRENCH_MODES) & target.DOIB.gt(1)
             & np.isfinite(target.trip_weight) & target.trip_weight.gt(0))
    target = target.loc[valid].copy()
    target["distance_km"] = target.DOIB / 1000

    # External zones explicitly distinguish Switzerland from France. Internal survey sectors are French.

    folder = Path(context.config("data_path")) / "hts_annemasse_2017_tpg" / "TPG" / "Doc" / "SIG"
    fine = gpd.read_file(folder / "EDGTFVG2016_ZF.TAB", ignore_geometry=True)
    external = gpd.read_file(folder / "EDGTFVG2016_ZonesExternes.TAB", ignore_geometry=True)
    external = external.loc[external.Num_ZF.astype("string").str.strip().ne("").fillna(False)]
    country = external.set_index("Num_ZF").NOM_D10.eq("Suisse").map({True: "CH", False: "FR"})
    if country.groupby(level=0).nunique().gt(1).any():
        raise ValueError("Conflicting countries for a survey external zone.")
    country = country.groupby(level=0).first()
    sectors = set(fine.ZoneFine.astype(str).str[:3])
    countries = []
    for column in ["LIEU_DEPART_FR", "LIEU_ARRIVEE_FR"]:
        zone = target[column].astype(str)
        mapped = zone.map(country)
        mapped = mapped.mask(zone.str[:3].isin(sectors) & mapped.isna(), "FR")
        countries.append(mapped)
    target["switzerland_involved"] = countries[0].eq("CH") | countries[1].eq("CH")
    metadata = {
        "stage": "data.external_population.hts_trips.trips",
        "file": str(folder.parent.parent / "deplacements_emd_full_OCT.csv"),
        "distance": "DOIB, crow-fly distance in metres; DIST is not used.",
        "weight": "trip_weight, derived from survey POND_PERS",
        "coverage": "Annemasse/French Geneva-region survey, as used by data.microcensus.shares; not the whole French simulated region.",
        "switzerland_selection": "Origin OR destination labelled Suisse in survey external zones; includes trips entirely within Switzerland. An unknown endpoint does not exclude a trip if its other endpoint is known to be Swiss.",
        "excluded_invalid_mode_distance_or_weight": int((~valid).sum()),
        "unknown_endpoint_country_sample_trips": int((countries[0].isna() | countries[1].isna()).sum()),
        "sample_trips": len(target),
        "switzerland_involved_sample_trips": int(target.switzerland_involved.sum()),
        "both_endpoints_in_switzerland_sample_trips": int((countries[0].eq("CH") & countries[1].eq("CH")).sum()),
    }
    return target, metadata


def summarize_shares(trips, source, edges, labels):
    trips = trips.copy()
    trips["distance_bin"] = pd.cut(trips.distance_km, edges, labels=labels, include_lowest=True)
    if trips.distance_bin.isna().any():
        raise ValueError("Distance bins do not cover all retained trips.")
    overall, binned = [], []
    for cohort in FRENCH_COHORTS:
        selected = trips if cohort == "all_fr_resident_trips" else trips.loc[trips.switzerland_involved]
        total_weight = selected.trip_weight.sum()
        counts = selected.main_mode.value_counts().reindex(FRENCH_MODES, fill_value=0)
        weights = selected.groupby("main_mode").trip_weight.sum().reindex(FRENCH_MODES, fill_value=0)
        for mode in FRENCH_MODES:
            overall.append({"source": source, "cohort": cohort, "mode": mode,
                            "sample_trips": int(counts[mode]), "cohort_sample_trips": len(selected),
                            "sample_persons_with_trips": selected.person.nunique(),
                            "weighted_trips": weights[mode], "cohort_weighted_trips": total_weight,
                            "share": weights[mode] / total_weight if total_weight else np.nan})
        for label in labels:
            group = selected.loc[selected.distance_bin.eq(label)]
            denominator = group.trip_weight.sum()
            counts = group.main_mode.value_counts().reindex(FRENCH_MODES, fill_value=0)
            weights = group.groupby("main_mode").trip_weight.sum().reindex(FRENCH_MODES, fill_value=0)
            for mode in FRENCH_MODES:
                binned.append({"source": source, "cohort": cohort, "distance_bin_km": label, "mode": mode,
                               "sample_trips": int(counts[mode]), "bin_sample_trips": len(group),
                               "weighted_trips": weights[mode], "bin_weighted_trips": denominator,
                               "share": weights[mode] / denominator if denominator else np.nan})
    return pd.DataFrame(overall), pd.DataFrame(binned)


def endpoint_in_switzerland(x, y, swiss):
    """Test each endpoint independently so a missing opposite endpoint does not hide a trip."""
    return np.isfinite(x) & np.isfinite(y) & shapely.intersects_xy(swiss, x, y)


def french_mode_shares(trips, source, metadata):
    """Summarize the two French-resident cohorts without retaining raw trip tables."""
    edges = np.asarray(ModeShareAnalyzer.distance_bins, dtype=float) / 1000
    edges[-1] = np.inf
    labels = [f"{left:g}–{right:g}" if np.isfinite(right) else f"≥{left:g}"
              for left, right in zip(edges[:-1], edges[1:])]
    overall, distance = summarize_shares(trips, source, edges, labels)
    return {"overall": overall, "distance": distance, "distance_labels": labels,
            "metadata": metadata}


def simulated_french_mode_shares(analyzer, context):
    """Reuse the simulation analyzer's filtered trips, selecting residence rather than nationality."""
    trips = analyzer.trips.loc[analyzer.trips.canton_id.eq(analyzer.ex_cst.canton_id)].copy()
    trips = trips.rename(columns={"mode": "main_mode", "euclidean_distance_km": "distance_km",
                                  "person_weight": "trip_weight"})
    swiss = context.stage("data.spatial.swiss_border").to_crs(2056).geometry.union_all()
    shapely.prepare(swiss)
    origin = endpoint_in_switzerland(trips.origin_x, trips.origin_y, swiss)
    destination = endpoint_in_switzerland(trips.destination_x, trips.destination_y, swiss)
    trips["switzerland_involved"] = origin | destination
    metadata = {
        "simulation": "mode_choice.run" if getattr(analyzer, "source", None) == "eqasim" else str(get_simulation_path(context)),
        "population": "French residents (canton FR), excluding freight and crossborder subpopulations.",
        "sample_trips": len(trips),
        "switzerland_involved_sample_trips": int((origin | destination).sum()),
        "both_endpoints_in_switzerland_sample_trips": int((origin & destination).sum()),
        "distance": "Population Euclidean distance in kilometres; trips greater than one metre.",
        "weight": "One per simulated trip.",
    }
    return french_mode_shares(trips, "simulation", metadata)


def figures_directory(context):
    """Keep pre-simulation choice diagnostics separate from simulation figures."""
    if context.config('analysis.mode_shares.from_eqasim'):
        return Path(context.config('output_path')) / context.config('output_id') / 'mode_shares_eqasim'
    return Path(get_simulation_path(context)) / 'mode_shares'
