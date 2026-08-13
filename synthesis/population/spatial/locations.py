import geopandas as gpd
import pandas as pd
from data.statent.density import impute_parallel as impute_statent
from data.statpop.density import impute_parallel as impute_statpop
from data.spatial.ovgk import impute_parallel as impute_ovgk
from dmc.constants import constants
from matsim.scenario.population import HOME_DESTINATION_ID


def configure(context):
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.spatial.secondary.locations")

    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.sampled")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")
    context.stage("data.statent.density")
    context.stage("data.statpop.density")
    context.stage("data.spatial.ovgk")

    context.config("threads")


def execute(context):
    df_home = context.stage("synthesis.population.spatial.home.locations")
    df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")
    df_secondary = context.stage("synthesis.population.spatial.secondary.locations")[0]

    df_persons = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = context.stage("synthesis.population.activities")[["person_id", "activity_index", "purpose"]]

    # Home locations
    df_home_locations = df_locations[df_locations["purpose"] == "home"]
    df_home_locations = pd.merge(df_home_locations, df_persons, on="person_id")
    df_home_locations = pd.merge(df_home_locations, df_home[["household_id", "geometry"]], on="household_id")
    df_home_locations["destination_id"] = HOME_DESTINATION_ID
    df_home_locations = df_home_locations[["person_id", "activity_index", "destination_id", "geometry"]]

    # Work locations
    df_work_locations = df_locations[df_locations["purpose"] == "work"]
    df_work_locations = pd.merge(df_work_locations,
                                 df_work[["person_id", "destination_id", "geometry"]],
                                 on="person_id")
    df_work_locations = df_work_locations[["person_id", "activity_index", "destination_id", "geometry"]]

    # Education locations
    df_education_locations = df_locations[df_locations["purpose"] == "education"]
    df_education_locations = pd.merge(df_education_locations,
                                      df_education[["person_id", "destination_id", "geometry"]],
                                      on="person_id")
    df_education_locations = df_education_locations[["person_id", "activity_index", "destination_id", "geometry"]]

    # Border locations: the crossing comes straight from the same
    # data.cross_border.swiss_residents_od record synthesis.population.trips
    # already matched each crosser to, so the location is consistent with the
    # destination_country_raw carried on that same trip. Id and geometry are the
    # two halves of that one surveyed interview place -- the id is written as a
    # facility via data.cross_border.interview_places /
    # synthesis.population.destinations, and MATSim's ScenarioValidator requires
    # the activity to sit exactly on the facility it names.
    df_border_locations = df_locations[df_locations["purpose"] == "border"]

    df_cb_trips = context.stage("synthesis.population.trips")
    df_cb_trips = df_cb_trips[(df_cb_trips["preceding_purpose"] == "border") | (df_cb_trips["following_purpose"] == "border")]
    df_cb_trips = df_cb_trips[["person_id", "interview_point_id", "interview_geometry_point"]].drop_duplicates("person_id")
    df_cb_trips = df_cb_trips.rename(columns={"interview_point_id": "destination_id", "interview_geometry_point": "geometry"})

    df_border_locations = pd.merge(df_border_locations, df_cb_trips, on="person_id", how="left")
    df_border_locations = df_border_locations[["person_id", "activity_index", "destination_id", "geometry"]]

    # Secondary locations
    df_secondary_locations = df_locations[~df_locations["purpose"].isin(("home", "work", "education", "border"))].copy()
    df_secondary["activity_index"] = df_secondary["trip_index"]
    df_secondary_locations = pd.merge(df_secondary_locations,
                                      df_secondary[["person_id", "activity_index", "destination_id", "geometry"]],
                                      on=["person_id", "activity_index"], how="left")
    df_secondary_locations = df_secondary_locations[["person_id", "activity_index", "destination_id", "geometry"]]

    # Validation
    initial_count = len(df_locations)
    df_locations = pd.concat([df_home_locations, df_work_locations, df_education_locations, df_border_locations, df_secondary_locations])

    df_locations = df_locations.sort_values(by=["person_id", "activity_index"])
    final_count = len(df_locations)

    assert initial_count == final_count

    df_locations = gpd.GeoDataFrame(df_locations, crs="epsg:2056")

    # Extra attributes needed for mode choice in matsim
    df_locations_unique = df_locations[["geometry"]].drop_duplicates(subset=['geometry']).copy()

    # 1. Attach municipality to activities (TODO: Maybe this can be done in previous stages by keeping track of municipality id)
    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities,_ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type","municipality_id", "geometry"]]
    assert df_locations_unique.crs == df_municipalities.crs
    df_locations_unique = gpd.sjoin_nearest(df_locations_unique, df_municipalities, how="left").drop(columns=["index_right"])
    
    # 2. attache densities to activities
    df_locations_unique["x"] = df_locations_unique.geometry.x
    df_locations_unique["y"] = df_locations_unique.geometry.y
    threads = max(1, min(context.config("threads"), 8)) # avoid too many threads for this step as it can cause memory issues
    
    df_locations_unique = impute_statent(context, df_locations_unique, x="x", y="y", chunk_size=10_000,
                                radius=constants.EMPLOYEES_DENSITY_RADIUS, point_type="trip destination", 
                                measure="employees", output_column="employee_density", n_jobs = threads)
        
    df_locations_unique = impute_statent(context, df_locations_unique, x="x", y="y", chunk_size=10_000,
                                radius=constants.EMPLOYEES_DENSITY_RADIUS, point_type="trip destination", 
                                measure="companies", output_column="companies_density", n_jobs = threads)

    df_locations_unique = impute_statpop(context, context.stage("data.statpop.density"), df_locations_unique, 
                                x="x", y="y", chunk_size = 5000, n_jobs = threads,
                                radius=constants.POPULATION_DENSITY_RADIUS, point_type="trip destination")
    
    df_locations_unique = df_locations_unique.rename(columns={"population_density":"population_density"}) 
    
    df_locations_unique = df_locations_unique.astype({"employee_density": int, "companies_density": int, "population_density": int})
    
    # 3. attach OV Guteklasse
    df_locations_unique = impute_ovgk(context, df_locations_unique)
    
    # atache columns back to the main dataframe
    computed_columns = ['municipality_type', 'municipality_id', 'employee_density', 'companies_density', 'population_density','ovgk']
    df_locations = df_locations.merge(df_locations_unique[['geometry'] + computed_columns], on='geometry', how='left')    
    
    return df_locations
