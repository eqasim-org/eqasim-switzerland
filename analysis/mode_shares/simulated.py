import pandas as pd
from .utils import ModeShareAnalyzer
import logging

logger = logging.getLogger("synpp")

def configure(context):    
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.spatial.swiss_border")
    # context.stage("synthesis.population.enriched")        
    # context.stage("matsim.output")
    context.stage("data.external_population.constants")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output") 

def execute(context):
    # ensure dependency
    # _ = context.stage("matsim.output")

    mode_shares_analyzer = ModeShareAnalyzer(context, from_matsim = True)
    
    # Compute mode shares
    mode_shares = dict()
    mode_shares["global"] = mode_shares_analyzer.compute_mode_shares(consider_external_population=False)
    mode_shares["distance"] = mode_shares_analyzer.compute_mode_shares_by("distance_bin", consider_external_population=False)
    mode_shares["canton"] = mode_shares_analyzer.compute_mode_shares_by("canton_id", consider_external_population=True)
    mode_shares["income"] = mode_shares_analyzer.compute_mode_shares_by("income_class", consider_external_population=False)
    mode_shares["age"]    = mode_shares_analyzer.compute_mode_shares_by("age_class", consider_external_population=False)    
    mode_shares["sex"]    = mode_shares_analyzer.compute_mode_shares_by("sex", consider_external_population=False)    
    mode_shares["purpose"]    = mode_shares_analyzer.compute_mode_shares_by("purpose", consider_external_population=False)
    
    mode_shares["mode_distance"] = mode_shares_analyzer.compute_mode_distribution_by("distance_bin", consider_external_population=False)
    mode_shares["mode_canton"] = mode_shares_analyzer.compute_mode_distribution_by("canton_id", consider_external_population=True)
    mode_shares["mode_income"] = mode_shares_analyzer.compute_mode_distribution_by("income_class", consider_external_population=False)
    mode_shares["mode_age"]    = mode_shares_analyzer.compute_mode_distribution_by("age_class", consider_external_population=False)            
    mode_shares["mode_sex"]    = mode_shares_analyzer.compute_mode_distribution_by("sex", consider_external_population=False)
    
    mode_shares["distance_bins"] = ModeShareAnalyzer.distance_bins
    mode_shares["distance_labels"] = mode_shares_analyzer.get_distance_labels()
    mode_shares["age_bins"] = ModeShareAnalyzer.age_bins
    mode_shares["age_labels"] = mode_shares_analyzer.get_age_labels()
    mode_shares["distance_by_mode"] = mode_shares_analyzer.compute_distance_by_mode()
    mode_shares["distance_by_mode_and_purpose"] = mode_shares_analyzer.compute_distance_by_mode_and_purpose()
    
    return mode_shares