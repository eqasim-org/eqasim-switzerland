import pandas as pd
from .utils import ModeShareAnalyzer
import logging

logger = logging.getLogger("synpp")


def configure(context):    
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.spatial.swiss_border")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")      

def execute(context):
    mode_shares_analyzer = ModeShareAnalyzer(context)
    
    # Compute mode shares
    mode_shares = dict()
    mode_shares["global"] = mode_shares_analyzer.compute_mode_shares()
    mode_shares["distance"] = mode_shares_analyzer.compute_mode_shares_by("distance_bin")
    mode_shares["canton"] = mode_shares_analyzer.compute_mode_shares_by("canton_id")
    mode_shares["income"] = mode_shares_analyzer.compute_mode_shares_by("income_class")
    mode_shares["age"]    = mode_shares_analyzer.compute_mode_shares_by("age_class")    
    mode_shares["sex"]    = mode_shares_analyzer.compute_mode_shares_by("sex")    
    mode_shares["purpose"]    = mode_shares_analyzer.compute_mode_shares_by("purpose")
    
    mode_shares["mode_distance"] = mode_shares_analyzer.compute_mode_distribution_by("distance_bin")
    mode_shares["mode_canton"] = mode_shares_analyzer.compute_mode_distribution_by("canton_id")
    mode_shares["mode_income"] = mode_shares_analyzer.compute_mode_distribution_by("income_class")
    mode_shares["mode_age"]    = mode_shares_analyzer.compute_mode_distribution_by("age_class")    
    mode_shares["mode_sex"]    = mode_shares_analyzer.compute_mode_distribution_by("sex")

    mode_shares["distance_bins"] = ModeShareAnalyzer.distance_bins
    mode_shares["distance_labels"] = mode_shares_analyzer.get_distance_labels()
    mode_shares["age_bins"] = ModeShareAnalyzer.age_bins
    mode_shares["age_labels"] = mode_shares_analyzer.get_age_labels()
    mode_shares["distance_by_mode"] = mode_shares_analyzer.compute_distance_by_mode()
    mode_shares["distance_by_mode_and_purpose"] = mode_shares_analyzer.compute_distance_by_mode_and_purpose()
    
    return mode_shares