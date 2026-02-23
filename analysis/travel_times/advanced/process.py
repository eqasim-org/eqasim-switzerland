import pandas as pd
from analysis.travel_times.run import merge_and_filter_large_differences
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

def configure(context):
    """
    Configure the pipeline stages and settings.
    """
    context.stage("analysis.travel_times.matsim.route")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.travel_times.matsim.get")
    context.stage("analysis.travel_times.APIs.get")

    context.config("travel_times_from", default="tomtom")
    # assert context.config("travel_times_from").lower() != "all", \
    #     "travel_times_from must be either 'google', 'tomtom', 'mapbox' for calibration process"
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

def load_and_merge_data(context):
    """
    Load MATSim and API travel time data, merge them, and calculate errors.
    """
    # Get routed travel times from MATSim
    csv_files = context.stage("analysis.travel_times.matsim.route")
    api = context.config("travel_times_from")
    
    if api == "all":
        logger.warning("travel_times_from is set to 'all'. Defaulting to 'tomtom' for processing.")
        api = "tomtom"
    
    routed_data_path = csv_files[1][api]
    df_matsim = pd.read_csv(routed_data_path)[["identifier", "links"]]
    df_matsim["links"] = df_matsim["links"].str.split('-')
    df_matsim = df_matsim.merge(
        context.stage("analysis.travel_times.matsim.get")[api], on="identifier", how="left"
    )

    # Get API travel times
    df_api = context.stage("analysis.travel_times.APIs.get")[api]

    # Merge dataframes and calculate error
    df = merge_and_filter_large_differences(df_api, df_matsim)
    df = df[["identifier", "departure_time_api", "travel_time_min_api", "travel_time_min_matsim", "links"]]
    df["error_min"] = df["travel_time_min_matsim"] - df["travel_time_min_api"]

    return df

def prepare_network_links(context):
    """
    Prepare network links data, filtering for car modes and adding free travel time.
    """
    network = context.stage("analysis.counts.matching.network")
    links = network.net.links
    links = links[links["modes"].str.contains(r"\bcar\b", regex=True)]
    links["municipalityType"] = links["attributes"].apply(lambda x: x.get("municipalityType", None))
    links = links[["link_id", "freespeed", "length", "municipalityType", "highway"]].set_index("link_id")
    links["free_travel_time"] = links["length"] / links["freespeed"]
    
    # Define highway categories
    highway_categories = {
        "Category 1": ["motorway", "motorway_link", "trunk", "trunk_link"],
        "Category 2": ["primary", "primary_link"],
        "Category 3": ["secondary", "secondary_link"],
        "Category 4": ["tertiary", "tertiary_link"],
        "Category 5": ["residential", "unclassified", "living_street", "service", "track"]
    }
    
    highway_to_category = {}
    for cat, types in highway_categories.items():
        for t in types:
            highway_to_category[t] = cat
    
    links["highway_category"] = links["highway"].map(highway_to_category).fillna("Unknown")
    return links

def assign_free_travel_times(df, links):
    """
    Assign free travel times per category (municipalityType, highway_category) to each trip.
    """
    # Explode links to merge with link data
    exploded = df[["links"]].explode("links").reset_index()
    merged = exploded.merge(links.reset_index(), left_on="links", right_on="link_id", how="left")

    # Group by trip index and category, sum free travel times
    grouped = merged.groupby(["index", "municipalityType", "highway_category"])["free_travel_time"].sum().reset_index()
    result = grouped.groupby("index").apply(
        lambda g: g.set_index(["municipalityType", "highway_category"])["free_travel_time"].to_dict()
    ).reindex(df.index)

    df["free_travel_time"] = result
    return df

def calculate_free_time_percentages(df):
    """
    Calculate the percentage of total free travel time for each category per trip.
    """
    df["free_time_percentage"] = df["free_travel_time"].map(
        lambda d: {k: v / (total := sum(d.values())) for k, v in d.items()} if (total := sum(d.values())) > 0 else {}
    )
    return df

def compute_weighted_average_errors(df):
    """
    Compute weighted average errors per category based on free time percentages.
    """
    df_temp = df[["error_min", "free_time_percentage"]].copy()
    df_temp["categories"] = df_temp["free_time_percentage"].apply(lambda d: list(d.keys()))
    df_temp = df_temp.explode("categories")
    df_temp["pct"] = df_temp.apply(lambda row: row["free_time_percentage"][row["categories"]], axis=1)
    df_temp["weighted_error"] = df_temp["error_min"] * df_temp["pct"]
    grouped = df_temp.groupby("categories").agg({"weighted_error": "sum", "pct": "sum"})
    averages = (grouped["weighted_error"] / grouped["pct"]).to_dict()

    # Convert for plotting
    df_averages = pd.DataFrame.from_dict(averages, orient='index', columns=['average_error_min'])
    df_averages.index = pd.MultiIndex.from_tuples(df_averages.index, names=['municipalityType', 'highway_category'])
    df_averages = df_averages.reset_index()

    return df_averages

def plot_error_heatmap(context, df_averages):
    """
    Plot a heatmap of average errors by municipality type and highway category.
    """
    pivot_table = df_averages.pivot(index='municipalityType', columns='highway_category', values='average_error_min')

    plt.figure(figsize=(12, 6))  # Increased width to accommodate labels
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Average Error (min) by Municipality Type and Highway Category (Just an Approximation)')
    plt.ylabel('Municipality Type')
    plt.xlabel('Highway Category')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to fit labels
    
    # get the path
    api = context.config("travel_times_from")    
    api = "tomtom" if api == "all" else api        
    out = os.path.join(context.config("output_path"),context.config("output_id"),context.config("simulation_directory"),"travel_times_"+api)    
    path = os.path.join(out, "average_error_heatmap.png")
    plt.savefig(path)
    plt.close()

def execute(context):
    """
    Main execution function: load data, process, compute averages, and plot.
    """
    # Load and merge data
    df = load_and_merge_data(context)

    # Prepare network links
    links = prepare_network_links(context)

    # Assign free travel times to trips
    df = assign_free_travel_times(df, links)

    # Calculate percentages
    df = calculate_free_time_percentages(df)

    # Compute weighted averages
    df_averages = compute_weighted_average_errors(df)

    # Plot heatmap
    plot_error_heatmap(context, df_averages)

    return df