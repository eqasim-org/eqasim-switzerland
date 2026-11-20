from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter, GEH
from ..matching.results import save_count_results
from ..run_utils import manual_exclusion_mask
from ..cantons.ch_overrides import STATION_OSM_OVERRIDES
from pathlib import Path
import hashlib
import os
from ..paths import configure_simulation_path, get_analysis_output_path, matches_found
import geopandas as gpd
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.counts.cantons.ch")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.matching.compare")
    context.stage("data.spatial.swiss_border")

    context.config("input_downsampling")
    configure_simulation_path(context)
    context.config("only_weekday", default=True)

def execute(context):        
    # paths and parameters  
    count_stations_file, counts_data_file, year  = context.stage("analysis.counts.cantons.ch")
    city = "ch"
    sample_size = context.config("input_downsampling")
    minimum_months = 8
    include_incomplete_data = True
    path_to_images = get_analysis_output_path(context)
    os.makedirs(path_to_images, exist_ok=True)

    # Load the network and the counts
    network = context.stage("analysis.counts.matching.network")

    # load counts
    counts = Counts(counts_data_file, count_stations_file, 
                include_incomplete_data = include_incomplete_data, 
                minimum_months = minimum_months, context = context, year=year)
            
    # Match the data with the network
    matcher = TrafficDataMatcher()
    matched = matcher.match(network=network, 
                            counts=counts,
                            mode="bidirectional",
                            search_radius=80,
                            prioritize_road_types=True,
                            station_overrides=STATION_OSM_OVERRIDES)

    # Save resolved current link IDs and the OSM references used for overrides.

    audit_columns = [name for name in ("id", "measurement_group", "reference_osm_id", "link_id",
                     "distance", "angle_difference", "override_match_method") if name in matched]
    if "is_station_override" in matched:
        matched.loc[matched.is_station_override.fillna(False), audit_columns].to_csv(
            os.path.join(context.path(), "station_osm_overrides.csv"), index=False)

    if not matches_found(matched, city):
        return None

    # Compare the with simulation
    cmp     = context.stage("analysis.counts.matching.compare")
    flows   = cmp.compare_flow_total_efficient(counts, matched, network, 
                                            sample_size = sample_size, 
                                            get_average=False, 
                                            flow_col = 'flow_w' if context.config("only_weekday") else 'flow')
    # Reinstate only complete sensor footprints approved in the shared selection.

    manually_excluded = manual_exclusion_mask(flows, city="ch")
    low_volume = (flows.simulated_flow < 200 * 24) | (flows.flow < 200 * 24)
    flow_discrepancy = (abs(flows.flow - flows.simulated_flow) > 25000) | ~flows.pdiff.between(-70, 200)
    audit = flows.copy()
    audit["manual_exclusion"] = manually_excluded
    audit["low_volume"] = low_volume
    audit["flow_discrepancy"] = flow_discrepancy
    audit["kept_in_swiss_results"] = ~(manually_excluded | low_volume | flow_discrepancy)
    audit.to_csv(os.path.join(context.path(), "station_filter_audit.csv"), index=False)
    flows = flows[~manually_excluded].reset_index(drop=True)
    
    if not matches_found(flows, city, source="simulation flow results"):
        return None

    # Identify the stations that might be mismatched
    stations_to_drop = flows[(abs(flows.flow-flows.simulated_flow)>25000)|
                             (flows.simulated_flow< 200 * 24)|
                             (flows.flow< 200 * 24)|
                             (~flows.pdiff.between(-70,200))]["id"].unique()

    # geh = GEH(flows.flow.values, flows.simulated_flow.values, return_vector=True)
    # stations_to_drop = flows[(geh>50)]["id"].values
    # logger.info(f"Identified {len(stations_to_drop)} stations to be removed, out of {len(flows)} total stations. These stations will be dropped from the analysis.")

    # Plot the network and highligh these links in green  
    plotter = Plotter()
    roads_to_show = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link']
    matched_links = network.get_link_geometries(matched.link_id.tolist())

    # filter out stations to drop and save results
    all_flows = flows.copy()
    all_flows["dropped"] = all_flows["id"].isin(stations_to_drop)
    flows = flows[~flows['id'].isin(stations_to_drop)].reset_index(drop=True)
    matched = matched[~matched['id'].isin(stations_to_drop)].reset_index(drop=True)

    path_to_results = save_count_results(city, matched, flows, context.path())

    # Plot statistics
    plotter.plot_flow(flows = flows, 
                    counts = counts, 
                    distance_to_border = 3000, 
                    title = f"Observed vs Simulated Traffic Flows ({city})",
                    output_file = os.path.join(path_to_images, f"flow_comparaison_{city}.png"),
                    show_range = False,
                    show_geh = True)

    plotter.plot_flow(flows = flows, 
                    counts = counts, 
                    distance_to_border = 3000, 
                    title = f"Observed vs Simulated Traffic Flows ({city})",
                    output_file = os.path.join(path_to_images, f"flow_comparaison_without_border_{city}.png"),
                    show_range = False,
                    show_geh = False,
                    remove_near_border = True)
    
    plotter.plot_flow_by_road_type(flows, network, matched, counts,
                                distance_to_border = 0, 
                                title = f"Average Observed vs Simulated Flow by Highway Type ({city})",
                                output_file = os.path.join(path_to_images, f"flow_by_road_type_{city}.png"))
    
    
    # Plot the map in html  
    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))        
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                        matched_links.to_crs(epsg=4326)], 
                        data_to_show=["link_id"], 
                        point_gdf=[
                            Plotter.prepare_flow_map_points(
                                counts.counts, flows
                            ).to_crs(epsg=4326)
                        ],
                        point_data_to_show=Plotter.FLOW_MAP_TOOLTIP_FIELDS,
                        border = border,
                        path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}.html"))
    
    Plotter.create_map([network.get_ways(road_types = roads_to_show).to_crs(epsg=4326),
                    matched_links.to_crs(epsg=4326)], 
                    data_to_show=["link_id"], 
                    point_gdf=[
                        Plotter.prepare_flow_map_points(
                            counts.counts, all_flows
                        ).to_crs(epsg=4326)
                    ],
                    point_data_to_show=Plotter.FLOW_MAP_TOOLTIP_FIELDS,
                    border = border,
                    path_to_save= os.path.join(path_to_images, f"counts_on_network_{city}_unfiltered.html"))
    
    return path_to_results


def validate(context):
    root = Path(__file__).parent.parent
    digest = hashlib.sha256()
    for name in ("cantons/ch_overrides.py", "matching/station_overrides.py",
                 "matching/matcher.py", "matching/road_matching.py",
                 "matching/compare.py", "matching/plots.py", "matching/results.py", "run_utils.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()
