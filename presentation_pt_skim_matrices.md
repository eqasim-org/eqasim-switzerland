# Skim matrices and public transport

## 1. Introduction

This document provides a concise overview of the recent modifications to the eqasim pipeline for computing public transport (PT) travel times and trip costs. The objective is to ensure that these computed durations and costs can be directly used in the utility functions defined in the mode choice module, as introduced in presentation_mode_choice.md.

The updated scripts are designed to:
- compute detailed PT trip costs and travel time components, and
- allow users to generate these data either through MATSim or independently, using pre-computed or newly created skim matrices.

These functionalities are distributed across several eqasim modules:
- `data/pt_pricing/`: Generates (or accesses) the two .csv files required for the detailed PT trip cost model.
- `pt_preparation/pt_routing/`: Provides parameters related to PT route choice preferences.
- `pt_preparation/pt_pricing/`: Computes the skim matrices containing PT travel times and costs.
- `mode_choice/`: Integrates the skim matrices into the mode choice model.


## 2. The PT cost model

This paragraph provides a brief overview of the detailed PT cost model. While SBB/CFF applies a distance-based algorithm to price booked trips, local PT operators rely on a zonal fare system. The purpose of this model is to capture this dual structure. Although the detailed pricing logic is implemented directly in Java (and is therefore outside the scope of this short presentation), the Python eqasim pipeline is responsible for generating the two main input files required by the cost model:
-`SBB_all_distances.csv`: the distances used internally by SBB/CFF do not necessarily correspond to real geographical distances. This file stores the SBB-defined distances between pairs of train stations.
-`gtfs_zones.csv`: this file maps each GTFS stop ID to the PT community(ies) and fare zone(s) it belongs to.

The entry point of this module is `data.pt_pricing.pt_pricing`. Depending on the configuration parameter `generate_pt_pricing_inputs`, it will either generate new versions of these two files or load the files already present in `[data_path]/pt_pricing/output`, where data_path refers to the directory containing all simulation input files. The zip archive shared by Dib on polybox (`CantonVaud/02_data/simulation_data.zip`) includes released versions of both files, ready to be copied to the appropriate location before running simulations.

The other scripts in the `data/pt_pricing` submodule extract and process information from PDF documents available on the AllianceSwissPass website (`t603` for SBB distances, `t651` for local PT zones) to produce updated versions of the two CSV files. More detailed information on the extraction and processing approach can be added here if needed.


## 3. PT route choice parameters

These parameters quantify how travellers perceive different components of a PT trip. They translate physical attributes of a PT route (in vehicle time, walk time, transfers) into a generalized cost value, which can be compared across alternative routes. These parameters include:
- in-vehicle time coefficients, differentiated by mode (train vs. tram+subway vs. bus+other), to perceive different comfort levels or perceived speed or delay likelihood
- walking and waiting time penalties
- transfer penalties, differentiated by transfer type (train <-> train, vs. train <-> any other mode, vs. any other mode <-> any other mode). 

The submodule `pt_preparation/pt_routing` offers the possibility to estimate those parameters using microcensus trips. Starting from initial parameter estimates, PT microcensus trips are routed using a MATSim script and the observed routes are compared to the reference ones. The CMA-ES algorithm is used to propose new parameter candidates at each iteration, until an equilibrium is found. The optimization process is however really long (12-24h). Consequently, it is only run if `calibrate_pt_routing_params` is set to True. By default, this parameters is set to False, which means that a set of already optimized parameters, directly encoded in `pt_preparation/pt_routing/pt_routing_parameters`, is provided to the next pipeline stages.


## 4. Skim matrices

The basis for the computation of the skim matrices is the USPAT zones data set, provided in Polybox in `CantonVaud/02_data/USPAT/statistische-grundeinheiten_stufe1_2025-01-01_2056.gpkg`. Prior to running simulations, this file must be copied and pasted into the folder `[data_path]/spatial/USPAT/`. A shapefile containing the Swiss lakes perimeters, provided in Polybox in `CantonVaud/02_data/Lakes/g1s18.shp`, is also required to only consider the zone parts not overlapping with the lakes. All files in the `CantonVaud/02_data/Lakes/` folder must be copied into `[data_path]/spatial/lakes/`.

The skim matrice generation process runs as follows:

#### 4.1 Zone and points generation
The USPAT geopackage file is read in the `pt_preparation/pt_pricing/uspat_zones.py` script. The zones belonging to the study area cantons are selected. To identify those cantons, a list of canton name abbreviations can be defined in the config file with the parameter `uspat_cantons`. For instance, to select the Canton de Vaud and all neighboring cantons, the config file must contain the following line: ``uspat_cantons: ["VD", "BE", "NE", "FR", "VS", "GE"]``. In order to accurately model the external traffic, i.e. the trips coming from cantons outside of the study area, the 20 remaining cantons are added to the USPAT zones as individual zones.

The `pt_preparation/pt_pricing/uspat_points.py` script defines points within the USPAT zones. To do so, the STATENT locations are matched with the zones and a few points per zone, their exact number being specified by the `number_points_per_uspat_zone` config parameter, are sampled based on the reported number of employees. This makes sure that the generated trips are starting and ending at accessible points and correspond to plausible trips.

#### 4.2 Trip routing, travel time component estimates and trip costs

The script `pt_preparation/pt_pricing/run_price_estimation_from_uspat.py` uses the points sampled previously to generate origin-destination requests, having the following attributes:
- origin and destination coordinates, obtained from the points data set. For a given origin point and destination zone, only one destination point is chosen. This ensures that the number of requests between two zones does not increase too fast when the `number_points_per_uspat_zone` increases
- departure time, sampled randomly between 6AM and 6PM
- home coordinates, considered equal to the origin coordinates. The home coordinates have a solely technical purpose and are not used in the computation
- age, set to 32 for all requests. Similar to the home coordinates, the age has to be included for technical reasons and is not considered in the computation
- PT subscription ownership also has to be provided to the next part of the code and is set to False for all requests
- Half-fare subscription ownership: given the wide ownership rate of Half-fare subscription, and the complex effect it has on the trip cost when local zones are crossed, we decided to duplicate the requests. The first duplicate corresponds to a traveller owning a half-fare subscription; the second one to a traveller without any PT subscription.

The script then creates a MATSim config taking into account the detailed PT price model and the route preferences described previously. The requests are sent into chunks into a MATSim process computing, for each request
- the in-vehicle, waiting, access and egress time
- the number of transfers
- the travelled distance
- the cost with or without half-fare subscription, according to the request specification.

A considerable amount of time (36-48 hours) is required to compute all travel times and costs when using 5 points per zone. The result of this stage is a huge `.csv` file containing the request specifications and the results.

#### 4.3 Skim matrices computation

Finally, the results of the previous step are aggregated in the `pt_preparation/pt_pricing/process_results_uspat.py` script. For each origin and destination zone pair, the following attributes are averaged and returned:
- total travel time
- in-vehicle time
- access-egress time
- waiting time
- number of transfers
- Half-fare price
- price without half-fare
- travelled distance


#### 4.4 Skim matrices in the mode choice module

As mentioned above, the skim matrices generation process is extremely time-consuming. Pre-computed matrices are thus provided in Polybox, in `CantonVaud/02_data/Skim matrices PT/vdgefrbenevs.csv`. The stage `mode_choice/trips/get_skim_matrices.py` imports the skim matrices in the mode choice module. It requests the config parameter `generate_skim_matrices` (default = False). If set to True, the process described above (4.1 to 4.3) will be run. If set to False, the csv file provided by the config parameter `skim_matrices_path` is used.

If the skim matrices have to be used, the origin and destination zones are assigned to each trip processed in `mode_choice/trips/prepare_trips.py`. When computing the trip cost (in `mode_choice/cost/pt_skim_matrices.py`) and travel time components (in `mode_choice/variables/pt_skim_matrices.py`), the corresponding entries are then copied from the matrices to the trip variable data set returned by the scripts.


#### 4.5 Computing PT travel time components and trip costs without skim matrices

The travel durations and costs returned by the skim matrices approach are only estimates of what one traveller would actually experience, as these variables depend on the exact origin and destination of the trip, on the traveller's individual characteristics and on their exact departure time. The skim matrices are thus intended to be used as a proxy when access to Java is not possible.

Though, when access to Java and MATSim is provided, it is possible to compute the exact travel time and cost. The process is exactly the same as what has been described in 4.2. In  `mode_choice/cost/pt_java.py` and `mode_choice/variables/pt_java.py`, the synthetic trips are transformed into requests that are sent to MATSim scripts computing the exact experienced travel time components and costs.

The config parameter `use_skim_matrices` determines whether skim matrices have to be used or not. If True, the skim matrices are used; else, the Java approach is employed. 


 

