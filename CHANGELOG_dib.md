# Changelog

## OSM Version Support
The pipeline now supports OSM files in `.pbf` format. Since `.pbf` files are not supported by PT2MATSim, a new package has been created in `data.osm` for file conversion from `.pbf` to `.osm`.

In the config file, you need to provide the `osm_file` parameter, which is the name of the file in `data_path/osm` (e.g., `osm_file: switzerland-latest.osm.pbf`).

## Merging OSM Files
If you want to merge multiple OSM files (e.g., to include regions in France), download the necessary files, then specify the `osm_file` as a list in the config file (e.g., `osm_file: [switzerland-latest.osm.pbf, rhone-alpes-latest.osm.pbf, franche-comte-latest.osm.pbf]`).

You also need to specify a border offset in meters (e.g., `border_offset: 3000`). This determines how far from the Swiss border the new merged network file will cover.

## Export Detailed Network
If you want to export the detailed geometry of each link in the network, set `export_detailed_network: true` in the config.

## Traffic Lights
If you want to model traffic lights, set `add_traffic_lights: true`. This will read all traffic lights from the provided OSM file, then only keep traffic lights matched to an intersection. The traffic lights will be added to the network file as link attributes.

## Network Cleaning
If you want to merge nodes that do not represent intersections, set `simplify_network_in_eqasim: true`.

This process will:
- Remove loops (where `from_node == to_node`)
- Remove duplicated links
- Remove nodes that have one incoming link and one outgoing link with no changes in link attributes (speed, capacity, lanes)
- Remove unconnected links (some links are not connected to the whole network)

## Elevations
If you want to have an elevation attribute for each node (z-coordinate), set `assign_elevations: true`.

**Note:** Only nodes located within Switzerland will have their elevations obtained from Swisstopo. Nodes outside of Switzerland will have `z == 0`.

## Correct Capacities
Small links create artificial congestion, so their capacity is doubled in matsim-contrib/osm. However, in PT2MATSim, their capacity is not increased. To address this, I created a formula to increase their capacity. To enable this feature, set `correct_links_capacity: true`.

## Parse Turn Restrictions
If you want to include turn restrictions in the network, set `parseTurnRestrictions: true`.

## BIOGEME DMC Model
**IMPORTANT:** You need to install Biogeme version 3.2.10 first.

Add a DMC stage with all its substages to estimate a DMC model. You can run the stage `dmc.model`, which will:
- Create the model
- Estimate the Value of Time (VoT) and their distributions (plot distribution figures)
- Write the estimated parameters to a YAML file in the same format as MATSim/eqasim

You can specify this file path in the MATSim config (`eqasim.modeParametersPath`).

### New Parameters for DMC Model
Add these parameters to the pipeline config:

- `routed_trips_file`: str (default: "Routed_alternatives_v3_FINAL.txt") - filename of the Google routed trips data. Should be located in data_path/dmc.
- `ignore_car_passenger`: bool (default: False) - Whether to consider car passenger as a mode in the model
- `distance_cost_interaction`: bool (default: True) - Whether to consider cost-distance interaction
- `income_cost_interaction`: bool (default: True) - Whether to consider cost-income interaction
- `car_cost_per_km`: float (default: 0.26) - CHF per km
- `parking_cost_per_hour_CHF_urban`: float (default: 1.0) - CHF per hour
- `parking_cost_per_hour_CHF_suburban`: float (default: 0.5) - CHF per hour
- `parking_price_reduction_for_work`: float (default: 1.0) - Fraction of parking price that work trips would pay
- `urban_parking_search_min`: float (default: 2.0) - Minutes
- `suburban_parking_search_min`: float (default: 1.0) - Minutes
- `only_from_home_trips`: bool (default: False)

## Automatic Calibration
This section explains how to perform mode choice estimation and calibration:
1. Set the `estimate_dmc` parameter to `true`. This will estimate the parameters and use them when running MATSim.  
2. Set either `calibrate_alphas_in_matsim` or `calibrate_betas_in_matsim` to `true` to adjust the estimated values within the simulation so they align with the actual mode shares. IMPORTANT: if you decide to calibrate the alphas, make sure you expect to have the mode shares that are set in matsim.simulation.utils.get_calibration_args (in calibrate alphas case).

3. After the simulation, the optimal parameters are location in the last iteration folder (as optimal_parameters.yml or alphas.csv)
4. if you want to calibrate the alpha for each canton, this can be done after this stage. In MATSim config, go to `eqasim:alphaCalibration`, and set `level=canton`, and `filePath=cantons_mode_shares.csv`, which is a csv file, containing canton name as first column (same canton names as in households), and the remaining columns contain the mode shares for the five modes. If you want to do this calibration at the regional level (cluster of cantons, each cluster have its alpha), you can similarly set `level=cluster` and provide the csv file of the mode shares within each cluster. 

## Activate Delays at Intersections
To activate delays at intersections in MATSim, follow these steps:
1. Enable traffic light intersection delays: set `activate_traffic_light_delays` in the config to `true`.  
2. Enable unsignalized intersection delays: set `activate_unsignalized_intersections_delays` in the config to `true`.  


## Other Changes

### Population Data
- `matsim.scenario.population`: Now stores `(ptHasGleis7, ptHasJunior)` attributes

### Household Data
- `matsim.scenario.households`: Now stores `(cantonName, incomePerCapita, municipalityType)` attributes

### Activity Data
- `matsim.scenario.activities`: Now stores `("municipality_type", "municipality_id")` as activity attributes (required changes in writers)

### PT2MATSim Runtime
- `matsim.runtime.pt2matsim`: The `pt2matsim_path` parameter can be specified in the config. If specified, this JAR will be used, similar to how it's done for `runtime.eqasim` (useful for avoiding cloning errors on Euler)

### Simulation Runtime
- `matsim.simulation.run`: Added `last_iteration` parameter to the config
  - For full simulation: set `last_iteration: 60`
  - For testing: set to a small number to check that the simulation runs without errors

### Output Management
- `matsim.output`: When running the simulation, results are automatically moved to the output directory