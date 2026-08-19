from lxml import etree
import os.path
import pandas as pd
import gzip


def add_SBBPT_module(context):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    # Parse XML and preserve DOCTYPE and comments
    parser = etree.XMLParser(remove_blank_text=True)
    tree   = etree.parse(config_path, parser)
    root   = tree.getroot()

    # Create new module
    module = etree.Element("module", name="SBBTransit")

    etree.SubElement(module, "param", name="deterministicServiceModes",
                    value="rail,subway,ferry,tram,funicular,cable-car,gondola,other")
    etree.SubElement(module, "param", name="createLinkEventsInterval", value="10")

    # Append to root
    root.append(module)

    # Write back to file with DOCTYPE
    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


def add_ptZones_module(context):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    # Parse XML and preserve DOCTYPE and comments
    parser = etree.XMLParser(remove_blank_text=True)
    tree   = etree.parse(config_path, parser)
    root   = tree.getroot()

    # Create new module
    module = etree.Element("module", name="ptZones")

    etree.SubElement(module, "param", name="ptZonesFilePath",  value=f"{context.path()}/gtfs_zones.csv")
    etree.SubElement(module, "param", name="sbbDistancesPath", value=f"{context.path()}/SBB_all_distances.csv")
    etree.SubElement(module, "param", name="pricingDescriptionPath", value=f"{context.path()}/pricingDescription.xml")

    # Append to root
    root.append(module)

    # Write back to file with DOCTYPE
    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


def adjust_pt_routing_parameters(context, parameters):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    pt_modes = ["bus", "rail", "subway", "ferry", "tram", "funicular", "cable-car", "gondola", "other"]
    penalty_intermodal = parameters["raptorPenalties:transfer_intermodal"]
    penalty_rail       = parameters["raptorPenalties:transfer_rail"]
    penalty_other      = parameters["raptorPenalties:transfer_other"]

    # Parse XML and preserve DOCTYPE and comments
    parser = etree.XMLParser(remove_blank_text=True)
    tree   = etree.parse(config_path, parser)
    root   = tree.getroot()

    for module in root.findall(".//module"):
        # Update values of time in eqasim:raptor
        if module.get("name") == "eqasim:raptor":
            for param in module.findall("param"):
                param_name = param.get("name")
                if param_name in parameters.keys():
                    param.set("value", str(round(parameters.get(param_name), 3)))

        # Add mode to mode transfer penalties in swissRailRaptor
        if module.get("name") == "swissRailRaptor":
            for mode1 in pt_modes:
                for mode2 in pt_modes:
                    if mode1 == "rail" and mode2 == "rail":
                        value = penalty_rail
                    elif mode1 == "rail" or mode2 == "rail":
                        value = penalty_intermodal
                    else:
                        value = penalty_other
                    parameterset = etree.SubElement(module, "parameterset", type="modeToModeTransferPenalty")
                    etree.SubElement(parameterset, "param", name="fromMode", value = mode1)
                    etree.SubElement(parameterset, "param", name="toMode", value = mode2)
                    etree.SubElement(parameterset, "param", name="transferPenalty", value = str(round(value, 3)))

    # Write back to file with DOCTYPE
    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


def change_param(context, module_name, param_name, new_value):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    parser = etree.XMLParser(remove_blank_text=True)
    tree   = etree.parse(config_path, parser)
    root   = tree.getroot()

    for module in root.findall(".//module"):
        if module.get("name") == module_name:
            for param in module.findall("param"):
                if param_name == param.get("name"):
                    param.set("value", new_value)

    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


def modify_PCEs(transit_vehicles_input_path, transit_vehicles_output_path, scaling_factor):
    assert os.path.exists(transit_vehicles_input_path)

    parser = etree.XMLParser(remove_blank_text = True)
    tree   = etree.parse(transit_vehicles_input_path, parser)
    root   = tree.getroot()

    ns_uri = "{http://www.matsim.org/files/dtd}"

    for vt in root.findall(f".//{ns_uri}vehicleType"):        
        nm = vt.find(f"{ns_uri}networkMode")
        if nm is not None:
            
            pce_elem = vt.find(f"{ns_uri}passengerCarEquivalents")
            current_pce = float(pce_elem.get("pce", "1.0")) if pce_elem is not None else 1.0
            new_pce = round(scaling_factor * current_pce, 4)
            
            if pce_elem is None:
                pce_elem = etree.SubElement(vt, f"{ns_uri}passengerCarEquivalents")
            
            pce_elem.set("pce", str(new_pce))

    xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    with gzip.open(transit_vehicles_output_path, "wb") as f:
        f.write(xml_bytes)


def get_mode_shares_calibration_args(context):
    additional_args = []
    if context.config("calibrate_betas_in_matsim"):
        cache_path = context.working_directory
        bounds = """car.alpha_u:5.0,
                    walk.alpha_u:5.0,
                    bike.alpha_u:5.0,
                    cp.alpha_u:5.0,
                    cp.travelTimeExponent:0.3,
                    walk.travelTimeExponent:0.3,
                    car.betaTravelTime_u_min:0.3"""
                
        additional_args.extend(["--config:eqasim:calibration.activate", "true",
                                "--config:eqasim:calibration.runCalibration", "true",
                                "--config:eqasim:calibration.optimizer", "cmaes",
                                "--config:eqasim:calibration.metric", "mse",
                                "--config:eqasim:calibration.betaMomentum", "0.4",
                                "--config:eqasim:calibration.eqasimCachePath", cache_path,
                                "--config:eqasim:calibration.bounds", bounds,
                                "--config:eqasim:calibration.distanceBins", "0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000",
                                "--config:eqasim:calibration.maxEval", "4000",
                                "--config:eqasim:calibration.repoCommit", "7b1bf4d4c8c1fddc9d3479a96513d5c5d3268294",
                                "--config:eqasim:calibration.optimizerPath", "optimizer",
                                "--config:eqasim:termination.threshold", "0.0000001"
                                ])
    else:
        if context.config("calibrate_alphas_in_matsim"):
            level = context.config("alphaCalibration.level")
            global_shares_output_path, cantonal_shares_output_path = context.stage("data.microcensus.shares")
            
            shares = pd.read_csv(global_shares_output_path).set_index("mode")

            additional_args.extend([
                "--config:eqasim:alphaCalibration.activate", "true",
                "--config:eqasim:alphaCalibration.beta", "0.2",
                "--config:eqasim:alphaCalibration.calibratedModes", "car,pt,walk,bike,car_passenger",
                "--config:eqasim:alphaCalibration.carModeShare", str(shares.loc["car","mode_share"]),
                "--config:eqasim:alphaCalibration.ptModeShare", str(shares.loc["pt","mode_share"]),
                "--config:eqasim:alphaCalibration.walkModeShare", str(shares.loc["walk","mode_share"]),
                "--config:eqasim:alphaCalibration.bikeModeShare", str(shares.loc["bike","mode_share"]),
                "--config:eqasim:alphaCalibration.carPassengerModeShare", str(shares.loc["car_passenger","mode_share"]),
                "--config:eqasim:alphaCalibration.level", level,
                "--config:eqasim:alphaCalibration.filePath", cantonal_shares_output_path,
                "--config:eqasim:termination.threshold", "0.0000001"
            ])

    return additional_args


def get_delays_args(context):
    activate_tl_delays = context.config("activate_traffic_light_delays")
    activate_unsignalized_delays = context.config("activate_unsignalized_intersections_delays")
    
    additional_args = ["--config:eqasim:flow.writeFlowInterval", "1000",
                       "--config:eqasim:intersectionDelays.writeDelayInterval", "1000"] #it takes too much space
    if activate_tl_delays or activate_unsignalized_delays:
        additional_args.extend([
            "--config:eqasim:intersectionDelays.activate", "true",
            "--config:eqasim:intersectionDelays.activateTl", str(activate_tl_delays).lower(),
            "--config:eqasim:intersectionDelays.activateUnsignalized", str(activate_unsignalized_delays).lower()
        ])
    return additional_args  


def get_network_calibration_args(context):
    calibrate_network = context.config("network_calibration.activate")
    calibrate_counts = context.config("network_calibration.calibrate_disutilities")
    calibrate_freespeed = context.config("network_calibration.calibrate_freespeed")
    
    additional_args = []    
    if calibrate_network:
        assert (calibrate_counts or calibrate_freespeed), "Network calibration is activated, one of disutilities calibration or freespeed calibration need to be activated"
    
    additional_args.extend(
            ["--config:eqasim:networkCalibration.activate", "true",
            "--config:eqasim:networkCalibration.calibrate", str(calibrate_network).lower(),
            "--config:eqasim:networkCalibration.correctCapacities", str(context.config("correct_links_capacity")).lower(),
            "--config:eqasim:networkCalibration.minSpeed", str(context.config("minimum_speed"))]
    )
    
    objective = []
    if calibrate_counts:  
        objective.append("penalty")
        if context.config("network_calibration.calibrate_agents_ascs"):
            objective.append("agent")
        if context.config("network_calibration.calibrate_crossborder_population"):        
            objective.append("subpopulations")
             
    if calibrate_freespeed:
        objective.append("freespeed")
        
    additional_args.extend([
        "--config:eqasim:networkCalibration.objective", ",".join(objective)
        ])
       
    return additional_args

def network_calibration_files_paths(context):
    calibrate_network = context.config("network_calibration.activate")
    calibrate_counts = context.config("network_calibration.calibrate_disutilities")
    calibrate_freespeed = context.config("network_calibration.calibrate_freespeed")

    args = []
    if calibrate_network and calibrate_counts:
        calibration_counts_file = context.stage("analysis.counts.target")
        calibration_regions = context.stage("calibration.road_regions.penalty_calibration")
        args.extend([
            "--countsFile", calibration_counts_file,
            "--countSpecialRegionPath", calibration_regions
        ])

    if calibrate_network and calibrate_freespeed:
        calibration_travel_times = context.stage("analysis.travel_times.APIs.target")
        calibration_freespeed = context.stage("calibration.road_regions.freespeed_calibration")
        args.extend([
            "--speedsFile", calibration_travel_times,
            "--speedsSpecialRegionPath", calibration_freespeed
        ])

    return args




def get_dmc_parameters_args(context):
    mode_parameters_path = "%s/dmc_parameters.yml" % context.path("matsim.simulation.prepare")
    cost_parameters_path = "%s/cost_parameters.yml" % context.path("matsim.simulation.prepare")
    additional_args = []
    additional_args.extend(["--config:eqasim.costParametersPath", cost_parameters_path])
    additional_args.extend(["--config:eqasim.modeParametersPath", mode_parameters_path])
    return additional_args

