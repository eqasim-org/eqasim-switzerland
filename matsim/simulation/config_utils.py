from lxml import etree
import os.path
import pandas as pd


def add_SBBPT_module(context):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    # Parse XML and preserve DOCTYPE and comments
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(config_path, parser)
    root = tree.getroot()

    # Create new module
    module = etree.Element("module", name="SBBPt")

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
    tree = etree.parse(config_path, parser)
    root = tree.getroot()

    # Create new module
    module = etree.Element("module", name="ptZones")

    etree.SubElement(module, "param", name="ptZonesFilePath",  value=f"{context.path()}/gtfs_zones.csv")
    etree.SubElement(module, "param", name="sbbDistancesPath", value=f"{context.path()}/SBB_all_distances.csv")

    # Append to root
    root.append(module)

    # Write back to file with DOCTYPE
    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)


def adjust_pt_routing_parameters(context, parameters):
    config_path = f"{context.path()}/{context.config('output_prefix')}config.xml"
    assert os.path.exists(config_path)

    pt_modes = ["rail", "subway", "ferry", "tram", "funicular", "cable-car", "gondola", "other"]
    penalty_intermodal = parameters["raptorPenalties:transfer_intermodal"]
    penalty_rail       = parameters["raptorPenalties:transfer_rail"]
    penalty_other      = parameters["raptorPenalties:transfer_other"]

    # Parse XML and preserve DOCTYPE and comments
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(config_path, parser)
    root = tree.getroot()

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
    tree = etree.parse(config_path, parser)
    root = tree.getroot()

    for module in root.findall(".//module"):
        if module.get("name") == module_name:
            for param in module.findall("param"):
                if param_name == param.get("name"):
                    param.set("value", new_value)

    doctype_str = '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">'
    tree.write(config_path, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype_str)



def get_calibration_args(context):
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
                                "--config:eqasim:calibration.repoCommit", "b72fae6c3860169f5a837d4648dbf4b2dc7ac3a0",
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
    activate_unsignalized_delaus = context.config("activate_unsignalized_intersections_delays")
    additional_args = []
    if activate_tl_delays or activate_unsignalized_delaus:
        additional_args.extend([
            "--config:eqasim:intersectionDelays.activate", "true",
            "--config:eqasim:intersectionDelays.activateTl", str(activate_tl_delays).lower(),
            "--config:eqasim:intersectionDelays.activateUnsignalized", str(activate_unsignalized_delaus).lower()
        ])
    return additional_args  