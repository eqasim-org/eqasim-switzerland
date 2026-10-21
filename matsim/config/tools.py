"""Shared XML editing and MATSim configuration helpers."""
from lxml import etree
import re
import glob
from numbers import Integral, Real
from pathlib import Path
from urllib.parse import urljoin
import shutil
import os.path
import pandas as pd
import gzip


TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off", ""})


def parse_boolean(value, option_name="value"):
    """Parse booleans without treating non-empty strings such as ``"no"`` as true."""
    if isinstance(value, bool):
        return value

    if isinstance(value, Integral):
        if value in (0, 1):
            return bool(value)

    if isinstance(value, Real):
        if value in (0.0, 1.0):
            return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False

    raise ValueError(f"{option_name} must be a boolean value, got {value!r}")


def java_boolean(value, option_name="value"):
    return "true" if parse_boolean(value, option_name) else "false"



def read_config(path):
    return etree.parse(str(path), etree.XMLParser(load_dtd=False, no_network=True,
                                                 resolve_entities=False))


def write_config(tree, path):
    _remove_duplicate_config_elements(tree.getroot())
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True,
               doctype=tree.docinfo.doctype)


def get_module(tree, name):
    root = tree.getroot() if hasattr(tree, "getroot") else tree
    module = root.find(f"module[@name='{name}']")
    return module if module is not None else etree.SubElement(root, "module", name=name)


def get_param(name, parent):
    return parent.find(f"param[@name='{name}']")


def set_param(name, value, parent):
    parameter = get_param(name, parent)
    if parameter is None:
        parameter = etree.SubElement(parent, "param", name=name)
    parameter.set("value", _xml_value(value))
    return parameter


def get_parameter_set(parent, kind, match=None):
    """Upsert a parameter set by its identifying fields, retaining other sets."""
    match = match or {}
    matches = [item for item in parent.findall(f"parameterset[@type='{kind}']")
               if all((p := get_param(key, item)) is not None and p.get("value") == _xml_value(value)
                      for key, value in match.items())]
    item = matches[0] if matches else etree.SubElement(parent, "parameterset", type=kind)
    for duplicate in matches[1:]:
        parent.remove(duplicate)
    for key, value in match.items():
        set_param(key, value, item)
    return item


def copy_config(source, target, params=None, input_paths=None):
    """Copy a config, optionally resolving its inputs before moving it."""
    if input_paths:
        tree = read_config(source)
        for module_path, names in input_paths.items():
            parts = module_path.split(".")
            selector = f"module[@name='{parts[0]}']"
            selector += "".join(f"/parameterset[@type='{part}']" for part in parts[1:])
            for param in tree.findall(selector + "/param"):
                value = param.get("value", "")
                if param.get("name") in names and value and value != "null":
                    param.set("value", ";".join(urljoin(Path(source).resolve().as_uri(), item.strip())
                                               for item in value.split(";") if item.strip()))
        write_config(tree, target)
    else:
        shutil.copyfile(source, target)
    if params:
        change_params(target, params, create=True)
    return str(Path(target).resolve())


def add_SBBPT_module(config_path):
    change_params(config_path, {
        "SBBTransit.deterministicServiceModes": "rail,subway,ferry,tram,funicular,cable-car,gondola,other",
        "SBBTransit.createLinkEventsInterval": 10,
    }, create=True)


def add_ptZones_module(config_path, directory=None):
    directory = Path(directory) if directory is not None else Path()
    change_params(config_path, {
        "ptZones.ptZonesFilePath": str(directory / "gtfs_zones.csv"),
        "ptZones.sbbDistancesPath": str(directory / "SBB_all_distances.csv"),
        "ptZones.pricingDescriptionPath": str(directory / "pricingDescription.xml"),
    }, create=True)


def adjust_pt_routing_parameters(config_path, parameters):
    tree = read_config(config_path)
    raptor = get_module(tree, "eqasim:raptor")
    for name, value in parameters.items():
        if not name.startswith("raptorPenalties:"):
            set_param(name, round(value, 3), raptor)
    swiss_raptor = get_module(tree, "swissRailRaptor")
    modes = ["bus", "rail", "subway", "ferry", "tram", "funicular", "cable-car", "gondola", "other"]
    for origin in modes:
        for destination in modes:
            kind = "rail" if origin == destination == "rail" else "intermodal" if "rail" in (origin, destination) else "other"
            item = get_parameter_set(swiss_raptor, "modeToModeTransferPenalty",
                                     {"fromMode": origin, "toMode": destination})
            set_param("transferPenalty", round(parameters[f"raptorPenalties:transfer_{kind}"], 3), item)
    write_config(tree, config_path)


def disable_population_inputs(config_path):
    change_params(config_path, {
        "facilities.inputFacilitiesFile": None, "households.inputFile": None,
        "plans.inputPlansFile": None, "vehicles.vehiclesFile": None,
        "DiscreteModeChoice.modeAvailability": "SwissModeAvailability",
    }, create=True)
    # Routing-only consumers have no freight population.
    tree = read_config(config_path)
    for module in tree.findall("module[@name='replanning']"):
        for item in list(module):
            if (p := get_param("subpopulation", item)) is not None and p.get("value") == "freight":
                module.remove(item)
    write_config(tree, config_path)


_PATH_SEGMENT = re.compile(r"^(?P<name>.+?)(?:\[(?P<index>\d+)\])?$")
_DEDUPLICATED_TAGS = {"module", "parameterset", "param"}

############### changing the config #####################
def change_params(config_path, params, output_path=None, create=False):
    """Change parameters in a MATSim XML configuration file.

    ``params`` is an iterable of ``(path, value)`` pairs (a dictionary is also
    accepted). Paths are dot-separated and may contain any number of nested
    modules or parameter sets, for example::

        controller.lastIteration
        DiscreteModeChoice.selector:MultinomialLogit.maximumUtility
        parentModule.childModule.someParameterSet.parameterName

    A path component matches either ``<module name="...">`` or
    ``<parameterset type="...">``. If several sibling containers have the
    same name/type, all are changed by default. Append a zero-based index to
    select one of them, e.g. ``transitRouter.penalty[2].transferPenalty``.

    Values are converted to strings. Booleans and ``None`` use MATSim's usual
    XML spellings: ``true``/``false`` and ``null``.

    Before saving, semantically identical sibling modules, parameter sets, and
    parameters are collapsed to their first occurrence. Parameter sets of the
    same type that contain different values are preserved.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"MATSim config file does not exist: {config_path}")

    if output_path is None:
        output_path = config_path

    tree = read_config(config_path)
    changes = params.items() if hasattr(params, "items") else params

    for param in changes:
        change_param(tree, param, create=create)

    write_config(tree, output_path)


def change_param(tree, param, create=False):
    """Apply one ``(path, value)`` parameter change to a tree or element."""
    try:
        param_path, param_value = param
    except (TypeError, ValueError) as error:
        raise ValueError("A parameter change must be a (path, value) pair") from error

    if not isinstance(param_path, str):
        raise TypeError("The parameter path must be a string")

    path = param_path.split(".")
    if len(path) < 2 or any(not component for component in path):
        raise ValueError(
            "The parameter path must contain at least a module and parameter "
            "name, separated by dots"
        )

    root = tree.getroot() if hasattr(tree, "getroot") else tree
    containers = [root]

    for depth, component in enumerate(path[:-1]):
        name, index = _parse_path_component(component, param_path)
        next_containers = []

        for container in containers:
            matches = [
                child for child in container if _container_name(child) == name
            ]

            if not matches and create and index in (None, 0):
                matches = [etree.SubElement(container, "module" if depth == 0 else "parameterset",
                                           **{"name" if depth == 0 else "type": name})]
            if index is None:
                next_containers.extend(matches)
            elif index < len(matches):
                next_containers.append(matches[index])

        if not next_containers:
            raise ValueError(
                f"Could not find module or parameter set {component!r} "
                f"while resolving parameter path {param_path!r}"
            )

        containers = next_containers

    param_name, param_index = _parse_path_component(path[-1], param_path)
    matching_params = []
    for container in containers:
        matches = [
            child
            for child in container
            if _local_name(child) == "param" and child.get("name") == param_name
        ]

        if not matches and create and param_index in (None, 0):
            matches = [etree.SubElement(container, "param", name=param_name)]
        if param_index is None:
            matching_params.extend(matches)
        elif param_index < len(matches):
            matching_params.append(matches[param_index])

    if not matching_params:
        raise ValueError(f"Could not find parameter {param_path!r}")

    value = _xml_value(param_value)
    for element in matching_params:
        element.set("value", value)

    return len(matching_params)


def _parse_path_component(component, full_path):
    match = _PATH_SEGMENT.fullmatch(component)
    if match is None:
        raise ValueError(
            f"Invalid component {component!r} in parameter path {full_path!r}"
        )

    index = match.group("index")
    return match.group("name"), None if index is None else int(index)


def _container_name(element):
    tag = _local_name(element)
    if tag == "module":
        return element.get("name")
    if tag == "parameterset":
        return element.get("type")
    return None


def _local_name(element):
    """Return an element's local tag name, also for namespaced XML."""
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def _xml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _remove_duplicate_config_elements(parent):
    """Recursively remove equivalent sibling MATSim configuration elements."""
    removed = 0

    # Work from the leaves upwards. Removing duplicates inside a container may
    # reveal that the complete container is itself a duplicate of a sibling.
    for child in parent:
        if isinstance(child.tag, str):
            removed += _remove_duplicate_config_elements(child)

    seen = set()
    for child in list(parent):
        if _local_name(child) not in _DEDUPLICATED_TAGS:
            continue

        signature = _semantic_element_signature(child)
        if signature in seen:
            parent.remove(child)
            removed += 1
        else:
            seen.add(signature)

    return removed


def _semantic_element_signature(element):
    """Return a hashable XML signature that ignores comments and indentation."""
    text = element.text
    if text is not None and not text.strip():
        text = None

    children = tuple(
        _semantic_element_signature(child)
        for child in element
        if isinstance(child.tag, str)
    )

    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        children,
    )



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


def mode_shares_calibration_parameters(context):
    if parse_boolean(context.config("calibrate_betas_in_matsim"), "calibrate_betas_in_matsim"):
        return {
            "eqasim:calibration.activate": True,
            "eqasim:calibration.runCalibration": True,
            "eqasim:calibration.optimizer": "cmaes",
            "eqasim:calibration.metric": "mse",
            "eqasim:calibration.betaMomentum": 0.4,
            "eqasim:calibration.eqasimCachePath": context.working_directory,
            "eqasim:calibration.bounds": "car.alpha_u:5.0,walk.alpha_u:5.0,bike.alpha_u:5.0,cp.alpha_u:5.0,cp.travelTimeExponent:0.3,walk.travelTimeExponent:0.3,car.betaTravelTime_u_min:0.3",
            "eqasim:calibration.distanceBins": "0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000",
            "eqasim:calibration.maxEval": 4000,
            "eqasim:calibration.repoCommit": "7b1bf4d4c8c1fddc9d3479a96513d5c5d3268294",
            "eqasim:calibration.optimizerPath": "optimizer",
            "eqasim:termination.threshold": 0.0000001,
        }
    if not parse_boolean(context.config("calibrate_alphas_in_matsim"), "calibrate_alphas_in_matsim"):
        return {}
    global_shares, cantonal_shares = context.stage("data.microcensus.shares")
    shares = pd.read_csv(global_shares).set_index("mode")
    params = {
        "eqasim:alphaCalibration.activate": True,
        "eqasim:alphaCalibration.beta": 0.2,
        "eqasim:alphaCalibration.calibratedModes": "car,pt,walk,bike,car_passenger",
        "eqasim:alphaCalibration.level": context.config("alphaCalibration.level"),
        "eqasim:alphaCalibration.filePath": cantonal_shares,
        "eqasim:termination.threshold": 0.0000001,
    }
    for mode, name in (("car", "car"), ("pt", "pt"), ("walk", "walk"),
                       ("bike", "bike"), ("car_passenger", "carPassenger")):
        params[f"eqasim:alphaCalibration.{name}ModeShare"] = shares.loc[mode, "mode_share"]
    return params


def delay_parameters(context):
    traffic_lights = parse_boolean(context.config("activate_traffic_light_delays"), "activate_traffic_light_delays")
    unsignalized = parse_boolean(context.config("activate_unsignalized_intersections_delays"),
                                "activate_unsignalized_intersections_delays")
    params = {"eqasim:flow.writeFlowInterval": 1000,
              "eqasim:intersectionDelays.writeDelayInterval": 1000}
    if traffic_lights or unsignalized:
        params.update({"eqasim:intersectionDelays.activate": True,
                       "eqasim:intersectionDelays.activateTl": traffic_lights,
                       "eqasim:intersectionDelays.activateUnsignalized": unsignalized})
    return params


def get_network_cal_config_as_bool(context):
    calibrate_network = parse_boolean(
            context.config("network_calibration.activate"), "network_calibration.activate")
    calibrate_disutilities = parse_boolean(
        context.config("network_calibration.calibrate_disutilities"),
        "network_calibration.calibrate_disutilities",
    )
    calibrate_freespeed = parse_boolean(
        context.config("network_calibration.calibrate_freespeed"),
        "network_calibration.calibrate_freespeed",
    )
    calibrate_agent_acs = parse_boolean(
        context.config("network_calibration.calibrate_agents_ascs"),
        "network_calibration.calibrate_agents_ascs",
    )
    calibrate_subpopulations = parse_boolean(
            context.config("network_calibration.calibrate_subpopulations"),
            "network_calibration.calibrate_subpopulations",
        )
    return calibrate_network, calibrate_disutilities, calibrate_freespeed, calibrate_agent_acs, calibrate_subpopulations


def get_network_calibration_objectives(context):
    calibrate_network, calibrate_disutilities, calibrate_freespeed, calibrate_agent_acs, calibrate_subpopulations = get_network_cal_config_as_bool(context)
    objective = ["freespeed"] # freespeed is always used
    if calibrate_network and calibrate_disutilities: objective.append("penalty")
    if calibrate_network and calibrate_agent_acs: objective.append("agent")
    if calibrate_network and calibrate_subpopulations: objective.append("subpopulations")
    return ",".join(objective)

def get_network_calibration_calibrate(context, ignore=None):
    calibrate_network, calibrate_disutilities, calibrate_freespeed, calibrate_agent_acs, calibrate_subpopulations = get_network_cal_config_as_bool(context)
    calibrate = []
    ignore = set(ignore) if ignore is not None else set()
    if calibrate_network and calibrate_disutilities and "penalty" not in ignore: calibrate.append("penalty")
    if calibrate_network and calibrate_freespeed and "freespeed" not in ignore: calibrate.append("freespeed")
    if calibrate_network and calibrate_agent_acs and "agent" not in ignore: calibrate.append("agent")
    if calibrate_network and calibrate_subpopulations and "subpopulations" not in ignore: calibrate.append("subpopulations")
    return ",".join(calibrate)


def network_calibration_settings(context):
    active, disutilities, freespeed, agent_ascs, subpopulations = get_network_cal_config_as_bool(context)
    if active and not (disutilities or freespeed or agent_ascs or subpopulations):
        raise ValueError("Network calibration is active but no calibration component is enabled")
    # Keep the module enabled to apply speed/penalty factors even without optimization.
    return {
        "eqasim:networkCalibration.activate": True,
        "eqasim:networkCalibration.correctCapacities": parse_boolean(context.config("correct_links_capacity")),
        "eqasim:networkCalibration.minSpeed": context.config("minimum_speed"),
        "eqasim:networkCalibration.objective": get_network_calibration_objectives(context),
        "eqasim:networkCalibration.calibrate": get_network_calibration_calibrate(context),
    }


def need_counts_file(context):
    calibrate_network, calibrate_disutilities, calibrate_freespeed, calibrate_agent_acs, calibrate_subpopulations = get_network_cal_config_as_bool(context)
    return calibrate_network and (calibrate_disutilities or calibrate_agent_acs or calibrate_subpopulations)

def network_calibration_parameters(context):
    active, disutilities, freespeed, _, _ = get_network_cal_config_as_bool(context)
    params = {}
    if active and need_counts_file(context):
        params["eqasim:networkCalibration.countsFile"] = context.stage("analysis.counts.target")
        if disutilities:
            params["eqasim:networkCalibration.costCalibration.specialRegionPath"] = context.stage("calibration.road_regions.penalty_calibration")
    if active and freespeed:
        params["eqasim:networkCalibration.freespeedCalibration.observedTripsFile"] = context.stage("analysis.travel_times.APIs.target")
        params["eqasim:networkCalibration.freespeedCalibration.specialRegionPath"] = context.stage("calibration.road_regions.freespeed_calibration")
    return params


def simulation_parameters(context):
    """Settings formerly supplied as overrides by matsim.simulation.run."""
    vdf = parse_boolean(context.config("use_vdf"), "use_vdf")
    last_iteration = context.config("last_iteration")
    interval = max(int(last_iteration / 2), 1)
    directory = Path(context.path())
    params = {
        "eqasim.costParametersPath": str(directory / "cost_parameters.yml"),
        "eqasim.modeParametersPath": str(directory / "dmc_parameters.yml"),
        "controller.outputDirectory": "simulation_output",
        "controller.lastIteration": last_iteration,
        "controller.writeEventsInterval": interval,
        "controller.writePlansInterval": interval,
        "qsim.numberOfThreads": min(context.config("threads"), 16 if vdf else 12),
        "linkStats.writeLinkStatsInterval": last_iteration if vdf else interval,
        "linkStats.averageLinkStatsOverIterations": 8,
        "controller.writeTripsInterval": last_iteration if vdf else interval,
        # The VDF Java runner always uses schedule-based public transport.
        "eqasim.useScheduleBasedTransport": vdf or parse_boolean(context.config("useScheduleBasedTransport")),
    }
    if not vdf:
        params["scoring.writeExperiencedPlans"] = parse_boolean(context.config("writeexperiencedplans"))
    params.update(mode_shares_calibration_parameters(context))
    params.update(delay_parameters(context))
    params.update(network_calibration_settings(context))
    params.update(network_calibration_parameters(context))
    return params


def get_regions_path(path,kind="freespeed", subfolder = "network_calibration_files"):
    regions = []
    if kind=="freespeed":
        region_dir = os.path.join(path, subfolder)
        if os.path.exists(region_dir):
            regions = glob.glob(f"{region_dir}/freespeed_special_region_*.yml")
    elif kind=="penalty":
        region_dir = os.path.join(path, subfolder)
        if os.path.exists(region_dir):
            regions = glob.glob(f"{region_dir}/penalties_special_region_*.yml")
    
    if len(regions)==0:
        return ""
    
    # Keep regions in numeric suffix order (e.g. _2.yml before _10.yml).
    regions = sorted(
        regions,
        key=lambda region: int(os.path.splitext(region)[0].rsplit("_", 1)[-1]),
    )
    # only keep the region_dir/region.yml part, not the full path
    regions = [os.path.join(subfolder, os.path.basename(region)) for region in regions]
    return ";".join(regions)


def adapt_existing_scenario(config_path, sample_size, prefix, use_freight,*,storageCapExponent=1.0):
    """Apply local sampling/input changes to an already adapted external scenario."""
    change_params(config_path, {
        "qsim.flowCapacityFactor": sample_size,
        "qsim.storageCapacityFactor": sample_size ** storageCapExponent,
        "network.inputNetworkFile": prefix + "network.xml.gz",
        "plans.inputPlansFile": prefix + "population.xml.gz",
        "facilities.inputFacilitiesFile": prefix + "facilities.xml.gz",
        "households.inputFile": prefix + "households.xml.gz",
        "transit.transitScheduleFile": prefix + "transit_schedule.xml.gz",
        "transit.vehiclesFile": prefix + "transit_vehicles.xml.gz",
        "vehicles.vehiclesFile": prefix + "vehicles.xml.gz",
    }, create=True)
    tree = read_config(config_path)
    replanning = get_module(tree, "replanning")
    for item in list(replanning):
        if (p := get_param("subpopulation", item)) is not None and p.get("value") == "freight":
            replanning.remove(item)
    groups = ["crossborder", "null"]
    if parse_boolean(use_freight):
        groups.append("freight")
    dmc_weight = sum(float(p.get("value"))
                     for item in replanning.findall("parameterset")
                     if (name := get_param("strategyName", item)) is not None and name.get("value") == "DiscreteModeChoice"
                     if (p := get_param("weight", item)) is not None)
    for group in groups:
        for strategy, weight in (("ReRoute", 0.05),
                                 ("KeepLastSelected", 0.95 - (dmc_weight if group == "null" else 0))):
            item = get_parameter_set(replanning, "strategysettings",
                                     {"subpopulation": group, "strategyName": strategy})
            set_param("weight", weight, item)
    write_config(tree, config_path)
