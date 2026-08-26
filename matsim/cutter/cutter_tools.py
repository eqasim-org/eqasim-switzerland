import os
import re
from lxml import etree
import pandas as pd
import geopandas as gpd
from shapely import contains_xy
from matsim.readers import read_network
import glob

_PATH_SEGMENT = re.compile(r"^(?P<name>.+?)(?:\[(?P<index>\d+)\])?$")


def change_params(config_path, params, output_path = None):
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
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"MATSim config file does not exist: {config_path}")

    if output_path is None:
        output_path = config_path

    tree = etree.parse(config_path)
    changes = params.items() if hasattr(params, "items") else params

    for param in changes:
        change_param(tree, param)

    # Changing one value should not reformat the complete (and often very
    # large) MATSim configuration file.
    tree.write(
        output_path,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        doctype=tree.docinfo.doctype,  # keeps <!DOCTYPE config SYSTEM "...">
    )


def change_param(tree, param):
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

    for component in path[:-1]:
        name, index = _parse_path_component(component, param_path)
        next_containers = []

        for container in containers:
            matches = [
                child for child in container if _container_name(child) == name
            ]

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




def cut_csv_to_region(csv_path, region_path, output_path = None, only_origin=False,
                      origin_x='origin_x', origin_y='origin_y', 
                      destination_x='destination_x', destination_y='destination_y'):
    assert os.path.exists(csv_path)
    assert os.path.exists(region_path)
    
    # Read the file
    df = pd.read_csv(csv_path)
    region = gpd.read_file(region_path).union_all()

    # Check points entirely within the region
    within_region = contains_xy(region, df[origin_x], df[origin_y])
    if not only_origin:
        within_region &= contains_xy(region, df[destination_x], df[destination_y])

    df = df[within_region].reset_index(drop=True)
    assert len(df), f"There is no point left from: {csv_path}, within this region: {region_path}"

    # save or return
    if output_path is not None:
        df.to_csv(output_path, index=False, sep=",")
    else:
        return df


def cut_csv_to_network(csv_path, network_path, output_path = None, link_id_col = "linkId"):
    assert os.path.exists(csv_path)
    assert os.path.exists(network_path)

    # Read the file
    df = pd.read_csv(csv_path)
    links = read_network(network_path).links

    # Check points entirely within the region
    network_links = set(links.link_id.astype(str).unique())
    within_region = df[link_id_col].astype(str).isin(network_links)

    df = df[within_region].reset_index(drop=True)
    assert len(df), f"There is no point left from: {csv_path}, within this region: {network_path}"

    # save or return
    if output_path is not None:
        df.to_csv(output_path, index=False, sep=",")
    else:
        return df

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
    
    # only keep the region_dir/region.yml part, not the full path
    regions = [os.path.join(subfolder, os.path.basename(region)) for region in regions]
    return ";".join(regions)