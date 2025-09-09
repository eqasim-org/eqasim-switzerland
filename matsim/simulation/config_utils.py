from lxml import etree
import os.path

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

    print(parameters)

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
