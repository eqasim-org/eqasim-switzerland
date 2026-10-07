import xml.etree.ElementTree as ET

import xopen


def read_stop_coordinates(schedule_path):
    """Extract (x, y) coordinates of every stopFacility in a MATSim transit schedule file."""
    coordinates = []

    with xopen.xopen(schedule_path, "r") as f:
        for _, elem in ET.iterparse(f, events = ["end"]):
            if elem.tag == "stopFacility":
                coordinates.append((float(elem.attrib["x"]), float(elem.attrib["y"])))
            elem.clear()

    return coordinates
