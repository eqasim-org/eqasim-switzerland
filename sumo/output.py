"""Export a self-contained regional SUMO scenario and copies of its results."""

import hashlib
import json
from pathlib import Path
import shutil

from lxml import etree

from sumo import output_tools


def configure(context):
    context.stage("sumo.run")
    context.config("output_path")


def execute(context):

    # Resolve the completed simulation and protect every upstream cache from writes.

    result = context.stage("sumo.run")
    destination = Path(context.config("output_path")).expanduser().resolve() / "sumo"
    for key in ("config", "network", "input_routes"):
        if destination.is_relative_to(Path(result[key]).resolve().parent):
            raise ValueError("SUMO output_path must be outside the input and simulation caches")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("results", "replay", "additional"):
        (destination / name).mkdir(exist_ok=True)

    # Export the network, assigned demand and all locally generated diagnostics/results.

    output_tools.copy_compressed_xml(result["network"], destination / "network.net.xml.gz")
    output_tools.copy_compressed_xml(result["input_routes"], destination / "demand.rou.xml.gz")
    simulation = Path(result["config"]).resolve().parent
    for source in simulation.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / "results" / source.name)
    tree = etree.parse(result["config"])
    tree.find("input/net-file").set("value", "network.net.xml.gz")
    tree.find("input/route-files").set("value", "demand.rou.xml.gz")
    files = []
    for index, name in enumerate(tree.find("input/additional-files").get("value", "").split(",")):
        if name:
            source = Path(name)
            if not source.is_absolute():
                source = simulation / source
            files.append(output_tools.copy_additional(source, destination, str(index)))
    tree.find("input/additional-files").set("value", ",".join(files))
    output_tools.output_paths(tree, "replay")
    config = destination / (result["model"] + ".sumocfg")
    tree.write(str(config), encoding="utf-8", xml_declaration=True, pretty_print=True)

    # Supply a GUI copy that works without a local Swiss projection database.

    output_tools.cartesian_copy(destination / "network.net.xml.gz", destination / "network.gui.net.xml.gz")
    tree.find("input/net-file").set("value", "network.gui.net.xml.gz")
    gui = destination / "gui.sumocfg"
    tree.write(str(gui), encoding="utf-8", xml_declaration=True, pretty_print=True)
    (destination / "HOW_TO_OPEN.txt").write_text(
        "Open gui.sumocfg in SUMO-GUI (SUMO 1.28.0), then press Play.\n"
        "This locally reruns the exported scenario. Server results remain in results/;\n"
        "local simulation outputs go to replay/. Scale is already 100%: keep Scale Traffic at 1.\n"
        "The GUI network preserves Cartesian road geometry but omits geographic conversion;\n"
        "use network.net.xml.gz for georeferenced analysis.\n")
    manifest = {"directory": str(destination), "model": result["model"], "count": result["count"],
                "config": str(config), "gui_config": str(gui), "results": str(destination / "results")}
    (destination / "export.json").write_text(json.dumps(manifest, indent=2)+"\n")

    # Return a small export manifest; source simulation files remain cached and reusable.

    return manifest


def validate(context):
    folder = Path(context.config("output_path")).expanduser().resolve() / "sumo"
    available = all((folder / name).is_file() for name in
                    ("network.net.xml.gz", "network.gui.net.xml.gz", "demand.rou.xml.gz", "gui.sumocfg", "export.json"))
    return {"helpers": hashlib.sha256(Path(output_tools.__file__).read_bytes()).hexdigest(), "available": available}
