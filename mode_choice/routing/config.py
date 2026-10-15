"""A private router config using the existing fitted DMC cost parameters."""
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

from mode_choice.defaults import ROUTER_CONFIG_INPUTS


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("dmc.params")


def execute(context):
    source = Path(context.stage("calibration.pt_pricing.generate_config")).resolve()
    _, cost_parameters = context.stage("dmc.params")
    tree = ET.parse(source)
    root = tree.getroot()
    # Moving the config must not change where transit and pricing inputs resolve.
    for module, names in ROUTER_CONFIG_INPUTS.items():
        for param in root.findall(f"./module[@name='{module}']/param"):
            value = param.get("value", "")
            if param.get("name") in names and value and value != "null":
                param.set("value", urljoin(source.as_uri(), value))
    eqasim = root.find("./module[@name='eqasim']")
    if eqasim is None:
        eqasim = ET.SubElement(root, "module", name="eqasim")
    param = eqasim.find("./param[@name='costParametersPath']")
    if param is None:
        param = ET.SubElement(eqasim, "param", name="costParametersPath")
    param.set("value", str(Path(cost_parameters).resolve()))
    target = Path(context.path()) / "router_config.xml"
    with target.open("w") as stream:
        stream.write('<?xml version="1.0" encoding="utf-8"?>\n'
                     '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">\n')
        stream.write(ET.tostring(root, encoding="unicode"))
    return str(target.resolve())
