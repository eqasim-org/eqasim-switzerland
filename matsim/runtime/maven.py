import subprocess as sp
import os, shutil
import logging
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
logger = logging.getLogger("synpp")


def configure(context):
    context.config("maven_binary", "mvn")
    context.config("maven_skip_tests", False)


def _proxy_settings(directory):
    """Create job-local Maven settings from the cluster's proxy environment."""
    proxies = {}
    for protocol in ("http", "https"):
        address = os.environ.get(f"{protocol}_proxy") or os.environ.get(f"{protocol.upper()}_PROXY")
        if address:
            proxies[protocol] = urlsplit(address if "://" in address else "http://" + address)
    if not proxies:
        return None

    user_settings = Path.home() / ".m2" / "settings.xml"
    if user_settings.exists():
        root = ET.parse(user_settings).getroot()
    else:
        root = ET.Element("{http://maven.apache.org/SETTINGS/1.0.0}settings")
    namespace = root.tag.partition("}")[0] + "}" if root.tag.startswith("{") else ""
    if namespace:
        ET.register_namespace("", namespace[1:-1])
    tag = lambda name: namespace + name
    container = root.find(tag("proxies"))
    if container is None:
        container = ET.SubElement(root, tag("proxies"))
    exclusions = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or "localhost,127.0.0.1"
    exclusions = "|".join("*" + item if item.startswith(".") else item
                          for item in (part.strip() for part in exclusions.split(",")) if item)
    for protocol, proxy in proxies.items():
        if not proxy.hostname:
            raise ValueError(f"Invalid {protocol} proxy: missing hostname")
        for existing in list(container):
            if existing.findtext(tag("protocol"), "http") == protocol:
                container.remove(existing)
        element = ET.SubElement(container, tag("proxy"))
        values = {"id": f"cluster-{protocol}", "active": "true", "protocol": protocol,
                  "host": proxy.hostname, "port": str(proxy.port or (443 if proxy.scheme == "https" else 80)), "nonProxyHosts": exclusions}
        if proxy.username:
            values["username"] = unquote(proxy.username)
        if proxy.password:
            values["password"] = unquote(proxy.password)
        for name, value in values.items():
            ET.SubElement(element, tag(name)).text = value
    path = Path(directory) / "maven-settings.xml"
    # Settings can contain proxy credentials or existing Maven server credentials.
    with open(path, "wb", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
    path.chmod(0o600)
    return str(path)


def run(context, arguments = [], cwd = None):
    """
        This function calls Maven.
    """
    # Make sure there is a dependency
    settings = context.stage("matsim.runtime.maven")

    if cwd is None:
        cwd = context.path()

    # Prepare temp folder
    temp_path = "%s/__java_temp" % context.path()
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    vm_arguments = [
        "-Djava.io.tmpdir=%s" % temp_path
    ]

    if settings["skip_tests"]:
        vm_arguments.append("-DskipTests=true")

    command_line = [shutil.which(settings["binary"])]
    if os.environ.get("MAVEN_USE_ENV_PROXY", "").lower() == "true":
        proxy_settings = _proxy_settings(temp_path)
        if proxy_settings:
            logger.info("Using Maven proxy settings from the job environment")
            command_line.extend(["--settings", proxy_settings])
    command_line += vm_arguments + arguments

    return_code = sp.check_call(command_line, cwd = cwd)

    if not return_code == 0:
        raise RuntimeError("Maven return code: %d" % return_code)


def validate(context):
    if shutil.which(context.config("maven_binary")) in ["", None]:
        raise RuntimeError("Cannot find Maven binary at: %s" % context.config("maven_binary"))

    if not b"3." in sp.check_output([
        shutil.which(context.config("maven_binary")),
        "-version"
    ], stderr = sp.STDOUT):
        logger.warning("Maven of at least version 3.x.x is recommended!")


def execute(context):
    return {
        "binary": context.config("maven_binary"),
        "skip_tests": context.config("maven_skip_tests")
    }
