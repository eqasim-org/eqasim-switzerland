"""Deterministic departure-route retention shared with the native XML exporter."""

import math

from lxml import etree


def retention_settings(context):
    fraction = float(context.config("sumo_initial_route_retention"))
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("sumo_initial_route_retention must be between zero and one")
    return fraction, int(context.config("random_seed"))


def retain_route(identifier, fraction, seed):
    """FNV-1a over UTF-8 with a final avalanche; identical to the native draw."""
    value = 14695981039346656037
    for byte in f"{seed}:{identifier}".encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return (value >> 11) / 2**53 < fraction


def retain_departure_routes(payloads, fraction, seed):
    """Leave explicit device parameters intact and suppress only pre-departure routing."""
    for payload in payloads:
        element = etree.fromstring(payload)
        if retain_route(element.get("id"), fraction, seed):
            if element.find("param[@key='device.rerouting.pre-period']") is None:
                etree.SubElement(
                    element, "param", key="device.rerouting.pre-period", value="0"
                )
            yield etree.tostring(element)
        else:
            yield payload
