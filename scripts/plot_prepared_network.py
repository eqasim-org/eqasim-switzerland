"""Create an attribute-inspection HTML map from a prepared MATSim network."""
import argparse
import gzip
import html
import json
from pathlib import Path

import folium
import pandas as pd
import shapely
from lxml import etree
from pyproj import Transformer
from shapely.geometry import LineString, Point, mapping
from shapely.ops import transform


def build(cache, output, latitude, longitude, radius):
    project = Transformer.from_crs(4326, 2056, always_xy=True).transform
    unproject = Transformer.from_crs(2056, 4326, always_xy=True).transform
    center = Point(*project(longitude, latitude))
    boundary = center.buffer(radius, quad_segs=128)
    geometries = {}
    detailed_ids = set()
    for chunk in pd.read_csv(cache / "switzerland_detailed_network.csv", dtype=str, chunksize=50000):
        detailed_ids.update(chunk.LinkId)
        curves = shapely.from_wkt(chunk.Geometry.to_numpy())
        mask = shapely.intersects(curves, boundary)
        for link_id, curve in zip(chunk.LinkId[mask], curves[mask]):
            geometries[link_id] = curve
    print(f"Detailed curves intersecting buffer: {len(geometries)}", flush=True)

    nodes = {}
    features = []
    fallback = 0
    xmin, ymin, xmax, ymax = boundary.bounds
    with gzip.open(cache / "switzerland_network.xml.gz", "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",), tag=("node", "link")):
            if element.tag == "node":
                nodes[element.get("id")] = (float(element.get("x")), float(element.get("y")))
            else:
                link_id = element.get("id")
                curve = geometries.get(link_id)
                source = "Detailed geometry"
                if curve is None and link_id not in detailed_ids:
                    start, end = nodes[element.get("from")], nodes[element.get("to")]
                    if max(start[0], end[0]) >= xmin and min(start[0], end[0]) <= xmax and max(start[1], end[1]) >= ymin and min(start[1], end[1]) <= ymax:
                        candidate = LineString([start, end])
                        if candidate.intersects(boundary):
                            curve = candidate
                            source = "Straight fallback (no detailed geometry)"
                            fallback += 1
                if curve is not None:
                    properties = dict(element.attrib)
                    for attribute in element.findall("attributes/attribute"):
                        # Prefix custom attributes so none overwrite XML link fields.
                        properties[f"attribute: {attribute.get('name')}"] = {
                            "class": attribute.get("class"), "value": "".join(attribute.itertext())}
                    properties["geometry_source"] = source
                    for child in element:
                        if isinstance(child.tag, str) and child.tag != "attributes":
                            properties.setdefault("nested_xml", []).append(etree.tostring(child, encoding="unicode"))
                    rows = []
                    for key, value in properties.items():
                        if isinstance(value, dict):
                            value = f"{value['value']} ({value['class']})"
                        elif isinstance(value, list):
                            value = json.dumps(value)
                        rows.append(f"<tr><th style='text-align:left;padding:3px 8px'>{html.escape(key)}</th><td style='padding:3px 8px;white-space:normal'>{html.escape(str(value))}</td></tr>")
                    properties["hover"] = "<div style='max-height:420px;max-width:650px;overflow:auto'><table>" + "".join(rows) + "</table></div>"
                    features.append({"type": "Feature", "properties": properties,
                                     "geometry": mapping(transform(unproject, curve.intersection(boundary)))})
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    if not features:
        raise ValueError("No network links found in the buffer")

    m = folium.Map(location=[latitude, longitude], zoom_start=13, tiles=None, prefer_canvas=True)
    background = folium.TileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap", attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
        max_zoom=19).add_to(m)
    folium.Circle([latitude, longitude], radius=radius, color="#111827", weight=2,
                  fill=False, dash_array="6 6", tooltip=f"{radius / 1000:g} km buffer").add_to(m)
    folium.Marker([latitude, longitude], tooltip=f"{latitude}, {longitude}").add_to(m)
    folium.GeoJson({"type": "FeatureCollection", "features": features}, name="Prepared network",
        style_function=lambda f: {"color": "#1460b4" if "Detailed" in f["properties"]["geometry_source"] else "#e87914", "weight": 3, "opacity": 0.85},
        highlight_function=lambda f: {"color": "#e11d48", "weight": 7, "opacity": 1},
        tooltip=folium.GeoJsonTooltip(fields=["hover"], labels=False, sticky=True),
        popup=folium.GeoJsonPopup(fields=["hover"], labels=False, max_width=700),
    ).add_to(m)
    folium.LayerControl().add_to(m)
    bounds = transform(unproject, boundary).bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    panel = f"""<div style="position:fixed;top:10px;left:55px;z-index:1000;background:white;padding:12px;border-radius:6px;max-width:460px;font:13px sans-serif;box-shadow:0 1px 6px #888">
    <b>Prepared Switzerland network · {radius/1000:g} km</b><br>{latitude}, {longitude}<br>
    {len(features):,} directed links · {fallback:,} straight fallbacks (orange)<br>
    Hover for all attributes; click to pin the details.<br>
    <small>Source: {html.escape(cache.name)}<br>Detailed curves clipped at the buffer boundary.</small><br>
    Background opacity <input aria-label="Background opacity" type="range" min="0" max="1" step="0.05" value="1" oninput="{background.get_name()}.setOpacity(Number(this.value))">
    <div id="tile-status" style="color:#b45309"></div></div>"""
    m.get_root().html.add_child(folium.Element(panel))
    m.get_root().script.add_child(folium.Element(f"window.addEventListener('load',function(){{{background.get_name()}.on('tileerror',function(){{document.getElementById('tile-status').textContent='OSM tiles unavailable; network and attributes remain usable.';}});}});"))
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    print(f"Saved {output}: {len(features)} links, {fallback} fallbacks, {output.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latitude", type=float, default=46.252196)
    parser.add_argument("--longitude", type=float, default=6.138081)
    parser.add_argument("--radius", type=float, default=5000)
    args = parser.parse_args()
    build(args.cache, args.output, args.latitude, args.longitude, args.radius)
