import html
import json
import logging
import os

import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from data.cross_border.destinations import make_entry_border_facility_id, make_exit_border_facility_id
from matsim.simulation.cross_border_links import build_directional_border_link_table


logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.cross_border.interview_places")
    context.stage("data.spatial.swiss_border")
    context.config("cross_border_link_mapping_network_path", default=None)

    if context.config("cross_border_link_mapping_network_path") is None:
        context.stage("matsim.scenario.network.mapped")


def _make_interview_place_destination_table(df_interview_places):
    df = df_interview_places[[
        "border_crossing_point_id", "interview_place", "label", "importance", "geometry"
    ]].copy()

    # Reuse the same input shape as data.cross_border.destinations so this
    # analysis stage exercises the production directional-link matcher directly.
    df["entry_interview_point_id"] = df["border_crossing_point_id"].apply(make_entry_border_facility_id)
    df["entry_interview_geometry_point"] = df["geometry"]
    df["exit_interview_point_id"] = df["border_crossing_point_id"].apply(make_exit_border_facility_id)
    df["exit_interview_geometry_point"] = df["geometry"]

    return df


def _popup_table(fields):
    rows = []
    for key, value in fields:
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )
    return "<table class=\"popup-table\">" + "".join(rows) + "</table>"


def _to_lat_lon(transformer, x, y):
    lon, lat = transformer.transform(x, y)
    return [lat, lon]


def _build_map_records(assignments):
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    entry_links = []
    exit_links = []
    interview_points = []

    for item in assignments.itertuples(index=False):
        link_record = {
            "coords": [
                _to_lat_lon(transformer, x, y)
                for x, y in item.link_geometry.coords
            ],
            "popup": _popup_table([
                ("interview place", item.interview_place),
                ("direction", item.direction),
                ("facility id", item.facility_id),
                ("link id", item.link_id),
                ("distance m", f"{item.distance:.1f}"),
                ("from inside CH", item.from_inside_ch),
                ("to inside CH", item.to_inside_ch),
                ("from border m", f"{item.from_border_distance:.1f}"),
                ("to border m", f"{item.to_border_distance:.1f}"),
                ("crosses border", item.crosses_swiss_border),
                ("match type", item.match_type),
            ]),
        }

        if item.direction == "entry":
            entry_links.append(link_record)
        else:
            exit_links.append(link_record)

        interview_points.append({
            "coord": _to_lat_lon(transformer, item.geometry.x, item.geometry.y),
            "direction": item.direction,
            "popup": _popup_table([
                ("crossing point", item.border_crossing_point_id),
                ("interview place", item.interview_place),
                ("label", item.label),
                ("direction", item.direction),
                ("facility id", item.facility_id),
                ("link id", item.link_id),
                ("distance m", f"{item.distance:.1f}"),
                ("match type", item.match_type),
            ]),
        })

    return entry_links, exit_links, interview_points


def _write_html(assignments, output_path):
    entry_links, exit_links, interview_points = _build_map_records(assignments)
    all_points = [
        point
        for record in entry_links + exit_links
        for point in record["coords"]
    ] + [record["coord"] for record in interview_points]

    center = [46.8, 8.2]
    if len(all_points) > 0:
        center = [
            sum(point[0] for point in all_points) / len(all_points),
            sum(point[1] for point in all_points) / len(all_points),
        ]

    # Leaflet uses OSM tiles so the exported file is small and easy to inspect in
    # a browser. Keep the page map-only so Leaflet has a stable viewport size.
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cross-Border Interview Place Link Mapping</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html,
    body {{
      height: 100%;
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111827;
      background: #ffffff;
    }}
    .swatch {{
      display: inline-block;
      width: 26px;
      height: 4px;
      margin-right: 6px;
      vertical-align: middle;
      border-radius: 2px;
    }}
    #map {{
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
    }}
    .leaflet-container {{
      width: 100%;
      height: 100%;
    }}
    .leaflet-control-layers {{
      font-size: 13px;
    }}
    .leaflet-popup-content {{
      min-width: 260px;
    }}
    .popup-table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0;
      font-size: 12px;
    }}
    .popup-table th,
    .popup-table td {{
      padding: 3px 5px;
      border-bottom: 1px solid #e5e7eb;
      vertical-align: top;
      text-align: left;
    }}
    .popup-table th {{
      width: 95px;
      background: #f8fafc;
      font-weight: 700;
    }}
    .map-title {{
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #d8dee9;
      border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.18);
      line-height: 1.35;
    }}
    .map-title h1 {{
      margin: 0 0 6px;
      font-size: 16px;
      font-weight: 700;
    }}
    .legend {{
      display: grid;
      gap: 4px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const entryLinks = {json.dumps(entry_links)};
    const exitLinks = {json.dumps(exit_links)};
    const interviewPoints = {json.dumps(interview_points)};
    const initialCenter = {json.dumps(center)};

    const map = L.map("map", {{
      preferCanvas: true,
      center: initialCenter,
      zoom: 9,
      zoomControl: true
    }});

    const baseMaps = {{
      "OpenStreetMap": L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
      }}),
      "Carto Light": L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
        maxZoom: 20,
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
      }})
    }};
    baseMaps["OpenStreetMap"].addTo(map);

    const entryLayer = L.layerGroup();
    const exitLayer = L.layerGroup();
    const interviewLayer = L.layerGroup();
    const bounds = [];

    function addLine(record, layer, color) {{
      const line = L.polyline(record.coords, {{
        color: color,
        weight: 6,
        opacity: 0.9,
        lineCap: "round"
      }}).bindPopup(record.popup);
      line.addTo(layer);
      record.coords.forEach(coord => bounds.push(coord));
    }}

    entryLinks.forEach(record => addLine(record, entryLayer, "#12805c"));
    exitLinks.forEach(record => addLine(record, exitLayer, "#c23829"));

    interviewPoints.forEach(record => {{
      const color = record.direction === "entry" ? "#12805c" : "#c23829";
      const marker = L.circleMarker(record.coord, {{
        radius: 5,
        color: "#ffffff",
        weight: 1.5,
        fillColor: color,
        fillOpacity: 0.95
      }}).bindPopup(record.popup);
      marker.addTo(interviewLayer);
      bounds.push(record.coord);
    }});

    entryLayer.addTo(map);
    exitLayer.addTo(map);
    interviewLayer.addTo(map);

    const title = L.control({{ position: "topleft" }});
    title.onAdd = function() {{
      const div = L.DomUtil.create("div", "map-title");
      div.innerHTML = `
        <h1>Cross-Border Interview Place Link Mapping</h1>
        <div class="legend">
          <span><span class="swatch" style="background:#12805c"></span>entry, abroad to Switzerland</span>
          <span><span class="swatch" style="background:#c23829"></span>exit, Switzerland to abroad</span>
          <span><span class="swatch" style="background:#111827"></span>interview point</span>
        </div>
      `;
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      return div;
    }};
    title.addTo(map);

    L.control.layers(baseMaps, {{
      "Entry links": entryLayer,
      "Exit links": exitLayer,
      "Interview places": interviewLayer
    }}, {{
      collapsed: false
    }}).addTo(map);

    function fitMap() {{
      map.invalidateSize();
      if (bounds.length > 0) {{
        map.fitBounds(bounds, {{ padding: [28, 28] }});
      }}
    }}

    window.addEventListener("load", fitMap);
    window.addEventListener("resize", () => map.invalidateSize());
    setTimeout(fitMap, 150);
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def execute(context):
    # Supplying an existing network path avoids triggering the expensive mapped
    # network stage when the diagnostic is run independently.
    network_path = context.config("cross_border_link_mapping_network_path")
    if network_path is None:
        logger.info("No cross_border_link_mapping_network_path configured; using matsim.scenario.network.mapped.")
        network_path = context.stage("matsim.scenario.network.mapped")["network"]
    else:
        logger.info("Using configured network for cross-border link mapping: %s", network_path)

    logger.info("Loading cross-border interview places ...")
    df_interview_places = context.stage("data.cross_border.interview_places").copy()
    logger.info("Loaded %d interview-place points.", len(df_interview_places))

    logger.info("Loading Swiss border geometry ...")
    swiss_border = context.stage("data.spatial.swiss_border")

    logger.info("Preparing directional entry/exit facilities for interview places ...")
    df_directional = _make_interview_place_destination_table(df_interview_places)
    logger.info("Prepared %d directional facility records.", 2 * len(df_directional))

    logger.info("Matching directional interview-place facilities to network links ...")
    assignments = build_directional_border_link_table(
        df_directional,
        None,
        network_path,
        swiss_border,
    )
    logger.info("Matched %d directional facilities to network links.", len(assignments))

    logger.info("Attaching interview-place metadata to link assignments ...")
    metadata = df_interview_places[[
        "border_crossing_point_id", "interview_place", "label", "importance"
    ]].copy()

    assignments["border_crossing_point_id"] = assignments["facility_id"].str.replace(
        r"_(entry|exit)$", "", regex=True
    )
    assignments = assignments.merge(metadata, on="border_crossing_point_id", how="left")

    assignments = assignments[[
        "border_crossing_point_id", "interview_place", "label", "importance",
        "direction", "facility_id", "link_id", "distance",
        "from_inside_ch", "to_inside_ch", "from_border_distance",
        "to_border_distance", "crosses_swiss_border", "match_type",
        "geometry", "link_geometry",
    ]].sort_values(["border_crossing_point_id", "direction"])

    csv_path = os.path.join(context.path(), "interview_place_link_mapping.csv")
    html_path = os.path.join(context.path(), "interview_place_link_mapping.html")

    logger.info("Writing interview-place link mapping CSV: %s", csv_path)
    # Convert geometries to WKT in the CSV so the file remains plain and easy to
    # inspect in non-GIS tools.
    csv = pd.DataFrame(assignments.drop(columns=["geometry", "link_geometry"]))
    csv["point_wkt"] = assignments["geometry"].apply(lambda geometry: geometry.wkt)
    csv["link_wkt"] = assignments["link_geometry"].apply(lambda geometry: geometry.wkt)
    csv.to_csv(csv_path, index=False)

    logger.info("Writing interview-place link mapping HTML: %s", html_path)
    _write_html(assignments, html_path)
    logger.info("Finished interview-place link mapping export.")

    return {
        "csv": csv_path,
        "html": html_path,
    }
