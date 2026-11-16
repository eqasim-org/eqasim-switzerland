"""Review buffered municipal opening-hours regions and their source coverage."""

import json
from pathlib import Path

import folium
from branca.element import Element
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE = "data.spatial.opening_hours_regions"
TILES = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def configure(context):
    context.stage(STAGE)
    context.config("analysis_path")


def execute(context):

    # Read the stage's actual GeoDataFrame and its municipality-level evidence.

    regions = context.stage(STAGE)
    cache = Path(context.path(STAGE))
    municipalities = gpd.read_parquet(cache / "municipalities.parquet")
    profiles = pd.read_parquet(cache / "profiles.parquet")
    summary = json.loads((cache / "summary.json").read_text())

    # Export a review copy and regenerate the profiles and interactive map.

    output = Path(context.config("analysis_path")) / "opening_hours_municipalities"
    output.mkdir(parents=True, exist_ok=True)
    regions.to_parquet(output / "regions.parquet", index=False)
    municipalities.to_parquet(output / "municipalities.parquet", index=False)
    profiles.to_parquet(output / "profiles.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(regions, municipalities, profiles, summary, output)

    # Return review artifacts without changing the clustering-stage contract.

    return {"html": str(output / "regions.html"), "figure": str(output / "profiles.png"),
            "summary": str(output / "summary.json")}


def write_report(regions, municipalities, profiles, summary, output):
    counts = profiles.groupby("opening_hours_region").agg(municipalities=("municipality_id", "size"),
        shop_fallback=("shop_imputed", "sum"), leisure_fallback=("leisure_imputed", "sum"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, purpose in zip(axes, ("shop", "leisure")):
        columns = [f"{purpose}_{slot:02d}" for slot in range(48)]
        for label, group in profiles.groupby("opening_hours_region"):
            ax.plot(np.arange(15, 1440, 30)/60, group[columns].mean().to_numpy(),
                color=COLORS[int(label) % len(COLORS)], label=f"Region {label} ({len(group)} municipalities)")
        ax.set(title=purpose, xlabel="Hour of day", ylabel="Mean municipal share of businesses open", xlim=(0,24), ylim=(0,1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Municipality profiles — adaptive buffers, ≥{summary['min_businesses']} businesses per purpose; seed {summary['random_seed']}")
    fig.savefig(output / "profiles.png", dpi=170)
    plt.close(fig)

    map_ = folium.Map(location=[46.8,8.2], zoom_start=8, tiles=None, prefer_canvas=True)
    basemap = folium.TileLayer(tiles=TILES, attr="© swisstopo / geo.admin.ch", name="Swisstopo national map",
                              opacity=.65, max_zoom=18).add_to(map_)
    display = regions.drop(columns="municipality_ids").copy()
    for column in counts:
        display[column] = display.opening_hours_region.map(counts[column])
    display.geometry = display.geometry.simplify(40, preserve_topology=True)
    def style(feature):
        color = COLORS[int(feature["properties"]["opening_hours_region"]) % len(COLORS)]
        return {"color": color, "weight": 1, "fillColor": color, "fillOpacity": .45}
    folium.GeoJson(display.to_crs(4326).to_json(), name="Opening-hours regions", style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["opening_hours_region", "municipalities", "shop_fallback", "leisure_fallback"])).add_to(map_)
    columns = ["municipality_id", "shop_businesses", "leisure_businesses", "shop_local_businesses",
               "leisure_local_businesses", "buffer_m", "buffer_expansions", "shop_target_met", "leisure_target_met",
               "shop_imputed", "leisure_imputed"]
    municipal_display = municipalities.merge(profiles[columns], on="municipality_id", validate="one_to_one")
    municipal_display.geometry = municipal_display.geometry.simplify(40, preserve_topology=True)
    folium.GeoJson(municipal_display.to_crs(4326).to_json(), name="Municipalities and business coverage", show=False,
        style_function=style, tooltip=folium.GeoJsonTooltip(fields=["municipality_id", "opening_hours_region", *columns[1:]])).add_to(map_)
    for name, location in {"Geneva": [46.2044,6.1432], "Lausanne": [46.5197,6.6323], "Zurich": [47.3769,8.5417]}.items():
        folium.CircleMarker(location, radius=3, color="#222", fill=True, tooltip=name).add_to(map_)
    folium.LayerControl().add_to(map_)
    pane = f'''<div style="position:fixed;bottom:20px;left:15px;z-index:1000;background:white;padding:14px;
        max-width:530px;max-height:42vh;overflow:auto;font:13px sans-serif;box-shadow:0 1px 8px #777">
        <b>Municipal opening-hours regions — adaptive buffers</b>
        <p>Seed {summary['random_seed']}; {summary['municipalities']} municipalities. Buffers select businesses;
        displayed regions retain municipal boundaries. Display geometry simplified by 40m; Parquet is full detail.</p>
        <label>Background opacity <input aria-label="Background opacity" type="range" min="0" max="1" step="0.05"
        value="0.65" oninput="{basemap.get_name()}.setOpacity(this.value)"></label>
        {counts.to_html()}<p>Target: at least {summary['min_businesses']} shopping and {summary['min_businesses']} leisure businesses.
        Start at {summary['buffer_m']/1000:g} km, expand in {summary['buffer_step_m']/1000:g} km steps until both targets are met.
        Expanded {summary['expanded_municipalities']} municipalities; median radius {summary['median_buffer_m']/1000:g} km;
        maximum {summary['maximum_buffer_m']/1000:g} km. Targets not met: {summary['targets_not_met']}.</p>
        <p>Fallback is used only when the available observations cannot meet a purpose target.</p>
        <p>Without buffers: {summary['shop_sparse_without_buffer']} shop and {summary['leisure_sparse_without_buffer']} leisure
        profiles below the threshold. With buffers: {summary['shop_imputed']} and {summary['leisure_imputed']}.</p>
        <a href="profiles.png">Opening profiles</a><p>© swisstopo / geo.admin.ch. Background tiles require internet;
        the table remains usable if tiles fail.</p></div>'''
    map_.get_root().html.add_child(Element(pane))
    map_.save(output / "regions.html")
