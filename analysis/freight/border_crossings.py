"""Map original foreign freight regions and the border anchors actually used in trips."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from synpp.pipeline import process_stages
import yaml

from analysis.freight.compare import CachedFreightReader
from analysis.freight.trips_html import freight_zone_pairs

logger = logging.getLogger("synpp")
STAGES = {
    "npvm": "synthesis.freight.npvm.trips", "npvm_od": "synthesis.freight.npvm.od",
    "npvm_locations": "synthesis.freight.npvm.locations", "gte": "synthesis.freight.gte.trips",
    "gqgv": "synthesis.freight.gqgv.trips", "gte_od": "data.freight.gte.od",
    "gqgv_od": "data.freight.gqgv.od", "nuts": "data.spatial.nuts",
    "border": "data.spatial.swiss_border",
}
ROLES = ("incoming", "outgoing", "transit_entry", "transit_exit")
OPTIONS = ("analysis_path", "random_seed", "input_downsampling", "enable_scaling", "scaling_year")


def configure(context):
    for name in OPTIONS:
        context.config(name)
    context.config("npvm_freight_demand_factor", default=1.0)
    for alias, name in STAGES.items():
        context.stage(name, alias=alias)
    context.stage("synthesis.freight.trips", config={"use_npvm_freight": False}, alias="legacy")


def execute(context):

    # Load both freight methods and their original source-zone geography.

    inputs = {alias: context.stage(alias) for alias in [*STAGES, "legacy"]}
    options = {name: context.config(name) for name in (*OPTIONS, "npvm_freight_demand_factor")}
    frames, regions = source_trips(inputs, options)
    boundary = inputs["border"].union_all().boundary
    payload = {"models": {}}
    tables = {"region_anchor_counts": [], "region_counts": [], "anchor_counts": []}

    # Read actual foreign endpoints and merge coincident points into shared anchors.

    for model in ("NPVM", "Legacy"):
        data, exports = build_model(model, frames[model], regions[model], boundary)
        payload["models"][model] = data
        for name, frame in exports.items():
            tables[name].append(frame)
    payload["border"] = json.loads(gpd.GeoSeries([boundary], crs=2056).to_crs(4326).to_json())

    # Export aggregate audits and record the source and counting conventions.

    output = Path(options["analysis_path"]) / "freight_border_crossings"
    output.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), **options,
        "source_caches": {alias: Path(context.path(alias)).name for alias in [*STAGES, "legacy"]},
        "anchor_basis": "Actual foreign endpoint coordinates from generated trip records, not re-estimated crossings.",
        "geometry_basis": "Original foreign NUTS/NPVM polygons simplified by 500 m for display only; point-only additional NPVM zones retain their source coordinates.",
        "count_basis": "A domestic/foreign OD has one foreign endpoint; a foreign/foreign OD has two. Counts are sampled trip records and endpoint occurrences, not observed border passages.",
        "direction_basis": "Incoming/outgoing have one domestic endpoint. Transit entry/exit are the two foreign endpoints of an OD; they do not prove a routed Swiss crossing.",
        "scope_basis": "NPVM model-domestic geography includes Liechtenstein/enclaves; legacy domestic geography uses CH NUTS IDs.",
        "npvm_crossing_source": inputs["npvm_locations"].get("crossing_source", "unspecified"),
        "placement_basis": ("Legacy: geometric border placement. NPVM: foreign road access points from the converted network pickle, with crossings selected by OD-specific directed free-flow routing."
                            if inputs["npvm_locations"].get("crossing_source") == "network_routed" else
                            "Legacy: nearest boundary vertex to foreign NUTS centroid. NPVM: road interview places scored by importance, using existing entry/exit network matches where available and bidirectional references otherwise. Nearest-two fallback weights also include distance. Matched curves provide one-metre right offsets; unclassified bidirectional points retain their original coordinates."
                            if inputs["npvm_locations"].get("crossing_source") == "interview_places" else
                            "Legacy: nearest boundary vertex to foreign NUTS centroid. NPVM: crossings filtered for truck (car only if the export has no truck mode), excluding local roads. Each direction uses the best two adjacent levels in-zone, or the nearest two distinct points otherwise; per-point weights motorway/trunk 8, primary 4, secondary 2, tertiary 1, divided by (1 + distance to region in km) for nearest-two fallback pools. Points are one metre right; foreign origins use in and foreign destinations use out, including transit."
                            if inputs["npvm_locations"].get("use_border_crossings", False) else
                            "Legacy: nearest boundary vertex to foreign NUTS centroid. NPVM: nearest point along boundary to foreign zone centroid or supplied point."),
        "limitation": "Anchors are not observed crossing counts or a full OD route assignment. Geometric-mode anchors need not be road-accessible; NPVM points follow the selected source's direction and offset information; coordinate shifts do not force final MATSim link IDs.",
        "global_background": "Optional Blue Marble imagery through NASA GIBS public WMTS; swisstopo is the default."}
    if hasattr(context, "reader"):
        metadata["input_provenance"] = context.reader.provenance
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    # Render a standalone map with embedded data and Leaflet assets.

    render(payload, metadata, output / "index.html")
    for model, data in payload["models"].items():
        logger.info("%s: %d cross-border ODs, %d foreign endpoint occurrences, %d regions, %d anchors",
                    model, len(data["trips"]), data["diagnostics"]["foreign_endpoints"],
                    len(data["regions"]["features"]), len(data["anchors"]))
    return str(output / "index.html")


def source_trips(inputs, options):
    """Recover source IDs in original row order; do not alter generated coordinates."""
    legacy = inputs["legacy"].copy()
    components = pd.concat([inputs["gte"], inputs["gqgv"]], ignore_index=True)
    pd.testing.assert_frame_equal(legacy[components.columns].reset_index(drop=True), components, check_dtype=False)
    pairs = pd.concat([freight_zone_pairs(inputs[f"{kind}_od"], inputs[kind], options)
                       for kind in ("gte", "gqgv")], ignore_index=True)
    legacy[["origin_zone", "destination_zone"]] = pairs.to_numpy()
    for end in ("origin", "destination"):
        legacy[end + "_foreign"] = ~legacy[end + "_zone"].str.startswith("CH")
    old_regions = inputs["nuts"][["nuts_id", "nuts_name", "geometry"]].rename(
        columns={"nuts_id": "id", "nuts_name": "name"}).copy()
    old_regions = old_regions.loc[~old_regions.id.str.startswith("CH")].copy()
    new = inputs["npvm"].copy()
    counts = inputs["npvm_od"]
    pairs = counts.loc[counts.index.repeat(counts.n_trips), ["origin_zone", "destination_zone"]].reset_index(drop=True)
    if len(new) != len(pairs) or not np.array_equal(new.agent_id, np.arange(len(new))):
        raise ValueError("NPVM source-zone counts do not align with generated agents")
    new[["origin_zone", "destination_zone"]] = pairs.astype(str).to_numpy()
    new_regions = inputs["npvm_locations"]["zones"].loc[lambda x: ~x.is_swiss,
        ["zone_id", "name", "geometry"]].rename(columns={"zone_id": "id"}).copy()
    new_regions["id"] = new_regions.id.astype(str)
    for end in ("origin", "destination"):
        new[end + "_foreign"] = new[end + "_zone"].isin(set(new_regions.id))
    return {"Legacy": legacy, "NPVM": new}, {"Legacy": old_regions, "NPVM": new_regions}


def build_model(model, frame, regions, boundary):
    """Audit foreign endpoints and their many-to-one source-region/anchor mapping."""
    frame = frame.reset_index(drop=True)
    origin_foreign = frame.origin_foreign.to_numpy(dtype=bool)
    destination_foreign = frame.destination_foreign.to_numpy(dtype=bool)
    incoming = origin_foreign & ~destination_foreign
    outgoing = ~origin_foreign & destination_foreign
    transit = origin_foreign & destination_foreign
    events = []
    for mask, end, role in ((incoming, "origin", "incoming"), (outgoing, "destination", "outgoing"),
                            (transit, "origin", "transit_entry"), (transit, "destination", "transit_exit")):
        rows = frame.loc[mask]
        events.append(pd.DataFrame({"trip_row": rows.index, "region_id": rows[end + "_zone"].astype(str),
            "x": rows[end + "_x"].to_numpy(dtype=float), "y": rows[end + "_y"].to_numpy(dtype=float), "role": role}))
    events = pd.concat(events, ignore_index=True)
    if not np.isfinite(events[["x", "y"]].to_numpy()).all():
        raise ValueError(f"{model} contains invalid foreign endpoint coordinates")
    expected = int(incoming.sum() + outgoing.sum() + 2 * transit.sum())
    if len(events) != expected:
        raise AssertionError("Foreign endpoint accounting failed")
    used = set(events.region_id)
    regions = regions.loc[regions.id.isin(used)].drop_duplicates("id").copy().to_crs(2056)
    if used - set(regions.id):
        raise ValueError(f"{model} missing foreign regions: {sorted(used - set(regions.id))}")
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    # Exact coordinates define shared anchors; display transformations do not affect grouping.
    coordinates = events[["x", "y"]].drop_duplicates().sort_values(["x", "y"]).reset_index(drop=True)
    coordinates["anchor_id"] = [f"{model}:{i}" for i in range(len(coordinates))]
    events = events.merge(coordinates, on=["x", "y"], how="left", validate="many_to_one")
    allocation = events.groupby(["region_id", "anchor_id", "role"], sort=True).size().unstack("role", fill_value=0)
    allocation = allocation.reindex(columns=ROLES, fill_value=0).reset_index()
    allocation["total"] = allocation[list(ROLES)].sum(axis=1)
    allocation.insert(0, "model", model)
    region_counts = allocation.groupby("region_id", as_index=False)[[*ROLES, "total"]].sum()
    anchor_counts = allocation.groupby("anchor_id", as_index=False)[[*ROLES, "total"]].sum().merge(coordinates, on="anchor_id", validate="one_to_one")
    anchors = []
    for row in anchor_counts.to_dict("records"):
        lon, lat = transformer.transform(row["x"], row["y"])
        records = allocation.loc[allocation.anchor_id.eq(row["anchor_id"])].to_dict("records")
        anchors.append({**row, "lat": lat, "lon": lon,
            "boundary_distance_m": float(boundary.distance(Point(row["x"], row["y"]))), "allocations": records})
    region_counts.insert(0, "model", model)
    anchor_counts.insert(0, "model", model)
    anchor_counts["boundary_distance_m"] = [float(boundary.distance(Point(x, y))) for x, y in zip(anchor_counts.x, anchor_counts.y)]
    centers = regions.geometry.centroid
    lon, lat = transformer.transform(centers.x.to_numpy(), centers.y.to_numpy())
    regions["centroid_lat"], regions["centroid_lon"] = lat, lon
    regions = regions.merge(region_counts.drop(columns="model"), left_on="id", right_on="region_id", validate="one_to_one")
    regions["source_geometry"] = regions.geometry.geom_type
    regions.geometry = regions.geometry.simplify(500, preserve_topology=True)
    geojson = json.loads(regions.to_crs(4326).to_json(drop_id=True))
    anchor_by_side = {(int(r.trip_row), r.role): r.anchor_id for r in events.itertuples()}
    trips = []
    for index in np.flatnonzero(origin_foreign | destination_foreign):
        r = frame.iloc[index]
        a_lon, a_lat = transformer.transform(float(r.origin_x), float(r.origin_y))
        b_lon, b_lat = transformer.transform(float(r.destination_x), float(r.destination_y))
        scope = "incoming" if incoming[index] else "outgoing" if outgoing[index] else "transit"
        trips.append({"origin_zone": str(r.origin_zone), "destination_zone": str(r.destination_zone),
            "a": [a_lat, a_lon], "b": [b_lat, b_lon], "scope": scope,
            "origin_anchor": anchor_by_side.get((int(index), "incoming" if incoming[index] else "transit_entry")),
            "destination_anchor": anchor_by_side.get((int(index), "outgoing" if outgoing[index] else "transit_exit"))})
    diagnostics = {"incoming_trips": int(incoming.sum()), "outgoing_trips": int(outgoing.sum()),
        "transit_ods": int(transit.sum()), "foreign_endpoints": expected,
        "max_boundary_distance_m": max((a["boundary_distance_m"] for a in anchors), default=0.),
        "anchors_over_1m_from_border": sum(a["boundary_distance_m"] > 1 for a in anchors)}
    if int(allocation.total.sum()) != expected:
        raise AssertionError("Region-anchor aggregates lost endpoints")
    return {"regions": geojson, "anchors": anchors, "trips": trips, "diagnostics": diagnostics}, {
        "region_anchor_counts": allocation, "region_counts": region_counts, "anchor_counts": anchor_counts}


def render(payload, metadata, target):
    assets = {}
    for extension in ("js", "css"):
        with urlopen(f"https://unpkg.com/leaflet@1.9.4/dist/leaflet.{extension}", timeout=30) as response:
            assets[extension] = response.read().decode()
    template = Path(__file__).with_name("border_crossings.html.template").read_text()
    for key, value in {"__LEAFLET_JS__": assets["js"], "__LEAFLET_CSS__": assets["css"],
        "__DATA__": json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
        "__META__": json.dumps(metadata, allow_nan=False).replace("<", "\\u003c")}.items():
        template = template.replace(key, value)
    target.write_text(template, encoding="utf-8")


class BorderCacheReader(CachedFreightReader):
    COMPUTE = CachedFreightReader.COMPUTE | {"analysis.freight.border_crossings"}

    def __init__(self, configuration):
        self.root = Path(configuration["working_directory"])
        self.registry = process_stages([dict(descriptor="analysis.freight.border_crossings", config={}, downstream=[])], configuration["config"])
        self.results = {}
        self.provenance = []


def build(configuration_path):
    configuration = yaml.safe_load(Path(configuration_path).read_text())
    reader = BorderCacheReader(configuration)
    root = next(key for key, stage in reader.registry.items() if stage["wrapper"].name == "analysis.freight.border_crossings")
    return reader.load(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", default="config_dib_10.yml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(build(args.configuration))


if __name__ == "__main__":
    main()
