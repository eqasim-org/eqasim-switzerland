"""Compare NPVM freight assigned to network crossings and scored interview places."""

import argparse
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from synpp.pipeline import process_stages
import yaml

from analysis.freight import border_crossings, demand_map
from analysis.freight.compare import CachedFreightReader, ReaderContext, distribution

logger = logging.getLogger("synpp")
MODELS = {"Network": False, "Interview": True}
SCOPES = ("all", "domestic", "international", "incoming", "outgoing", "transit")
ROOT = "analysis.freight.compare_crossing_sources"


def configure(context):
    for option in ("analysis_path", "input_downsampling", "random_seed"):
        context.config(option)
    network_file = context.config("npvm_freight_border_crossings_file", default=None)
    context.stage("synthesis.freight.npvm.od", alias="od")
    context.stage("data.spatial.swiss_border", alias="border")
    for model, interview in MODELS.items():
        options = {"npvm_freight_use_border_crossings": True,
                   "npvm_freight_use_interview_places": interview,
                   "npvm_freight_crossing_source": "interview" if interview else "network" if network_file else "route"}
        context.stage("synthesis.freight.npvm.trips", config=options, alias=model)
        context.stage("synthesis.freight.npvm.locations", config=options, alias=model + "_locations")


def execute(context):

    # Run both crossing sources against the same sampled NPVM OD demand.

    od = context.stage("od")
    trips = {model: context.stage(model) for model in MODELS}
    locations = {model: context.stage(model + "_locations") for model in MODELS}
    frames = {model: attach_source_zones(trips[model], od, locations[model]) for model in MODELS}
    paired, shifts = compare_endpoints(frames)
    output = Path(context.config("analysis_path")) / "npvm_crossing_source_comparison"
    output.mkdir(parents=True, exist_ok=True)

    # Aggregate actual crossing usage, trip lengths and departure-time distributions.

    payload = {"models": {}, "paired": paired, "shifts_km": shifts.distance_km.tolist()}
    map_payload = {"models": {}}
    boundary = context.stage("border").to_crs(2056).union_all().boundary
    map_payload["border"] = json.loads(gpd.GeoSeries([boundary], crs=2056).to_crs(4326).to_json())
    summaries = []
    for model, frame in frames.items():
        regions = locations[model]["zones"].loc[lambda x: ~x.is_swiss,
            ["zone_id", "name", "geometry"]].rename(columns={"zone_id": "id"}).copy()
        regions["id"] = regions.id.astype(str)
        data, tables = border_crossings.build_model(model, frame, regions, boundary)
        map_payload["models"][model] = data
        payload["models"][model] = {}
        for scope in SCOPES:
            part = select_scope(frame, scope)
            stats = distribution(part)
            stats["foreign_endpoints"] = int(part.origin_foreign.sum() + part.destination_foreign.sum())
            stats["anchors_used"] = count_anchors(part)
            payload["models"][model][scope] = stats
            summaries.append({"model": model, "scope": scope,
                              **{k: v for k, v in stats.items() if not isinstance(v, (list, dict))}})
        trips[model].to_parquet(output / (model.lower() + "_trips.parquet"), index=False, compression="zstd")
        locations[model]["border_assignments"].to_parquet(
            output / (model.lower() + "_candidates.parquet"), index=False, compression="zstd")
        for name, table in tables.items():
            table.to_parquet(output / (model.lower() + "_" + name + ".parquet"), index=False, compression="zstd")

    # Export a separate comparison so the existing NPVM-versus-legacy reports stay intact.

    pd.DataFrame(summaries).to_parquet(output / "summary.parquet", index=False, compression="zstd")
    shifts.to_parquet(output / "endpoint_shifts.parquet", index=False, compression="zstd")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "input_downsampling": context.config("input_downsampling"), "random_seed": context.config("random_seed"),
        "title": "NPVM freight · network crossings versus interview places",
        "subtitle": "Same sampled NPVM demand and seed · different crossing source and allocation weights",
        "colors": {"Network": "#216db1", "Interview": "#c06436"},
        "network_crossings_file": context.config("npvm_freight_border_crossings_file"),
        "distance_basis": "Straight-line distances between generated endpoints, not routed kilometres.",
        "count_basis": "Sampled vehicle trips, without expansion. Transit is one trip and two foreign endpoints.",
        "source_basis": {"Network": ("Foreign road access from convert_osm/network.pkl and OD-specific free-flow routing."
                         if locations["Network"].get("crossing_source") == "network_routed" else
                         "Network-export crossings; truck mode if available; exclude local roads; road-class weights."),
                         "Interview": "Current road interview references; refreshed entry/exit matches against the cached final network; importance-score weights. Truck permissions are not supplied by the reference."},
        "selection_basis": "Network routing: three sampled or nearest external major-road access points, with first Swiss entry/last exit chosen per OD. Direct crossing/interview modes: best two adjacent levels in-zone or nearest two otherwise, weighted by hierarchy or importance and distance.",
        "routing_basis": "Directed matched curves provide right offsets; coordinate selection does not force final MATSim link IDs.",
        "scope_basis": "Source NPVM zones define domestic versus foreign; model-domestic geography includes Liechtenstein/enclaves.",
        "controls": paired,
        "configured_stage_paths": {name: str(context.path(name)) for name in ["od", "border", *MODELS,
                           *[m + "_locations" for m in MODELS]]}}
    if hasattr(context, "reader"):
        metadata["input_provenance"] = context.reader.provenance
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    border_crossings.render(map_payload, metadata, output / "crossings.html")
    demand_map.render(frames, locations, metadata, output)
    render(payload, metadata, output)
    logger.info("NPVM crossing-source comparison: %s", output / "index.html")
    return str(output / "index.html")


def attach_source_zones(trips, od, locations):
    pairs = od.loc[od.index.repeat(od.n_trips), ["origin_zone", "destination_zone", "source"]].reset_index(drop=True)
    if len(pairs) != len(trips) or not np.array_equal(trips.agent_id, np.arange(len(trips))):
        raise ValueError("NPVM trip IDs do not align with sampled OD rows")
    frame = trips.copy()
    frame[pairs.columns] = pairs.astype(str).to_numpy()
    foreign = set(locations["zones"].loc[~locations["zones"].is_swiss, "zone_id"].astype(str))
    for endpoint in ("origin", "destination"):
        frame[endpoint + "_foreign"] = frame[endpoint + "_zone"].isin(foreign)
    a, b = frame.origin_foreign, frame.destination_foreign
    frame["scope"] = np.select([~a & ~b, a & ~b, ~a & b], ["domestic", "incoming", "outgoing"], default="transit")
    xy = frame[["origin_x", "origin_y", "destination_x", "destination_y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Invalid NPVM endpoint coordinates")
    frame["distance_km"] = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]) / 1000
    return frame


def select_scope(frame, scope):
    if scope == "all":
        return frame
    if scope == "international":
        return frame.loc[frame.scope.ne("domestic")]
    return frame.loc[frame.scope.isin(["incoming", "outgoing"]) if scope == "crossborder" else frame.scope.eq(scope)]


def count_anchors(frame):
    points = [frame.loc[frame[e + "_foreign"], [e + "_x", e + "_y"]].to_numpy() for e in ("origin", "destination")]
    return len(np.unique(np.vstack(points), axis=0))


def compare_endpoints(frames):
    a, b = frames["Network"], frames["Interview"]
    for column in ("agent_id", "origin_zone", "destination_zone", "source", "origin_foreign", "destination_foreign"):
        if not a[column].equals(b[column]):
            raise ValueError(f"Demand differs between the crossing-source runs: {column}")
    records = []
    domestic_changes = 0
    for endpoint in ("origin", "destination"):
        columns = [endpoint + "_x", endpoint + "_y"]
        delta = a[columns].to_numpy() - b[columns].to_numpy()
        distance = np.hypot(delta[:, 0], delta[:, 1]) / 1000
        mask = a[endpoint + "_foreign"].to_numpy()
        domestic_changes += int((distance[~mask] > 1e-9).sum())
        records.append(pd.DataFrame({"agent_id": a.loc[mask, "agent_id"], "endpoint": endpoint,
            "zone_id": a.loc[mask, endpoint + "_zone"], "distance_km": distance[mask]}))
    shifts = pd.concat(records, ignore_index=True)
    values = shifts.distance_km
    stats = {"same_demand": True, "domestic_endpoint_changes": domestic_changes,
        "departure_time_changes": int((a.departure_time != b.departure_time).sum()),
        "foreign_endpoints": len(shifts), "moved_over_1m": int((values > .001).sum()),
        "mean_shift_km": float(values.mean()) if len(values) else None,
        "median_shift_km": float(values.median()) if len(values) else None,
        "p95_shift_km": float(values.quantile(.95)) if len(values) else None}
    return stats, shifts


def render(payload, metadata, output):
    template = Path(__file__).with_name("compare_crossing_sources.html.template").read_text()
    for placeholder, value in {"__PLOTLY__": get_plotlyjs(),
        "__DATA__": json.dumps(payload, allow_nan=False).replace("<", "\\u003c"),
        "__META__": json.dumps(metadata, allow_nan=False).replace("<", "\\u003c"),
        "__MAP_HTML__": html.escape((output / "crossings.html").read_text(), quote=True)}.items():
        template = template.replace(placeholder, value)
    if (output / "demand.html").is_file():
        section = '<section class="panel"><h2>NPVM demand by zone and traffic type</h2><p>Select a source, traffic scope and zone. Entry markers are green; exit markers are orange. Counts refer to the sampled demand.</p><p><a href="demand.html">Open demand map separately</a></p><iframe title="NPVM demand by zone" srcdoc="' + html.escape((output / "demand.html").read_text(), quote=True) + '"></iframe></section>'
        template = template.replace('<section class="panel"><h2>Crossing locations and traffic allocation', section + '<section class="panel"><h2>Crossing locations and traffic allocation')
    (output / "index.html").write_text(template)


class SourceReader(CachedFreightReader):
    """Refresh interview matching locally; read all large upstream prerequisites unchanged."""
    COMPUTE = CachedFreightReader.COMPUTE | {ROOT, "data.cross_border.interview_places", "data.cross_border.road_candidates"}

    def __init__(self, configuration):
        self.root = Path(configuration["working_directory"])
        self.registry = process_stages([dict(descriptor=ROOT, config={}, downstream=[])], configuration["config"])
        self.results, self.provenance, self.local_paths = {}, [], {}
        target = Path(configuration["config"]["analysis_path"]) / "npvm_crossing_source_comparison" / "prerequisites"
        for key, stage in self.registry.items():
            if stage["wrapper"].name == "data.cross_border.road_candidates":
                self.local_paths[key] = target / key
                self.local_paths[key].mkdir(parents=True, exist_ok=True)


def build(configuration_path, network_crossings=None):
    configuration = yaml.safe_load(Path(configuration_path).read_text())
    if network_crossings:
        configuration["config"]["npvm_freight_border_crossings_file"] = str(Path(network_crossings).resolve())
    reader = SourceReader(configuration)
    key = next(key for key, stage in reader.registry.items() if stage["wrapper"].name == ROOT)
    return reader.load(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", default="config_dib_10.yml")
    parser.add_argument("--network-crossings", help="Existing network GeoParquet to use without rebuilding conversion")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(build(args.configuration, args.network_crossings))


if __name__ == "__main__":
    main()
