"""Compare NPVM and legacy freight using a Synpp stage or freight-only cache reader."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import pickle
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from pyproj import Transformer
from synpp.pipeline import get_config_value, process_stages
import yaml

from analysis.freight.trips_html import freight_zone_pairs
from synthesis.freight.npvm.od import MATRIX_OPTIONS
from synthesis.freight.npvm.locations import read_zones

logger = logging.getLogger("synpp")
DEPENDENCIES = {
    "npvm": "synthesis.freight.npvm.trips", "npvm_od": "synthesis.freight.npvm.od",
    "npvm_locations": "synthesis.freight.npvm.locations", "gte": "synthesis.freight.gte.trips",
    "gqgv": "synthesis.freight.gqgv.trips", "gte_od": "data.freight.gte.od",
    "gqgv_od": "data.freight.gqgv.od", "cantons": "data.spatial.cantons",
}
OPTIONS = ("analysis_path", "input_downsampling", "random_seed", "enable_scaling", "scaling_year")
MODELS = ("Legacy", "NPVM")
SCOPES = ("all", "domestic", "crossborder", "transit")
DISTANCE_EDGES = np.arange(0., 505., 5.)
SPECIAL = {"External": "Foreign source zone", "Unassigned": "Domestic source zone outside/missing Swiss canton", "Invalid": "Invalid endpoint coordinates"}


def configure(context):
    for name in OPTIONS:
        context.config(name)
    context.config("npvm_freight_demand_factor", default=1.0)
    context.config("data_path")
    context.config("npvm_freight_swiss_zones_file", default="npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg")
    for option, default in MATRIX_OPTIONS.values():
        context.config(option, default=default)
    for alias, name in DEPENDENCIES.items():
        context.stage(name, alias=alias)
    context.stage("synthesis.freight.trips", config={"use_npvm_freight": False}, alias="legacy")


def execute(context):

    # Load both methods explicitly, regardless of the configured production freight source.

    inputs = {alias: context.stage(alias) for alias in [*DEPENDENCIES, "legacy"]}
    options = {name: context.config(name) for name in (*OPTIONS, "npvm_freight_demand_factor")}
    options["npvm_matrix_files"] = {source: context.config(option) for source, (option, _) in MATRIX_OPTIONS.items()}
    cantons = inputs["cantons"].loc[inputs["cantons"].canton_id.between(1, 26)].copy().to_crs(2056)
    cantons["id"] = cantons.canton_id.astype(int).astype(str)
    cantons = cantons.sort_values("canton_id").reset_index(drop=True)
    if len(cantons) != 26 or cantons.canton_id.duplicated().any():
        raise ValueError("Freight comparison requires the 26 distinct Swiss canton polygons")

    # Recover original OD geography before foreign endpoints are placed on the Swiss border.

    frames = enrich_trips(inputs, options, cantons)

    # Place legacy endpoints in the same NPVM zone geography used by the new demand.

    zone_file = Path(context.config("data_path")) / context.config("npvm_freight_swiss_zones_file")
    npvm_zones = read_zones(zone_file, "No", "N_Gem", True)
    npvm_zones["id"] = npvm_zones.zone_id.astype(str)
    npvm_zones = npvm_zones.sort_values("zone_id").reset_index(drop=True)
    assign_npvm_zones(frames, npvm_zones)

    # Aggregate distributions, endpoint counts, unique locations and canton OD flows.

    payload, tables = analyse(frames, cantons, npvm_zones)
    sources = {alias: Path(context.path(alias)).name for alias in [*DEPENDENCIES, "legacy"]}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), **options, "source_caches": sources,
                "legacy_basis": "GTE/GQGV daily demand; legacy freight projections applied when enable_scaling=true",
                "npvm_crossing_source": inputs["npvm_locations"].get("crossing_source", "unspecified"),
                "npvm_basis": "Supplied NPVM average-weekday vehicle matrices (default release: 2023), adjusted only by npvm_freight_demand_factor and input_downsampling",
                "distance_basis": "Euclidean distance between generated EPSG:2056 endpoints, not routed distance. Foreign endpoints are on the Swiss border.",
                "total_distance_basis": "Sum of all valid trip distances within the selected scope, at the configured sample fraction; no expansion or routing.",
                "npvm_zone_basis": "NPVM native domestic OD zone IDs; legacy domestic endpoints spatially assigned to the same published model-domestic NPVM polygons. Foreign endpoints remain External.",
                "npvm_zone_file": str(context.config("npvm_freight_swiss_zones_file")),
                "canton_basis": "Source-foreign endpoints remain External regardless of border coordinates; domestic endpoints are spatially assigned to the 26 Swiss cantons.",
                "scope_basis": "Domestic/cross-border/transit refer to the source-zone classification. The NPVM domestic zone layer also includes Liechtenstein/enclaves.",
                "departure_basis": "Both methods draw from the same freight departure-time profile; their differences are sampling variation.",
                "location_basis": "Unique coordinates rounded to 1 mm; foreign coordinates describe border proxies, not original foreign establishments."}
    if hasattr(context, "reader"):
        metadata["input_provenance"] = context.reader.provenance

    # Export the standalone HTML, aggregate CSVs and auditable source information.

    output = Path(options["analysis_path"]) / "freight_comparison"
    write_report(payload, tables, metadata, output)
    return str(output / "comparison.html")


def enrich_trips(inputs, options, cantons):
    """Align sampled source zones with trip rows and retain their foreign/domestic identity."""
    legacy = inputs["legacy"].copy()
    components = pd.concat([inputs["gte"], inputs["gqgv"]], ignore_index=True)
    pd.testing.assert_frame_equal(legacy[components.columns].reset_index(drop=True), components, check_dtype=False)
    old_pairs = pd.concat([freight_zone_pairs(inputs[f"{kind}_od"], inputs[kind], options)
                           for kind in ("gte", "gqgv")], ignore_index=True)
    legacy[["origin_zone", "destination_zone"]] = old_pairs.to_numpy()
    legacy["source"] = ["GTE"] * len(inputs["gte"]) + ["GQGV"] * len(inputs["gqgv"])
    for end in ("origin", "destination"):
        legacy[end + "_domestic"] = legacy[end + "_zone"].str.startswith("CH")
    npvm = inputs["npvm"].copy()
    counts = inputs["npvm_od"]
    pairs = counts.loc[counts.index.repeat(counts.n_trips), ["origin_zone", "destination_zone", "source"]].reset_index(drop=True)
    if len(npvm) != len(pairs) or not np.array_equal(npvm.agent_id, np.arange(len(npvm))):
        raise ValueError("NPVM sampled OD counts do not align with freight agent IDs")
    npvm[["origin_zone", "destination_zone", "source"]] = pairs.to_numpy()
    domestic_ids = set(inputs["npvm_locations"]["zones"].loc[lambda x: x.is_swiss, "zone_id"])
    for end in ("origin", "destination"):
        npvm[end + "_domestic"] = npvm[end + "_zone"].isin(domestic_ids)
    frames = {"Legacy": legacy, "NPVM": npvm}
    for label, frame in frames.items():
        if frame.agent_id.duplicated().any():
            raise ValueError(f"Duplicate {label} freight agent IDs")
        for end in ("origin", "destination"):
            frame[end + "_canton"], frame[end + "_boundary_ties"] = assign_cantons(frame, end, cantons)
        a = frame[["origin_x", "origin_y"]].to_numpy(dtype=float)
        b = frame[["destination_x", "destination_y"]].to_numpy(dtype=float)
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (a > 0).all(axis=1) & (b > 0).all(axis=1)
        frame["distance_km"] = np.where(valid, np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]) / 1000, np.nan)
        frame["scope"] = np.select([frame.origin_domestic & frame.destination_domestic,
                                     ~frame.origin_domestic & ~frame.destination_domestic],
                                    ["domestic", "transit"], default="crossborder")
    return frames


def assign_cantons(frame, endpoint, cantons):
    xy = frame[[endpoint + "_x", endpoint + "_y"]].to_numpy(dtype=float)
    valid = np.isfinite(xy).all(axis=1) & (xy > 0).all(axis=1)
    domestic = frame[endpoint + "_domestic"].to_numpy(dtype=bool)
    labels = np.full(len(frame), "External", dtype=object)
    labels[domestic] = "Unassigned"
    labels[~valid] = "Invalid"
    ties = np.zeros(len(frame), dtype=bool)
    selected = np.flatnonzero(valid & domestic)
    if len(selected):
        points = gpd.GeoDataFrame({"row": selected}, geometry=gpd.points_from_xy(xy[selected, 0], xy[selected, 1]), crs=2056)
        joined = gpd.sjoin(points, cantons[["id", "geometry"]], how="inner", predicate="intersects")
        repeated = joined.row.value_counts().loc[lambda x: x > 1].index
        ties[repeated.to_numpy(dtype=int)] = True
        joined["sort_id"] = joined.id.astype(int)
        joined = joined.sort_values(["row", "sort_id"]).drop_duplicates("row")
        labels[joined.row.to_numpy(dtype=int)] = joined.id.to_numpy()
    return labels, ties


def assign_npvm_zones(frames, zones):
    """Preserve NPVM source IDs; spatially disaggregate legacy domestic coordinates."""
    ids = set(zones.id)
    for model, frame in frames.items():
        for end in ("origin", "destination"):
            if model == "NPVM":
                xy = frame[[end + "_x", end + "_y"]].to_numpy(dtype=float)
                valid = np.isfinite(xy).all(axis=1) & (xy > 0).all(axis=1)
                source_ids = frame[end + "_zone"].astype(str)
                domestic = frame[end + "_domestic"].to_numpy(dtype=bool)
                labels = np.full(len(frame), "External", dtype=object)
                labels[domestic] = "Unassigned"
                known = domestic & source_ids.isin(ids).to_numpy()
                labels[known] = source_ids.to_numpy()[known]
                labels[~valid] = "Invalid"
                ties = np.zeros(len(frame), dtype=bool)
            else:
                labels, ties = assign_cantons(frame, end, zones)
            frame[end + "_npvm_zone"] = labels
            frame[end + "_npvm_boundary_ties"] = ties


def endpoint_metrics(frame, endpoint, region_ids, column=None):
    """Group once so the same accounting remains efficient for thousands of zones."""
    column = column or endpoint + "_canton"
    counts = frame.groupby(column).size()
    coordinates = frame[[endpoint + "_x", endpoint + "_y"]].to_numpy(dtype=float)
    valid = np.isfinite(coordinates).all(axis=1) & (coordinates > 0).all(axis=1)
    locations = pd.DataFrame(np.round(coordinates[valid], 3), columns=["x", "y"])
    locations["region"] = frame.loc[valid, column].to_numpy()
    unique = locations.drop_duplicates().groupby("region").size()
    result = {key: {"trips": int(counts.get(key, 0)), "locations": int(unique.get(key, 0))}
              for key in [*region_ids, *SPECIAL]}
    if sum(x["trips"] for x in result.values()) != len(frame):
        raise AssertionError("Region endpoint accounting does not sum to all trips")
    return result


def distribution(frame):
    d = frame.distance_km.dropna().to_numpy()
    counts, _ = np.histogram(d, bins=np.r_[DISTANCE_EDGES, np.inf])
    departures = pd.to_numeric(frame.departure_time, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(departures) & (departures >= 0)
    hours = np.floor(departures[valid] / 3600).astype(int)
    unique_hours, n_hours = np.unique(hours, return_counts=True)
    def stat(q):
        return float(np.quantile(d, q)) if len(d) else None
    return {"total": len(frame), "valid_distances": len(d), "invalid_distances": int(frame.distance_km.isna().sum()),
            "total_distance_km": float(d.sum()),
            "zero_distance": int((d == 0).sum()), "mean_km": float(d.mean()) if len(d) else None,
            "median_km": stat(.5), "p90_km": stat(.9), "p95_km": stat(.95), "p99_km": stat(.99),
            "histogram": counts.tolist(), "cdf_x": np.quantile(d, np.linspace(0, 1, 1001)).tolist() if len(d) else [],
            "cdf_y": np.linspace(0, 100, 1001).tolist() if len(d) else [],
            "hours": dict(zip(map(str, unique_hours), map(int, n_hours))), "invalid_departures": int((~valid).sum())}


def location_cells(frame, endpoint):
    xy = frame[[endpoint + "_x", endpoint + "_y"]].to_numpy(dtype=float)
    valid = np.isfinite(xy).all(axis=1) & (xy > 0).all(axis=1)
    grid = pd.DataFrame({"x": np.floor(xy[valid, 0] / 5000) * 5000 + 2500,
                         "y": np.floor(xy[valid, 1] / 5000) * 5000 + 2500,
                         "external": ~frame.loc[valid, endpoint + "_domestic"].to_numpy(dtype=bool)})
    grid = grid.groupby(["x", "y", "external"], as_index=False).size()
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    lon, lat = transformer.transform(grid.x.to_numpy(), grid.y.to_numpy())
    return [[round(float(a), 5), round(float(b), 5), int(n), bool(e)]
            for a, b, n, e in zip(lat, lon, grid["size"], grid.external)]


def analyse(frames, cantons, npvm_zones=None):
    """Create additive trip counts and common distance bins for every displayed scope."""
    canton_ids = cantons.id.tolist()
    payload = {"models": {}, "cantons": [{"id": r.id, "name": r.canton_name} for r in cantons.itertuples()],
               "special": SPECIAL, "distance_labels": [f"{int(x)}–{int(x+5)}" for x in DISTANCE_EDGES[:-1]] + ["500+"],
               "zones": json.loads(cantons[["id", "canton_name", "geometry"]].assign(
                   geometry=cantons.geometry.simplify(120, preserve_topology=True)).to_crs(4326).to_json(drop_id=True))}
    tables = {key: [] for key in ["summary", "canton_counts", "distance_distribution", "departure_times",
                                 "source_counts", "canton_od", "location_cells", "npvm_zone_counts"]}
    if npvm_zones is not None:
        payload["npvm_zones"] = json.loads(npvm_zones[["id", "name", "geometry"]].assign(
            geometry=npvm_zones.geometry.simplify(15, preserve_topology=True)).to_crs(4326).to_json(drop_id=True))
        payload["npvm_special"] = {**SPECIAL, "Unassigned": "Domestic endpoint without a matching NPVM zone"}
    for model, frame in frames.items():
        payload["models"][model] = {}
        for scope in SCOPES:
            part = frame if scope == "all" else frame.loc[frame.scope.eq(scope)]
            stats = distribution(part)
            stats["endpoints"] = {end: endpoint_metrics(part, end, canton_ids) for end in ("origin", "destination")}
            stats["cells"] = {end: location_cells(part, end) for end in ("origin", "destination")}
            stats["boundary_ties"] = {end: int(part[end + "_boundary_ties"].sum()) for end in ("origin", "destination")}
            if npvm_zones is not None:
                stats["npvm_endpoints"] = {end: endpoint_metrics(part, end, npvm_zones.id.tolist(), end + "_npvm_zone")
                                           for end in ("origin", "destination")}
                stats["npvm_boundary_ties"] = {end: int(part[end + "_npvm_boundary_ties"].sum())
                                               for end in ("origin", "destination")}
                for endpoint, metrics in stats["npvm_endpoints"].items():
                    for zone_id, values in metrics.items():
                        tables["npvm_zone_counts"].append(dict(model=model, scope=scope, endpoint=endpoint, zone_id=zone_id, **values))
            payload["models"][model][scope] = stats
            tables["summary"].append({"model": model, "scope": scope, **{k: v for k, v in stats.items() if not isinstance(v, (list, dict))}})
            for endpoint, metrics in stats["endpoints"].items():
                for canton_id, values in metrics.items():
                    tables["canton_counts"].append(dict(model=model, scope=scope, endpoint=endpoint, canton_id=canton_id, **values))
            for i, n in enumerate(stats["histogram"]):
                tables["distance_distribution"].append(dict(model=model, scope=scope, distance_bin_km=payload["distance_labels"][i], trips=n))
            for hour, n in stats["hours"].items():
                tables["departure_times"].append(dict(model=model, scope=scope, departure_hour=int(hour), trips=n))
            for source, n in part.groupby("source").size().items():
                tables["source_counts"].append(dict(model=model, scope=scope, source=source, trips=int(n)))
            for (origin, destination), n in part.groupby(["origin_canton", "destination_canton"]).size().items():
                tables["canton_od"].append(dict(model=model, scope=scope, origin=origin, destination=destination, trips=int(n)))
            for endpoint, cells in stats["cells"].items():
                for lat, lon, count, external in cells:
                    tables["location_cells"].append(dict(model=model, scope=scope, endpoint=endpoint, lat=lat, lon=lon, trips=count, external_source=external))
    return payload, {name: pd.DataFrame(rows) for name, rows in tables.items()}


def write_report(payload, tables, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    assets = {}
    for extension in ("js", "css"):
        with urlopen(f"https://unpkg.com/leaflet@1.9.4/dist/leaflet.{extension}", timeout=30) as response:
            assets[extension] = response.read().decode()
    template = Path(__file__).with_name("compare.html.template").read_text()
    replacements = {"__LEAFLET_JS__": assets["js"], "__LEAFLET_CSS__": assets["css"], "__PLOTLY_JS__": get_plotlyjs(),
                    "__DATA__": json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
                    "__META__": json.dumps(metadata, allow_nan=False).replace("<", "\\u003c")}
    for key, value in replacements.items():
        template = template.replace(key, value)
    (output / "comparison.html").write_text(template, encoding="utf-8")
    logger.info("Freight comparison: %s", output / "comparison.html")


class CachedFreightReader:
    """Run freight-only code against exact existing prerequisites; never rebuild population inputs."""
    COMPUTE = {"analysis.freight.compare", "synthesis.freight.trips", "synthesis.freight.npvm.od",
               "synthesis.freight.npvm.locations", "synthesis.freight.npvm.trips",
               "synthesis.freight.npvm.crossings", "synthesis.freight.npvm.network_access",
               "synthesis.freight.npvm.routed_crossings", "synthesis.freight.npvm.zone_locations",
               "synthesis.freight.npvm.crossing_points"}

    def __init__(self, configuration):
        self.root = Path(configuration["working_directory"])
        self.registry = process_stages([dict(descriptor="analysis.freight.compare", config={}, downstream=[])], configuration["config"])
        self.results = {}
        self.provenance = []

    def load(self, key):
        if key not in self.results:
            stage = self.registry[key]
            name = stage["wrapper"].name
            if name in self.COMPUTE:
                self.results[key] = stage["wrapper"].execute(ReaderContext(self, stage))
                self.provenance.append({"stage": name, "method": "executed freight/comparison code using cached prerequisites"})
            else:
                path = self.root / (key + ".p")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing exact prerequisite cache {path}. Build that stage with this configuration first; no population stages were launched.")
                with path.open("rb") as stream:
                    self.results[key] = pickle.load(stream)
                self.provenance.append({"stage": name, "method": "read existing prerequisite cache",
                    "file": path.name, "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
        return self.results[key]


class ReaderContext:
    def __init__(self, reader, stage):
        self.reader, self.stage_definition = reader, stage
        self.working_directory = str(reader.root)

    def config(self, key):
        return get_config_value(key, self.stage_definition["config"])

    def dependency(self, name):
        definition = self.stage_definition["aliases"].get(name, {"descriptor": name, "config": {}})
        index = self.stage_definition["required_stages"].index(definition)
        return self.stage_definition["dependencies"][index]

    def stage(self, name):
        return self.reader.load(self.dependency(name))

    def path(self, name=None):
        key = self.dependency(name) if name is not None else self.stage_definition["hash"]
        local = getattr(self.reader, "local_paths", {}).get(key)
        return str(local if local is not None else self.reader.root / (key + ".cache"))


def build(configuration_path):
    configuration = yaml.safe_load(Path(configuration_path).read_text())
    reader = CachedFreightReader(configuration)
    root = next(key for key, stage in reader.registry.items() if stage["wrapper"].name == "analysis.freight.compare")
    return reader.load(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", default="config_dib_10.yml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(build(args.configuration))


if __name__ == "__main__":
    main()
