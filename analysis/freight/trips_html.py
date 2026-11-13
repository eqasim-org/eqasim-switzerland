"""Generate an interactive freight/LCV zone and trip explorer from a configuration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from urllib.request import urlopen

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from synpp.pipeline import process_stages
import yaml

STAGES = {"Freight": "synthesis.freight.trips", "LCV": "synthesis.lcv.trips"}
AUXILIARY = {
    "nuts": "data.spatial.nuts",
    "gte_od": "data.freight.gte.od",
    "gqgv_od": "data.freight.gqgv.od",
    "gte_trips": "synthesis.freight.gte.trips",
    "gqgv_trips": "synthesis.freight.gqgv.trips",
}
COORDINATES = ["origin_x", "origin_y", "destination_x", "destination_y"]
CONFIG_OPTIONS = ("analysis_path", "data_path", "random_seed", "input_downsampling")


def configure(context):
    for option in CONFIG_OPTIONS:
        context.config(option)
    auxiliary = auxiliary_stages(context.config("use_npvm_freight", default=True))
    for name in dict.fromkeys([*STAGES.values(), *auxiliary.values()]):
        context.stage(name)


def execute(context):

    # Load the trip tables and source zones selected by the pipeline configuration.

    auxiliary = auxiliary_stages(context.config("use_npvm_freight"))
    frames = {label: context.stage(name) for label, name in {**STAGES, **auxiliary}.items()}
    configuration = {"config": {option: context.config(option) for option in CONFIG_OPTIONS}}
    configuration["config"]["use_npvm_freight"] = context.config("use_npvm_freight")
    sources = {label: {"stage": name, "cache_directory": Path(context.path(name)).name}
               for label, name in {**STAGES, **auxiliary}.items()}

    # Write the same interactive report used by the standalone cache reader.

    output = render(frames, configuration, sources, configuration_name="Synpp stage configuration")

    # Return the exported HTML path.

    return str(output)


def auxiliary_stages(use_npvm):
    if use_npvm:
        return {"npvm_od": "synthesis.freight.npvm.od",
                "npvm_locations": "synthesis.freight.npvm.locations"}
    return AUXILIARY


def load_trips(configuration):
    """Resolve configuration hashes without running stages or modifying caches."""
    definitions = [dict(descriptor=name, config={}, downstream=[]) for name in STAGES.values()]
    registry = process_stages(definitions, configuration["config"])
    cache = Path(configuration["working_directory"])
    sources = {}
    frames = {}
    labels = {**STAGES, **auxiliary_stages(configuration["config"].get("use_npvm_freight", True))}
    for stage in registry.values():
        name = stage["wrapper"].name
        if name not in labels.values():
            continue
        label = next(label for label, value in labels.items() if value == name)
        path = cache / (stage["hash"] + ".p")
        if not path.is_file():
            raise FileNotFoundError(f"No cache matching the configuration: {path}. Build the source stage first.")
        # These are trusted local Synpp results, never downloaded pickle files.
        with path.open("rb") as stream:
            frames[label] = pickle.load(stream)
        sources[label] = {"stage": name, "cache": path.name,
                          "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    return frames, sources


def freight_zone_pairs(od, trips, configuration):
    """Replay only the original multinomial OD draws, retaining their row order."""
    demands, origin_pdfs, od_pdfs = od
    rng = np.random.RandomState(configuration["random_seed"])
    parts = []
    for vehicle_type, total in demands.items():
        demand = np.round(configuration["input_downsampling"] * total)
        origins = rng.multinomial(n=demand, pvals=origin_pdfs[vehicle_type].values[:, 0])
        matrix = od_pdfs[vehicle_type]
        for i, count in enumerate(origins):
            if count == 0:
                continue
            counts = rng.multinomial(n=count, pvals=matrix.values[i, :])
            for j in np.flatnonzero(counts):
                parts.append(pd.DataFrame({"origin_zone": [str(matrix.index[i])] * counts[j],
                                           "destination_zone": [str(matrix.columns[j])] * counts[j],
                                           "vehicle_type": [vehicle_type] * counts[j]}))
    pairs = pd.concat(parts, ignore_index=True)
    if len(pairs) != len(trips) or not np.array_equal(pairs.vehicle_type, trips.vehicle_type):
        raise ValueError("Reconstructed freight OD draws do not align with the cached trips")
    return pairs[["origin_zone", "destination_zone"]]


def attach_zones(frames, configuration):
    """Use LCV's original zone IDs and recover freight IDs before spatial display."""
    freight = frames["Freight"].copy()
    if configuration["config"].get("use_npvm_freight", True):
        od = frames["npvm_od"]
        pairs = od.loc[od.index.repeat(od.n_trips), ["origin_zone", "destination_zone"]].reset_index(drop=True)
        pairs = pairs.astype(str)
        if len(pairs) != len(freight) or not np.array_equal(freight.agent_id, np.arange(len(freight))):
            raise ValueError("NPVM OD counts do not align with freight agent IDs")
    else:
        components = pd.concat([frames["gte_trips"], frames["gqgv_trips"]], ignore_index=True)
        pd.testing.assert_frame_equal(freight[components.columns].reset_index(drop=True), components,
                                      check_dtype=False)
        pairs = pd.concat([freight_zone_pairs(frames[f"{kind}_od"], frames[f"{kind}_trips"],
                                             configuration["config"]) for kind in ("gte", "gqgv")],
                          ignore_index=True)
    freight[["origin_zone", "destination_zone"]] = pairs.to_numpy()
    frames["Freight"] = freight
    lcv = frames["LCV"].copy()
    for column in ("origin_zone", "destination_zone"):
        lcv[column] = lcv[column].astype(int).astype(str)
    frames["LCV"] = lcv


def zone_features(frame, label, frames, configuration):
    """Export only used source zones, simplifying copies for HTML display."""
    ids = set(frame.origin_zone) | set(frame.destination_zone)
    if label == "Freight" and configuration["config"].get("use_npvm_freight", True):
        zones = frames["npvm_locations"]["zones"][["zone_id", "name", "geometry"]].copy()
        zones = zones.rename(columns={"zone_id": "id"})
        zones["id"] = zones.id.astype(str)
        zones["level"] = "NPVM"
    elif label == "Freight":
        zones = frames["nuts"][["nuts_id", "nuts_name", "nuts_level", "geometry"]].copy()
        zones = zones.rename(columns={"nuts_id": "id", "nuts_name": "name", "nuts_level": "level"})
        zones = zones.sort_values("level", kind="stable")
    else:
        path = Path(configuration["config"]["data_path"]) / "npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg"
        zones = gpd.read_file(path)[["No", "geometry"]].rename(columns={"No": "id"})
        zones["id"] = zones.id.astype(int).astype(str)
        zones["name"] = "NPVM " + zones.id
        zones["level"] = "NPVM"
    zones = zones[zones.id.isin(ids)].drop_duplicates("id").copy()
    missing = ids - set(zones.id)
    if missing:
        raise ValueError(f"{label}: missing polygons for source zones {sorted(missing)}")
    # Simplification affects display boundaries only, never trip coordinates or zone IDs.
    zones = zones.to_crs(2056)
    legacy_freight = label == "Freight" and not configuration["config"].get("use_npvm_freight", True)
    zones.geometry = zones.geometry.simplify(100 if legacy_freight else 15, preserve_topology=True)
    return json.loads(zones.to_crs(4326).to_json(drop_id=True))


def prepare(frame, label):
    """Keep every valid OD; geographic coordinates are only rounded for display."""
    id_column = "agent_id" if label == "Freight" else "trip_id"
    required = [id_column, *COORDINATES, "departure_time", "origin_zone", "destination_zone"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    coordinates = frame[COORDINATES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(coordinates).all(axis=1) & (coordinates > 0).all(axis=1)
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    lon_o, lat_o = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    lon_d, lat_d = transformer.transform(coordinates[:, 2], coordinates[:, 3])
    geographic = np.column_stack([lat_o, lon_o, lat_d, lon_d])
    valid &= np.isfinite(geographic).all(axis=1)
    valid &= (np.abs(geographic[:, [0, 2]]) <= 85).all(axis=1)
    valid &= (np.abs(geographic[:, [1, 3]]) <= 180).all(axis=1)
    times = pd.to_numeric(frame.departure_time, errors="coerce").to_numpy(dtype=float)
    rows = []
    for index in np.flatnonzero(valid):
        row = frame.iloc[index]
        vehicle_type = str(row.vehicle_type) if "vehicle_type" in frame else "lcv"
        rows.append([str(frame[id_column].iloc[index]), *np.round(geographic[index], 6).tolist(),
                     float(times[index]) if np.isfinite(times[index]) else None, vehicle_type,
                     str(row.origin_zone), str(row.destination_zone)])
    diagnostics = {"source_trips": len(frame), "mapped_trips": len(rows),
                   "invalid_coordinates": int((~valid).sum()),
                   "unknown_departures": int((~np.isfinite(times[valid])).sum()),
                   "coincident_endpoints": int((valid & (coordinates[:, :2] == coordinates[:, 2:]).all(axis=1)).sum())}
    if diagnostics["mapped_trips"] + diagnostics["invalid_coordinates"] != len(frame):
        raise AssertionError("OD accounting failed")
    return {"rows": rows, "diagnostics": diagnostics}


def build(configuration_path, output=None):
    configuration = yaml.safe_load(Path(configuration_path).read_text())
    frames, sources = load_trips(configuration)
    return render(frames, configuration, sources, output, Path(configuration_path).name)


def render(frames, configuration, sources, output=None, configuration_name="Synpp stage configuration"):
    # Work on a mapping copy; upstream cached tables and geometries stay read-only.
    frames = dict(frames)
    attach_zones(frames, configuration)
    data = {label: prepare(frames[label], label) for label in STAGES}
    for label in STAGES:
        data[label]["zones"] = zone_features(frames[label], label, frames, configuration)
        data[label]["zone_system"] = ("NPVM" if label == "LCV" or configuration["config"].get("use_npvm_freight", True)
                                      else "NUTS")
    meta = {"configuration": configuration_name,
            "input_downsampling": configuration["config"].get("input_downsampling"),
            "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
            "zone_assignment": ("LCV: cached NPVM IDs. Freight: aligned sampled NPVM OD counts."
                                if configuration["config"].get("use_npvm_freight", True) else
                                "LCV: cached NPVM IDs. Freight: original multinomial OD draws replayed "
                                "using cached OD inputs and configured seed; component trip alignment checked."),
            "zone_display": "Original source polygons simplified by 100 m (NUTS) / 15 m (NPVM) for display. "
                            "Zone IDs and trip coordinates are unchanged."}
    # Embed the library to keep the report usable without CDN access.
    assets = {}
    for extension in ("js", "css"):
        with urlopen(f"https://unpkg.com/leaflet@1.9.4/dist/leaflet.{extension}", timeout=30) as response:
            assets[extension] = response.read().decode("utf-8")
    template = Path(__file__).with_name("trips.html.template").read_text()
    replacements = {"__LEAFLET_CSS__": assets["css"], "__LEAFLET_JS__": assets["js"],
                    "__DATA__": json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
                    "__META__": json.dumps(meta, allow_nan=False).replace("<", "\\u003c")}
    for placeholder, content in replacements.items():
        template = template.replace(placeholder, content)
    target = Path(output) if output else Path(configuration["config"]["analysis_path"]) / "freight_lcv_od.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    print(target)
    for label, payload in data.items():
        print(label, json.dumps(payload["diagnostics"]))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", nargs="?", default="config_dib_10.yml")
    parser.add_argument("--output", type=Path, help="Override the configured analysis_path/freight_lcv_od.html")
    args = parser.parse_args()
    build(args.configuration, args.output)


if __name__ == "__main__":
    main()
