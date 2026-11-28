"""Compare PT journeys and passenger-car vehicle trips on ARE NPVM 2023 zones."""

import base64
import gzip
import hashlib
import json
import logging
from pathlib import Path
import struct
import time
from urllib.request import urlopen
from zipfile import ZipFile

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import shapely

from analysis.counts.paths import configure_simulation_path, get_simulation_path
from matsim.config.tools import resolve_output_file

LOGGER = logging.getLogger("synpp")
OD_BLOCK_ROWS = 64
REFERENCE_URL = "https://zenodo.org/records/15755932"
REPORT_URL = "https://www.are.admin.ch/dam/de/sd-web/4KquT1YIdGEZ/Aktualisierung%20NPVM%202023%20-%20Schlussbericht_Publikation_20260212.pdf"
ARCHIVE = "3_Wegematrix_OEV_NPVM_2023.zip"
ARCHIVE_MD5 = "3272fdb9442f91c62f4adfcc5b407064"
CAR_ARCHIVE = "3_Fahrtenmatrizen_Strasse_NPVM_2023.zip"
CAR_ARCHIVE_MD5 = "b4ec65b5fb5838f5e1dbef4610431995"
OPTIONS = {
    "analysis.compare_pt_to_npvm.matrix_path": "",
    "analysis.compare_pt_to_npvm.matrix_key": "1",
    "analysis.compare_pt_to_npvm.car_domestic_matrix_path": "",
    "analysis.compare_pt_to_npvm.car_domestic_matrix_key": "20",
    "analysis.compare_pt_to_npvm.car_external_matrix_path": "",
    "analysis.compare_pt_to_npvm.car_external_matrix_key": "7",
    "analysis.compare_pt_to_npvm.foreign_zones_path": "npvm/Verkehrszonen_Ausland_NPVM_2023.gpkg",
    "analysis.compare_pt_to_npvm.zones_path": "npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg",
}


def configure(context):
    configure_simulation_path(context)
    context.config("data_path")
    context.config("analysis_path")
    context.config("input_downsampling")
    for key, default in OPTIONS.items():
        context.config(key, default=default)


def execute(context):

    # Locate existing outputs and the separate PT and passenger-car reference files.

    data = Path(context.config("data_path"))
    simulation = Path(get_simulation_path(context))
    output = simulation / "compare_pt_to_npvm"
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(context.config("analysis_path")) / "npvm_reference"
    configured = context.config("analysis.compare_pt_to_npvm.matrix_path")
    pt_path = data / configured if configured else download_reference(reference)[0]
    car_options = [context.config(f"analysis.compare_pt_to_npvm.car_{kind}_matrix_path")
                   for kind in ["domestic", "external"]]
    car_defaults = download_reference(reference, "car") if not all(car_options) else []
    car_paths = [data / value if value else car_defaults[i] for i, value in enumerate(car_options)]
    paths = {"pt": [(pt_path, context.config("analysis.compare_pt_to_npvm.matrix_key"))],
             "car": [(path, context.config(f"analysis.compare_pt_to_npvm.car_{kind}_matrix_key"))
                     for kind, path in zip(["domestic", "external"], car_paths)]}
    trips = Path(resolve_output_file(simulation / "output_trips.csv"))
    fraction = float(context.config("input_downsampling"))
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("input_downsampling must be in (0, 1]")

    # Compare the two modes using the same NPVM zone polygons and municipality crosswalk.

    zones = read_zones(data / context.config("analysis.compare_pt_to_npvm.zones_path"), pt_path,
                       data / context.config("analysis.compare_pt_to_npvm.foreign_zones_path"))
    modes = {}
    metadata = {
        "schema": "npvm-crossborder-v5", "reference_url": REFERENCE_URL,
        "reference_report": REPORT_URL, "reference_year": 2023,
        "reference_period": "Average weekday (DWV), full day",
        "scope": "Domestic-only when the checkbox is off; domestic plus incoming/outgoing cross-border trips when on. At least one endpoint must be in the domestic NPVM area (including Liechtenstein/enclaves). All nationalities included; foreign-to-foreign trips and unmapped endpoints are excluded and audited.",
        "zone_count": len(zones), "input_downsampling": fraction,
        "simulation_expansion": 1 / fraction, "trips_file": str(trips), "modes": {},
        "municipality_basis": "Domestic municipalities are unions of their NPVM zones. Foreign areas retain their original NPVM zoning at both levels: they are not municipality estimates. Both levels retain the same trips.",
        "notes": "No adjustment between reference and population target years. Same-area trips included by default. OD lines are schematic, not routed. Light commercial vehicles and freight are excluded.",
    }
    definitions = {
        "pt": "Combined ÖV matrix: train, bus, tram, metro and other modelled PT systems. One person journey per linked trip, including transfers only once.",
        "car": "Passenger-car vehicles: PW Binnen plus PW QZD only. No LI (vans/LCV), LW or LZ (freight) matrices. Simulation uses car/car_loop drivers only; passengers, truck modes, freight and LCV agents/activities are excluded.",
    }
    for mode in ["pt", "car"]:
        components = [read_npvm(path, key, zones, mode=mode) for path, key in paths[mode]]
        npvm = sum((matrix for matrix, _ in components), sparse.csr_matrix((len(zones), len(zones))))
        raw_mapped_total = float(npvm.sum())
        npvm = filter_scope(npvm, zones.is_domestic.to_numpy(), include_crossborder=True)
        domestic_total = float(filter_scope(npvm, zones.is_domestic.to_numpy(), include_crossborder=False).sum())
        reference_audit = {
            "all_matrix_journeys": sum(audit["all_matrix_journeys"] for _, audit in components),
            "excluded_external_or_unmapped_journeys": sum(audit["excluded_external_or_unmapped_journeys"] for _, audit in components),
            "foreign_to_foreign_excluded_journeys": raw_mapped_total - float(npvm.sum()),
            "domestic_journeys": domestic_total,
            "crossborder_journeys": float(npvm.sum()) - domestic_total,
            "components": [{"path": str(path), "key": key, **audit}
                           for (path, key), (_, audit) in zip(paths[mode], components)],
        }
        simulated, simulation_audit = aggregate_simulation(trips, zones, fraction, mode=mode)
        municipalities, municipality_npvm, municipality_simulated = aggregate_municipalities(zones, npvm, simulated)
        modes[mode] = {
            "zones": (zones, npvm, simulated, zone_summary(npvm, simulated, zones)),
            "municipalities": (municipalities, municipality_npvm, municipality_simulated,
                               zone_summary(municipality_npvm, municipality_simulated, municipalities)),
        }
        metadata["modes"][mode] = {
            "definition": definitions[mode], "unit": "Person journeys" if mode == "pt" else "Vehicle trips",
            "npvm": reference_audit, "simulation": simulation_audit,
            "npvm_compared_journeys": float(npvm.sum()), "simulated_compared_journeys": float(simulated.sum()),
        }
        LOGGER.info("%s: %.0f reference, %.0f simulated full-population trips", mode.upper(), npvm.sum(), simulated.sum())
    metadata["municipality_count"] = int(municipalities.is_domestic.sum())
    metadata["foreign_zone_count"] = int((~zones.is_domestic).sum())
    metadata["domestic_zone_count"] = int(zones.is_domestic.sum())
    metadata["unmapped_note"] = "Some OMX zones have no published polygon. Their demand is excluded, rather than assigning simulation trips to invented catchments; exclusion totals are reported for each mode/component."

    # Save matrices and totals at both resolutions, then embed both modes in one report.

    for mode, levels in modes.items():
        for level, (areas, reference_matrix, simulated_matrix, totals) in levels.items():
            prefix = "zone" if level == "zones" else "municipality"
            # Keep the existing PT filenames; car demand has its own explicit prefix.
            tag = "" if mode == "pt" else "car_"
            sparse.save_npz(output / f"{tag}npvm_{prefix}_od.npz", reference_matrix)
            sparse.save_npz(output / f"{tag}simulated_{prefix}_od.npz", simulated_matrix)
            totals.to_csv(output / f"{tag}{prefix}_summary.csv", index=False)
            if mode == "pt":
                areas[["id", "name", "municipality_id", "municipality_name", "is_domestic"]].to_csv(output / f"{prefix}_index.csv", index=False)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_html(output / "index.html", modes, metadata)
    LOGGER.info("PT and car demand comparison: %s", output / "index.html")
    return str(output / "index.html")


def download_reference(directory, mode="pt"):
    """Extract only ÖV or PW demand into private analysis storage; shared inputs stay read-only."""
    directory.mkdir(parents=True, exist_ok=True)
    archive_name, checksum, members = ((ARCHIVE, ARCHIVE_MD5, ["OEV.omx"]) if mode == "pt"
                                       else (CAR_ARCHIVE, CAR_ARCHIVE_MD5, ["PW_Binnen.omx", "PW_QZD.omx"]))
    targets = [directory / name for name in members]
    if all(path.exists() for path in targets):
        return targets
    archive = directory / archive_name
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        with urlopen(f"https://zenodo.org/api/records/15755932/files/{archive_name}/content", timeout=120) as source, temporary.open("wb") as dest:
            while block := source.read(1024 * 1024):
                dest.write(block)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != checksum:
            raise ValueError(f"NPVM download checksum mismatch: {archive}")
    with ZipFile(archive) as contents:
        for member, target in zip(members, targets):
            if target.exists():
                continue
            temporary = target.with_suffix(".omx.part")
            with contents.open(member) as source, temporary.open("wb") as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)
            temporary.replace(target)
    return targets


def read_zones(path, matrix, foreign_path=None):
    frame = gpd.read_file(path).to_crs(2056)
    with h5py.File(matrix) as source:
        lookup = source["lookup/NO"][:]
    frame = frame.loc[frame.No.isin(lookup), ["No", "ID_Gem", "N_Gem", "geometry"]].copy()
    frame = frame.rename(columns={"No": "id", "ID_Gem": "municipality_id", "N_Gem": "municipality_name"})
    frame["id"] = frame.id.astype("int64")
    frame.geometry = shapely.force_2d(frame.geometry)
    frame = frame.dissolve(by="id", aggfunc="first").reset_index().sort_values("id").reset_index(drop=True)
    frame["name"] = frame.municipality_name.fillna("NPVM zone") + " · " + frame.id.astype(str)
    if frame.empty:
        raise ValueError("No supplied NPVM zone polygons match the OMX zone lookup")
    frame["is_domestic"] = True
    if foreign_path is not None:
        foreign = gpd.read_file(foreign_path).to_crs(2056)
        foreign = foreign.loc[foreign.ID_Zone.isin(lookup) & ~foreign.ID_Zone.isin(frame.id)].copy()
        foreign["id"] = foreign.ID_Zone.astype("int64")
        foreign["municipality_id"] = foreign.id
        label = foreign.Name.fillna(foreign.NUTS3).fillna(foreign.NUTS2).fillna("Foreign NPVM zone")
        foreign["municipality_name"] = foreign.NUTS2.fillna("").str[:2] + " · " + label
        foreign["name"] = foreign.municipality_name + " · " + foreign.id.astype(str)
        foreign["is_domestic"] = False
        foreign.geometry = shapely.force_2d(foreign.geometry)
        columns = ["id", "name", "municipality_id", "municipality_name", "is_domestic", "geometry"]
        frame = gpd.GeoDataFrame(pd.concat([frame[columns], foreign[columns]], ignore_index=True), crs=2056)
    return frame.sort_values("id").reset_index(drop=True)


def read_npvm(path, key, zones, mode="pt"):
    """Read ÖV or strictly PW demand with its own explicit OMX zone lookup."""
    parts = []
    total = 0.0
    roundoff_negative_cells = 0
    with h5py.File(path) as source:
        lookup = np.asarray(source["lookup/NO"])
        if len(np.unique(lookup)) != len(lookup):
            raise ValueError("NPVM matrix lookup contains duplicate zone IDs")
        positions = pd.Series(np.arange(len(lookup)), index=lookup).reindex(zones.id).to_numpy()
        if pd.isna(positions).any():
            raise ValueError("A mapped zone is absent from the NPVM matrix lookup")
        positions = positions.astype(int)
        matrix = source[f"data/{key}"]
        code = matrix.attrs.get("CODE", b"")
        if isinstance(code, bytes):
            code = code.decode("utf-8")
        if mode == "car" and not (str(code).casefold() == "pw" or str(code).casefold().startswith(("pw binnen", "pw qzd"))):
            raise ValueError(f"Expected passenger-car PW demand, got CODE={code!r}; LI/LW/LZ are excluded")
        if mode == "pt" and code and str(code).casefold() not in {"öv", "oev"}:
            raise ValueError(f"Expected the combined ÖV PT matrix, got CODE={code!r}")
        if matrix.shape != (len(lookup), len(lookup)):
            raise ValueError("NPVM matrix shape does not match its zone lookup")
        # Visit all rows to audit excluded demand, then re-order both axes by zone ID.
        for start in range(0, len(lookup), 256):
            block = np.asarray(matrix[start:start + 256], dtype=float)
            if not np.isfinite(block).all() or (block < -1e-12).any():
                raise ValueError("NPVM demand must be finite and nonnegative")
            roundoff_negative_cells += int((block < 0).sum())
            # The official PW matrix contains numerical zeros around -1e-16.
            np.maximum(block, 0, out=block)
            total += block.sum()
        for start in range(0, len(positions), 128):
            selected = positions[start:start + 128]
            order = np.argsort(selected)
            block = np.asarray(matrix[selected[order], :], dtype=float)[np.argsort(order)][:, positions]
            np.maximum(block, 0, out=block)
            parts.append(sparse.csr_matrix(block))
        sourcefile = source.attrs.get("SOURCEFILE", b"")
        if isinstance(sourcefile, bytes):
            sourcefile = sourcefile.decode("utf-8")
    result = sparse.vstack(parts, format="csr")
    audit = {"all_matrix_journeys": float(total),
             "excluded_external_or_unmapped_journeys": float(total - result.sum()),
             "unmapped_zones": int(len(lookup) - len(zones)),
             "sourcefile": str(sourcefile), "matrix_code": str(code),
             "nonzero_od_pairs": result.nnz, "roundoff_negative_cells_zeroed": roundoff_negative_cells}
    return result, audit


def assign_zones(x, y, zones):
    """Resolve exact boundary ties by zone ID; never snap endpoints across a border."""
    coordinates = pd.DataFrame({"x": x, "y": y}).drop_duplicates().reset_index(drop=True)
    valid = np.isfinite(coordinates.x) & np.isfinite(coordinates.y)
    points = gpd.GeoDataFrame(coordinates.loc[valid], geometry=gpd.points_from_xy(coordinates.loc[valid, "x"], coordinates.loc[valid, "y"]), crs=2056)
    areas = zones[["id", "geometry"]].copy()
    areas["is_domestic"] = zones.is_domestic.to_numpy() if "is_domestic" in zones else True
    matches = gpd.sjoin(points, areas, how="left", predicate="intersects")
    matches = matches.sort_values(["is_domestic", "id"], ascending=[False, True]).loc[lambda f: ~f.index.duplicated()]
    coordinates["id"] = matches.id.reindex(coordinates.index)
    return coordinates


def aggregate_simulation(path, zones, fraction, mode="pt"):
    parts = []
    audit = {"all_trips": 0, f"{mode}_trips_sample": 0, "excluded_lcv_or_freight_sample": 0,
             "excluded_external_or_unmapped_sample": 0}
    columns = ["person", "main_mode", "start_x", "start_y", "end_x", "end_y",
               "start_activity_type", "end_activity_type"]
    for chunk in pd.read_csv(path, sep=";", usecols=lambda column: column in columns,
                             dtype={"person": "str"}, chunksize=500_000):
        audit["all_trips"] += len(chunk)
        main_mode = chunk.main_mode.astype(str).str.removesuffix("_loop")
        excluded = pd.Series(False, index=chunk.index)
        if "person" in chunk:
            excluded |= chunk.person.str.startswith(("lcv_", "freight_"), na=False)
        for column in ["start_activity_type", "end_activity_type"]:
            if column in chunk:
                excluded |= chunk[column].astype(str).str.startswith("freight")
        if mode == "car" and not any(column in chunk for column in ["person", "start_activity_type", "end_activity_type"]):
            raise ValueError("Car comparison needs person IDs or activity types to exclude LCV and freight")
        audit["excluded_lcv_or_freight_sample"] += int((main_mode.eq(mode) & excluded).sum())
        pt = chunk.loc[main_mode.eq(mode) & ~excluded].copy()
        audit[f"{mode}_trips_sample"] += len(pt)
        if pt.empty:
            continue
        for endpoint, prefix in [("origin", "start"), ("destination", "end")]:
            matched = assign_zones(pt[f"{prefix}_x"], pt[f"{prefix}_y"], zones)
            pt = pt.merge(matched.rename(columns={"x": f"{prefix}_x", "y": f"{prefix}_y", "id": endpoint}), how="left", on=[f"{prefix}_x", f"{prefix}_y"], validate="many_to_one")
        covered = pt.origin.notna() & pt.destination.notna()
        audit["excluded_external_or_unmapped_sample"] += int((~covered).sum())
        parts.append(pt.loc[covered].groupby(["origin", "destination"]).size().rename("sample"))
    shape = (len(zones), len(zones))
    if not parts:
        return sparse.csr_matrix(shape, dtype=float), audit
    od = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    positions = pd.Series(np.arange(len(zones)), index=zones.id)
    result = sparse.coo_matrix((od["sample"].to_numpy() / fraction,
                               (od.origin.map(positions), od.destination.map(positions))), shape=shape).tocsr()
    if "is_domestic" in zones:
        raw_total = float(result.sum())
        result = filter_scope(result, zones.is_domestic.to_numpy(), include_crossborder=True)
        domestic_total = float(filter_scope(result, zones.is_domestic.to_numpy(), include_crossborder=False).sum())
        audit["foreign_to_foreign_excluded_sample"] = int(round((raw_total - result.sum()) * fraction))
        audit["domestic_sample"] = int(round(domestic_total * fraction))
        audit["crossborder_sample"] = int(round((result.sum() - domestic_total) * fraction))
    return result, audit


def filter_scope(matrix, is_domestic, include_crossborder):
    """Apply identical OD endpoint rules: domestic/domestic, optionally foreign/domestic."""
    result = matrix.copy().tocsr()
    origin_domestic = np.repeat(np.asarray(is_domestic, dtype=bool), np.diff(result.indptr))
    destination_domestic = np.asarray(is_domestic, dtype=bool)[result.indices]
    keep = origin_domestic | destination_domestic if include_crossborder else origin_domestic & destination_domestic
    result.data[~keep] = 0
    result.eliminate_zeros()
    return result


def zone_summary(npvm, simulated, zones):
    summary = zones[["id", "name", "municipality_id", "municipality_name"]].copy()
    summary["is_domestic"] = zones.is_domestic.to_numpy() if "is_domestic" in zones else True
    for model, matrix in [("npvm", npvm), ("simulated", simulated)]:
        domestic = filter_scope(matrix, summary.is_domestic.to_numpy(), include_crossborder=False)
        summary[f"{model}_domestic_origins"] = np.asarray(domestic.sum(axis=1)).ravel()
        summary[f"{model}_domestic_destinations"] = np.asarray(domestic.sum(axis=0)).ravel()
        summary[f"{model}_domestic_intrazonal"] = domestic.diagonal()
        summary[f"{model}_origins"] = np.asarray(matrix.sum(axis=1)).ravel()
        summary[f"{model}_destinations"] = np.asarray(matrix.sum(axis=0)).ravel()
        summary[f"{model}_intrazonal"] = matrix.diagonal()
    return summary


def aggregate_municipalities(zones, npvm, simulated):
    """Sum both OD axes by official municipality ID, preserving all retained journeys."""
    if zones.municipality_id.isna().any():
        raise ValueError("Every NPVM zone needs a municipality ID for the zoning switch")
    areas = zones[["municipality_id", "municipality_name", "geometry"]].copy()
    areas["is_domestic"] = zones.is_domestic.to_numpy() if "is_domestic" in zones else True
    municipalities = areas.dissolve(
        by="municipality_id", aggfunc="first").reset_index()
    municipalities["id"] = municipalities.municipality_id.astype("int64")
    municipalities = municipalities.sort_values("id").reset_index(drop=True)
    municipalities["name"] = municipalities.municipality_name + " · " + municipalities.id.astype(str)
    positions = pd.Series(np.arange(len(municipalities)), index=municipalities.id)
    mapping = sparse.coo_matrix((np.ones(len(zones)),
                                 (np.arange(len(zones)), zones.municipality_id.map(positions))),
                                shape=(len(zones), len(municipalities))).tocsr()
    aggregated = [(mapping.T @ (matrix @ mapping)).tocsr() for matrix in (npvm, simulated)]
    return municipalities, *aggregated


def pack_matrix(matrix):
    """Compact browser CSR: version, size, nnz, index width; rows, columns, values."""
    if max(matrix.shape) >= 65536:
        raise ValueError("The compact NPVM map supports fewer than 65,536 zones")
    matrix = matrix.copy()
    matrix.sort_indices()
    columns = matrix.indices.astype("<u2").tobytes()
    padding = b"\0" * (-len(columns) % 4)
    binary = (struct.pack("<4I", 1, matrix.shape[0], matrix.nnz, 2)
              + matrix.indptr.astype("<u4").tobytes() + columns + padding
              + matrix.data.astype("<f4").tobytes())
    return base64.b64encode(gzip.compress(binary, compresslevel=6)).decode("ascii")


def write_html(path, modes, metadata):
    features = {}
    for level, (areas, _, _, _) in modes["pt"].items():
        display = areas.copy()
        for mode, levels in modes.items():
            summary = levels[level][3].drop(columns=["name", "municipality_id", "municipality_name", "is_domestic"])
            summary = summary.rename(columns={column: f"{mode}_{column}" for column in summary if column != "id"})
            display = display.merge(summary, on="id", validate="one_to_one")
        anchors = gpd.GeoSeries(display.geometry.representative_point(), crs=display.crs).to_crs(4326)
        display["lat"] = anchors.y
        display["lon"] = anchors.x
        display.geometry = display.geometry.simplify(20, preserve_topology=True)
        features[level] = json.loads(display.to_crs(4326).to_json(drop_id=True))
    for mode, levels in modes.items():
        for level, (_, npvm, simulated, _) in levels.items():
            write_od_assets(path, mode, level, npvm, simulated)
    write_report(path, features, metadata)


def write_od_assets(path, mode, level, npvm, simulated):
    """Store precomputed outgoing/incoming rows in small browser-ready companion files."""
    directory = path.parent / "assets" / "od"
    directory.mkdir(parents=True, exist_ok=True)
    for direction in ["destinations", "origins"]:
        reference = npvm if direction == "destinations" else npvm.T.tocsr()
        simulation = simulated if direction == "destinations" else simulated.T.tocsr()
        for start in range(0, reference.shape[0], OD_BLOCK_ROWS):
            block = start // OD_BLOCK_ROWS
            key = f"{mode}_{level}_{direction}_{block}"
            data = {"npvm": pack_matrix(reference[start:start + OD_BLOCK_ROWS]),
                    "simulated": pack_matrix(simulation[start:start + OD_BLOCK_ROWS])}
            target = directory / f"{key}.js"
            temporary = target.with_suffix(".js.part")
            temporary.write_text(
                f"window.npvmODChunks[{json.dumps(key)}]={json.dumps(data, separators=(',', ':'))};\n",
                encoding="utf-8")
            temporary.replace(target)


def write_report(path, features, metadata):
    """Render only precomputed map features and totals; no full OD matrices in the HTML."""
    for collection in features.values():
        for feature in collection["features"]:
            feature["geometry"]["coordinates"] = round_map_coordinates(feature["geometry"]["coordinates"])
    data = json.dumps({"levels": features, "metadata": metadata, "od_block_rows": OD_BLOCK_ROWS,
                       "asset_version": str(time.time_ns())}, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(data, compresslevel=6)
    payload = base64.b64encode(compressed).decode("ascii")
    template = Path(__file__).with_name("map.html.template").read_text(encoding="utf-8")
    assets = path.parent / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "map_data.json.gz").write_bytes(compressed)
    for extension in ["css", "js"]:
        asset = assets / f"leaflet.{extension}"
        if not asset.exists():
            with urlopen(f"https://unpkg.com/leaflet@1.9.4/dist/leaflet.{extension}", timeout=60) as response:
                asset.write_bytes(response.read())
        template = template.replace(f"__LEAFLET_{extension.upper()}__", asset.read_text(encoding="utf-8"))
    temporary = path.with_suffix(".html.part")
    temporary.write_text(template.replace("__PAYLOAD__", payload)
                         + "<script>boot().catch(showError);</script></body></html>", encoding="utf-8")
    temporary.replace(path)


def round_map_coordinates(values):
    """Display coordinates to about one metre; analysis geometries are unchanged."""
    if not values:
        return values
    if isinstance(values[0], (int, float)):
        return [round(value, 5) for value in values]
    return [round_map_coordinates(value) for value in values]


def validate(context):
    """Rebuild after input replacement or a change to the map template."""
    files = [Path(resolve_output_file(Path(get_simulation_path(context)) / "output_trips.csv")),
             Path(context.config("data_path")) / context.config("analysis.compare_pt_to_npvm.zones_path"),
             Path(context.config("data_path")) / context.config("analysis.compare_pt_to_npvm.foreign_zones_path"),
             Path(__file__).with_name("map.html.template")]
    for option in ["matrix_path", "car_domestic_matrix_path", "car_external_matrix_path"]:
        configured = context.config(f"analysis.compare_pt_to_npvm.{option}")
        if configured:
            files.append(Path(context.config("data_path")) / configured)
    return ["npvm-display-v6", *[[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in files]]
