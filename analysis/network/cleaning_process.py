"""Build a spatially indexed before/after network-cleaning inspector."""
from pathlib import Path
from contextlib import closing
import argparse
import hashlib
import json
import logging

from .cleaning_data import build_database
from .cleaning_assets import build_assets
from .cleaning_preprocess import prepare_removals

logger = logging.getLogger(__name__)


def configure(context):
    context.stage("matsim.scenario.network.convert_osm_pt2matsim")
    context.stage("matsim.scenario.network.convert_osm")
    context.config("analysis_path")
    context.config("analysis.network.cleaning_process.center", default=[46.252196, 6.138081])
    context.config("analysis.network.cleaning_process.zoom", default=13)
    context.config("analysis.network.cleaning_process.offline_radius_m", default=8000)
    context.config("analysis.network.cleaning_process.max_features", default=10000)


def execute(context):

    # Resolve each network together with the geometry exported by its producer.

    before, before_geometry = context.stage("matsim.scenario.network.convert_osm_pt2matsim")
    after = context.stage("matsim.scenario.network.convert_osm")
    after_geometry = Path(context.path("matsim.scenario.network.convert_osm")) / "detailed_network.csv"
    audit = Path(context.path("matsim.scenario.network.convert_osm")) / "cleaning_events.jsonl.gz"

    # Export an analysis-owned database and a viewer, leaving source caches intact.

    output = Path(context.config("analysis_path")) / "network" / "cleaning_process"
    return build_inspector(
        before, before_geometry, after, after_geometry, output,
        center=context.config("analysis.network.cleaning_process.center"),
        zoom=context.config("analysis.network.cleaning_process.zoom"),
        offline_radius=context.config("analysis.network.cleaning_process.offline_radius_m"),
        max_features=context.config("analysis.network.cleaning_process.max_features"),
        audit=audit if audit.exists() else None,
    )


def validate(context):
    digest = hashlib.sha256()
    for name in ("cleaning_data.py", "cleaning_server.py", "cleaning_tiles.py", "cleaning_preprocess.py", "cleaning_assets.py", "cleaning_viewer.html"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def build_inspector(before, before_geometry, after, after_geometry, output, *,
                    center=(46.252196, 6.138081), zoom=13, offline_radius=8000, max_features=10000, audit=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {key: Path(value).resolve() for key, value in {
        "before": before, "before_geometry": before_geometry,
        "after": after, "after_geometry": after_geometry,
    }.items()}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        if output == path.parent or output in path.parents:
            raise ValueError("Analysis output must not contain upstream source files")
    if len(center) != 2 or not (-85 < center[0] < 85 and -180 <= center[1] <= 180):
        raise ValueError("center must be [latitude, longitude]")
    if not 1 <= max_features <= 50000 or not 0 <= offline_radius <= 20000:
        raise ValueError("max_features must be 1..50000; offline_radius_m must be 0..20000")
    source_info = {key: {"path": str(path), "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns} for key, path in sources.items()}
    fingerprint = {"sources": source_info, "builder": hashlib.sha256(
        Path(__file__).with_name("cleaning_data.py").read_bytes()).hexdigest()}
    database = output / "network.sqlite"
    metadata_path = output / "database_sources.json"
    reuse = database.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == fingerprint
    if not reuse:
        summary = build_database(sources, database)
        metadata_path.write_text(json.dumps(fingerprint, indent=2))
    else:
        import sqlite3
        with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
            summary = json.loads(connection.execute("SELECT value FROM metadata WHERE name='summary'").fetchone()[0])
        logger.info("Reusing spatial database: source paths, sizes and timestamps match")
    assets = build_assets(output, center, offline_radius)
    if audit is not None:
        prepare_removals(output, audit)
    else:
        (output / "cleaning_audit.sqlite").unlink(missing_ok=True)
    from .cleaning_tiles import build_visual_index
    build_visual_index(output)
    manifest = {"summary": summary, "sources": source_info, "center": list(center), "zoom": zoom,
                "max_features": max_features, "background": assets, "schema": 1}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "index.html").write_text(Path(__file__).with_name("cleaning_viewer.html").read_text())
    command = f"uv run -m analysis.network.cleaning_server '{output}' --port 8765"
    (output / "OPEN_VIEWER.txt").write_text(
        f"From the eqasim-switzerland repository:\n{command}\n\nOpen http://127.0.0.1:8765/\n"
        "For a remote cluster session, forward port 8765 or use your Jupyter /proxy/8765/ URL.\n"
        "The viewer requires this local server to query SQLite; opening index.html directly shows instructions.\n"
        "It does not run Synpp, alter networks, route trips, or start a simulation.\n")
    logger.info("Network cleaning inspector ready: %s", output)
    return {"html": str(output / "index.html"), "database": str(database), "directory": str(output)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "before-geometry", "after", "after-geometry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--offline-radius", type=float, default=8000)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_inspector(args.before, args.before_geometry, args.after,
                                    args.after_geometry, args.output, offline_radius=args.offline_radius, audit=args.audit), indent=2))
