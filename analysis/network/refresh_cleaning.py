"""Refresh an existing inspector from a verified cleaning replay without XML import."""
import argparse
from contextlib import closing
import json
import logging
from pathlib import Path
import shutil
import sqlite3

from .cleaning_preprocess import prepare_removals
from .cleaning_tiles import build_visual_index


def refresh(directory, run):
    directory, run = Path(directory).resolve(), Path(run).resolve()
    verification = json.loads((run / "verification.json").read_text())
    if not verification.get("verified") or verification["compared_database"] != str(directory / "network.sqlite"):
        raise ValueError("Replay must be verified against this inspector database")
    logging.info("Importing verified per-link removal reasons")
    audit = prepare_removals(directory, run / "cleaning_events.jsonl.gz")
    logging.info("Preparing directional offsets and disconnected-link circles")
    drawing = build_visual_index(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest.update(audit=audit, audit_available=True, display_revision=str(drawing.stat().st_mtime_ns))
    temporary = directory / "manifest.refresh.json"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(directory / "manifest.json")
    shutil.copyfile(Path(__file__).with_name("cleaning_viewer.html"), directory / "index.html")
    with closing(sqlite3.connect(drawing.as_uri()+"?mode=ro&immutable=1", uri=True)) as connection:
        offsets = int(connection.execute("SELECT value FROM display_metadata WHERE name='offset_links'").fetchone()[0])
    result = {"audit": audit, "offset_links": offsets, "directory": str(directory)}
    (run / "viewer_refresh.json").write_text(json.dumps(result, indent=2))
    logging.info("Inspector ready: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    refresh(args.directory, args.run)
