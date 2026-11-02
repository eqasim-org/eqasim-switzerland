"""Read unchanged official BFS projection CSV exports for offline scaling."""

import hashlib
import json
from pathlib import Path

import pandas as pd

RAW_DIRECTORY = Path("projections/raw")
POPULATION_TABLE = "px-x-0104020000_101"
HOUSEHOLD_FILE = "DF_SCEN_MENAGES.csv"
POPULATION_COLUMNS = ["Kanton", "Staatsangehörigkeit (Kategorie)", "Geschlecht",
                      "Alter", "Jahr", "Bevölkerungsstand am 31. Dezember"]
HOUSEHOLD_COLUMNS = ["DATAFLOW", "SCENARIO", "CANTON", "SIZE_HOUSEHOLD", "NUMBER_HOUSEHOLD_PERSON",
                     "FREQ", "TIME_HORIZ", "TIME_PERIOD", "OBS_VALUE"]


def configure(context):
    context.config("data_path")


def execute(context):

    # Decode the official CSV formats without modifying the external inputs.

    root = Path(context.config("data_path")) / RAW_DIRECTORY
    population_files, household_file = input_files(root)
    population = read_population(population_files)
    households = read_households(household_file)

    # Keep inspectable normalized tables in this stage's cache, not in data_path.

    population.to_csv(Path(context.path()) / "population.csv", index=False)
    households.to_csv(Path(context.path()) / "households.csv", index=False)

    # Downstream projection stages select the target year and build the margins.

    return {"population": population, "households": households}


def validate(context):
    root = Path(context.config("data_path")) / RAW_DIRECTORY
    population_files, household_file = input_files(root)
    versions = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [*population_files, household_file]}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in versions.items():
            if manifest.get("files", {}).get(name, {}).get("sha256") != digest:
                raise ValueError("Raw BFS export differs from its download manifest: %s" % name)
        versions[manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return versions


def input_files(root):
    # One export or non-overlapping year chunks are accepted; never use the old
    # hand-prepared CSV as an implicit fallback when raw sources are absent.
    manifest = root / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("status") != "complete":
        raise ValueError("Raw BFS projection download is incomplete: %s" % root)
    population = sorted(root.glob(POPULATION_TABLE + "*.csv"))
    households = root / HOUSEHOLD_FILE
    if not population or not households.is_file():
        raise FileNotFoundError(
            "Raw BFS projections missing under %s; run python -m "
            "data.statpop.projections.download --output-path %s" % (root, root))
    return population, households


def require_columns(frame, columns, path):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("BFS projection export %s is missing columns: %s" % (path, missing))


def read_population(paths):
    frames = []
    for path in paths:
        # PXWeb's official CSV response is comma-delimited Windows-1252. Its
        # year-end measure is a column header, so a January-only export fails.
        frame = pd.read_csv(path, encoding="cp1252")
        require_columns(frame, POPULATION_COLUMNS, path)
        frames.append(frame[POPULATION_COLUMNS])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(POPULATION_COLUMNS[:-1]).any():
        raise ValueError("Overlapping or duplicate population cells in raw BFS exports")
    return result


def read_households(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"CANTON": str})
    require_columns(frame, HOUSEHOLD_COLUMNS, path)
    if not frame["DATAFLOW"].eq("CH1.SCEN:DF_SCEN_MENAGES(1.0.0)").all():
        raise ValueError("Unexpected BFS household dataflow; review the source edition")
    return frame
