"""Interactive French residents and trip explorer, runnable independently of synthesis."""

import argparse
import base64
import gzip
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import from_wkt, get_x, get_y


logger = logging.getLogger("synpp")


def configure(context):
    context.config("external_population_folder")
    context.config("analysis_path")
    context.config("random_seed")
    context.config("analysis.french_activities.max_persons", default=0)


def execute(context):

    # Read the configured French source population, without synthesis dependencies.

    payload = prepare_data(context.config("external_population_folder"),
                           context.config("analysis.french_activities.max_persons"),
                           context.config("random_seed"))

    # Export a single HTML with embedded data; only map tiles/libraries need internet.

    output = Path(context.config("analysis_path")) / "french_activities" / "french_activities.html"
    write_html(payload, output)

    # Return the exported explorer path.

    return str(output)


def source_file(folder, suffix):
    matches = sorted(Path(folder).glob("*" + suffix))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one *{suffix} in {folder}, found {len(matches)}")
    return matches[0]


def prepare_data(folder, max_persons=0, seed=0):
    if not isinstance(max_persons, int) or max_persons < 0:
        raise ValueError("analysis.french_activities.max_persons must be a non-negative integer (0 = all)")
    persons = pd.read_csv(source_file(folder, "_persons.csv"), sep=";", dtype=str, keep_default_na=False)
    attribute_names = [column for column in persons if column != "person_id"]
    if persons.person_id.duplicated().any():
        raise ValueError("Duplicate person IDs in French population")
    original_count = len(persons)
    if max_persons and len(persons) > max_persons:
        persons = persons.sample(n=max_persons, random_state=seed).sort_index()
    homes = gpd.read_file(source_file(folder, "_homes.gpkg"))
    if homes.crs is None:
        raise ValueError("French home layer has no CRS")
    homes["household_id"] = homes.household_id.astype(str)
    homes = homes.to_crs(4326)
    homes["lon"], homes["lat"] = homes.geometry.x, homes.geometry.y
    persons = persons.merge(homes[["household_id", "lon", "lat"]], on="household_id", how="left", validate="many_to_one")
    valid = np.isfinite(persons[["lon", "lat"]]).all(axis=1)
    missing_homes = int((~valid).sum())
    persons = persons[valid].reset_index(drop=True)
    person_index = dict(zip(persons.person_id, persons.index))
    if persons.empty:
        raise ValueError("No persons with valid homes")
    logger.info("French explorer: %d persons; reading activity locations", len(persons))

    # The source importer defines activity WKT as EPSG:2154 (Lambert-93).
    transformer = Transformer.from_crs(2154, 4326, always_xy=True)
    parts = []
    for chunk in pd.read_csv(source_file(folder, "_activities.csv"), sep=";",
                             usecols=["person_id", "activity_index", "geometry"],
                             dtype={"person_id": str}, chunksize=200_000):
        chunk = chunk[chunk.person_id.isin(person_index)].copy()
        geometry = from_wkt(chunk.pop("geometry").to_numpy(), on_invalid="ignore")
        chunk["lon"], chunk["lat"] = transformer.transform(get_x(geometry), get_y(geometry))
        parts.append(chunk)
    activities = pd.concat(parts, ignore_index=True)
    activities = activities.drop_duplicates(["person_id", "activity_index", "lon", "lat"])
    if activities.duplicated(["person_id", "activity_index"]).any():
        raise ValueError("Duplicate person/activity indices")
    origins = activities.rename(columns={"activity_index": "preceding_activity_index", "lon": "ox", "lat": "oy"})
    destinations = activities.rename(columns={"activity_index": "following_activity_index", "lon": "dx", "lat": "dy"})
    trips, skipped = [], 0
    modes, purposes, origin_purposes = {}, {}, {}
    for chunk in pd.read_csv(source_file(folder, "_trips.csv"), sep=";", chunksize=200_000,
                             usecols=["person_id", "trip_index", "preceding_activity_index", "following_activity_index", "mode", "preceding_purpose", "following_purpose"],
                             dtype={"person_id": str, "trip_index": str}):
        chunk = chunk[chunk.person_id.isin(person_index)].copy()
        chunk = chunk.merge(origins, on=["person_id", "preceding_activity_index"], how="left", validate="many_to_one")
        chunk = chunk.merge(destinations, on=["person_id", "following_activity_index"], how="left", validate="many_to_one")
        valid = np.isfinite(chunk[["ox", "oy", "dx", "dy"]]).all(axis=1)
        skipped += int((~valid).sum())
        for row in chunk[valid].itertuples(index=False):
            mode = str(row.mode) if pd.notna(row.mode) else "unknown"
            purpose = str(row.following_purpose) if pd.notna(row.following_purpose) else "unknown"
            origin_purpose = str(row.preceding_purpose) if pd.notna(row.preceding_purpose) else "unknown"
            modes.setdefault(mode, len(modes))
            purposes.setdefault(purpose, len(purposes))
            origin_purposes.setdefault(origin_purpose, len(origin_purposes))
            trips.append([person_index[row.person_id], round(row.ox, 6), round(row.oy, 6),
                          round(row.dx, 6), round(row.dy, 6), modes[mode], purposes[purpose],
                          origin_purposes[origin_purpose]])
        logger.info("French explorer: prepared %d trips", len(trips))
    logger.info("French explorer: %d trips; %d skipped for missing coordinates", len(trips), skipped)
    attributes = describe_attributes(persons[attribute_names])
    return {"persons": [[r.person_id, round(r.lon, 6), round(r.lat, 6)] for r in persons.itertuples()],
            "attributes": attributes, "personAttributes": persons[attribute_names].to_numpy().tolist(),
            "trips": trips, "modes": list(modes), "purposes": list(purposes),
            "originPurposes": list(origin_purposes),
            "sourcePersons": original_count, "missingHomes": missing_homes, "missingTrips": skipped,
            "sampled": bool(max_persons and original_count > max_persons)}


def describe_attributes(frame):
    """Describe filter types without recoding the raw CSV strings."""
    result = []
    for name in frame:
        values = frame[name]
        present = values[values.ne("")]
        numeric = pd.to_numeric(present, errors="coerce")
        item = {"name": name, "missing": int(values.eq("").sum())}
        if len(present) and np.isfinite(numeric).all():
            item.update(type="integer" if (numeric % 1 == 0).all() else "float",
                        minimum=float(numeric.min()), maximum=float(numeric.max()))
        elif present.nunique() <= 100:
            item.update(type="categorical", categories=sorted(present.unique().tolist()))
        else:
            item.update(type="text")
        result.append(item)
    return result


def write_html(payload, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compressed = base64.b64encode(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())).decode()
    template = Path(__file__).with_name("french_activities_map.html").read_text()
    output.write_text(template.replace("__DATA__", compressed), encoding="utf-8")
    logger.info("French activities explorer saved to %s", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--output", help="Override HTML destination")
    parser.add_argument("--max-persons", type=int, help="Optional deterministic person sample; 0 includes all")
    args = parser.parse_args()
    import yaml
    with open(args.config) as stream:
        config = yaml.safe_load(stream)["config"]
    payload = prepare_data(config["external_population_folder"],
                           args.max_persons if args.max_persons is not None else config.get("analysis.french_activities.max_persons", 0),
                           config["random_seed"])
    write_html(payload, args.output or Path(config["analysis_path"]) / "french_activities" / "french_activities.html")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
