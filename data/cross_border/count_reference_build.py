"""Rebuild reviewed G/H/I/M/N/O traffic-count references from public raw data."""

import csv
import hashlib
import json
import logging
import math
import re
import shutil
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
from openpyxl import load_workbook

from data.cross_border.reference_files import reference_directory, file_versions


SOURCES = {
    "astra": ("astra_2023.xlsx", "https://www.astra.admin.ch/dam/fr/sd-web/z6BJ81AzKCJQ/jahresergebnisse_2023.xlsx"),
    "doubs": ("doubs_comptages.csv", "https://opendata.doubs.fr/api/explore/v2.1/catalog/datasets/comptages/exports/csv"),
    "vaud": ("vaud_comptages.zip", "https://viageo.ch/donnee/telecharger/300488?format=GPKG"),
    "geneva": ("geneva_counters.json",
        "https://app2.ge.ch/tergeoservices/rest/services/Hosted/OTC_COMPTAGE_TRAFIC/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&returnGeometry=true&outSR=2056&f=pjson"),
}
for _year in (2021, 2023, 2024, 2025):
    SOURCES[f"ticino_{_year}"] = (f"Tutti_posti_{_year}.pdf",
        f"https://www4.ti.ch/fileadmin/DT/temi/conteggi_traffico/documenti/Tutti_posti_{_year}.pdf")
SOURCE_PAGES = {
    "astra": SOURCES["astra"][1],
    "doubs": "https://opendata.doubs.fr/explore/dataset/comptages/",
    "vaud": "https://viageo.ch/catalogue/donnee/300488",
    "geneva": "https://sitg.ge.ch/donnees/otc-comptage-trafic",
}
MAPPING = Path(__file__).with_name("count_reference_sources.json")
logger = logging.getLogger("synpp")


def configure(context):
    context.config("data_path")
    context.config("cross_border_reference_path", None)
    context.config("cross_border_count_raw_path", None)


def execute(context):

    # Keep a copy of every downloaded source alongside the generated reference.

    output = Path(context.path())
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    source_dir = context.config("cross_border_count_raw_path")
    if source_dir:
        source_dir = Path(source_dir).expanduser()
        if not source_dir.is_absolute():
            source_dir = Path(context.config("data_path")) / source_dir
    files = {}
    manifest = {}
    for key, (filename, url) in SOURCES.items():
        destination = raw / filename
        if source_dir:
            shutil.copyfile(source_dir / filename, destination)
        else:
            download(url, destination)
        files[key] = destination
        manifest[key] = {"url": url, "filename": filename, "sha256": sha256(destination),
                         "bytes": destination.stat().st_size}

    # Read the selected crossing release and the reviewed counter-to-FID links.

    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    crossings = gpd.read_file(root / "border_interview_places_road_and_rail.gpkg", fid_as_index=True)
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    validate_mapping(crossings, mapping)
    if crossings.crs.to_epsg() != 2056:
        raise ValueError("Crossing coordinates must use Swiss LV95 (EPSG:2056)")

    # Extract the two-direction vehicle counts without changing their original day basis.

    vaud_gpkg = raw / "vaud_comptages.gpkg"
    with zipfile.ZipFile(files["vaud"]) as archive:
        members = [member for member in archive.namelist() if member.lower().endswith(".gpkg")]
        if len(members) != 1:
            raise ValueError("Vaud download must contain one GeoPackage")
        with archive.open(members[0]) as source, vaud_gpkg.open("wb") as target:
            shutil.copyfileobj(source, target)
    astra = astra_weekday_counts(files["astra"], {item["station"] for item in mapping if item["source"] == "astra"})
    doubs = doubs_daily_counts(files["doubs"],
        {(item["road"], item["pr"], item["year"]) for item in mapping if item["source"] == "doubs"})
    vaud = vaud_weekday_counts(vaud_gpkg,
        {(item["counter"], item["start"], item["end"]) for item in mapping if item["source"] == "vaud"})
    geneva = geneva_weekday_counts(files["geneva"], [item for item in mapping if item["source"] == "geneva"])
    ticino = {}
    for key, path in files.items():
        if key.startswith("ticino_"):
            ticino.update(ticino_weekday_counts(path, [item for item in mapping if item["source"] == key]))
    references = []
    for item in mapping:
        fid = item["fid"]
        point = crossings.loc[fid].geometry
        if item["source"] == "unmeasured":
            references.append(dict(fid=fid, name=item["name"], x=point.x, y=point.y,
                                   keep_assignment=True, note=item["note"]))
            continue
        if item["source"] == "astra":
            values = astra[item["station"]]
            if item["metric"] == "annual":
                low = high = rounded_count(values["annual"])
            else:
                monthly = [rounded_count(values["months"][month]) for month in item["months"]]
                low, high = min(monthly), max(monthly)
        elif item["source"] == "doubs":
            low = high = rounded_count(doubs[item["road"], item["pr"], item["year"]])
        elif item["source"] == "vaud":
            low = high = rounded_count(vaud[item["counter"], item["start"], item["end"]])
        elif item["source"] == "geneva":
            low = high = geneva[fid]["weekday_total"]
        elif item["source"].startswith("ticino_"):
            low = high = ticino[fid]["weekday_total"]
        else:
            raise ValueError("Unknown count source: %s" % item["source"])
        if low <= 0:
            raise ValueError("Nonpositive count for crossing fid %s" % fid)
        source_url = (SOURCES[item["source"]][1] + "#page=" + str(item["page"])
                      if item["source"].startswith("ticino_") else SOURCE_PAGES[item["source"]])
        references.append(dict(fid=fid, name=item["name"], x=point.x, y=point.y,
                               low=low, high=high, period=item["period"], day_type=item["day_type"],
                               vehicle_scope="all_motor_vehicles", source_url=source_url,
                               note=item["note"]))
        if fid in ticino:
            references[-1].update(counter=item["counter"], operating_days=ticino[fid]["operating_days"])
        if fid in geneva:
            references[-1].update(directional_counts=geneva[fid]["directions"],
                                  mixed_direction_years=geneva[fid]["mixed_direction_years"])
        if "official_fid" in item:
            references[-1]["official_fid"] = item["official_fid"]

    # Save the reference and the raw-file hashes for later reproduction.

    path = output / "traffic_count_references.json"
    path.write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "raw_sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Rebuilt %d G/H/I/M/N/O count-reference records at %s", len(references), path)
    return str(path)


def validate(context):
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    paths = [root / "border_interview_places_road_and_rail.gpkg", MAPPING]
    source_dir = context.config("cross_border_count_raw_path")
    if source_dir:
        source_dir = Path(source_dir).expanduser()
        if not source_dir.is_absolute():
            source_dir = Path(context.config("data_path")) / source_dir
        paths += [source_dir / filename for filename, _ in SOURCES.values()]
    return file_versions(paths)


def download(url, destination):
    request = Request(url, headers={"User-Agent": "eqasim-switzerland count reference builder"})
    with urlopen(request, timeout=120) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    if destination.stat().st_size == 0:
        raise ValueError("Empty traffic-count download from %s" % url)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded_count(value):
    if value is None:
        raise ValueError("Missing traffic count in official source")
    return round(float(value))


def validate_mapping(crossings, mapping):
    """Cover every raw FID once, but emit only one count per physical crossing."""
    covered = []
    counters = []
    for item in mapping:
        fids = item.get("aliases", [item["fid"]])
        if item["fid"] not in fids:
            raise ValueError("Canonical FID must belong to its aliases")
        covered.extend(fids)
        expected_place = "Gruppe " + item["group"] + ":"
        if any(fid not in crossings.index or crossings.loc[fid, "label"] != "road"
               or not crossings.loc[fid, "interview_place"].startswith(expected_place) for fid in fids):
            raise ValueError("Crossing group/FID differs from reviewed count mapping: %s" % item["fid"])
        if item["source"].startswith("ticino_"):
            counters.append(("ticino", item["counter"]))
        elif item["source"] == "geneva":
            counters.extend(("geneva", str(counter["counter"])) for counter in item["counters"])
    if len(set(covered)) != len(covered) or len(set(counters)) != len(counters):
        raise ValueError("Duplicate crossing alias or official counter in count mapping")
    selected = crossings.label.eq("road") & crossings.interview_place.str.startswith(
        tuple("Gruppe " + group + ":" for group in ("G", "H", "I", "M", "N", "O")))
    if set(crossings.index[selected]) != set(covered):
        raise ValueError("G/H/I/M/N/O road FIDs differ from reviewed count mapping; update count_reference_sources.json")


def geneva_weekday_counts(path, mapping):
    """Sum paired TJOM counters once; preserve and explicitly opt into mixed years."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "error" in data or data.get("exceededTransferLimit"):
        raise ValueError("Geneva counter download failed or was truncated")
    counters = {}
    for feature in data["features"]:
        record = feature["attributes"]
        counter = str(record["no_point_mesure"])
        if counter in counters:
            raise ValueError("Duplicate Geneva counter %s" % counter)
        counters[counter] = record
    result = {}
    for item in mapping:
        directions = []
        if len(item["counters"]) != 2 or {c["direction"] for c in item["counters"]} != {"entry", "exit"}:
            raise ValueError("Geneva reference must identify one entry and one exit counter")
        if len({str(c["counter"]) for c in item["counters"]}) != 2:
            raise ValueError("Geneva entry and exit must use distinct counters")
        for selected in item["counters"]:
            counter = str(selected["counter"])
            record = counters.get(counter)
            if record is None or str(record.get("tjom_annee")) != str(selected["year"]):
                raise ValueError("Geneva counter/year mismatch for FID %s, counter %s" % (item["fid"], counter))
            value = record.get("tjom")
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("Missing/nonpositive Geneva weekday count for counter %s" % counter)
            directions.append(dict(counter=counter, direction=selected["direction"],
                                   year=selected["year"], weekday_count=rounded_count(value)))
        mixed = len({d["year"] for d in directions}) > 1
        if mixed and not item.get("allow_mixed_direction_years", False):
            raise ValueError("Mixed Geneva direction years require explicit acceptance for FID %s" % item["fid"])
        result[item["fid"]] = dict(weekday_total=sum(d["weekday_count"] for d in directions),
                                   directions=directions, mixed_direction_years=mixed)
    return result


def ticino_weekday_counts(path, mapping):
    """Read published combined TFM; directional means need not add to this value."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError("Ticino count references require Poppler's pdftotext executable")
    pages = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                           capture_output=True, text=True).stdout.split("\f")
    result = {}
    for item in mapping:
        page = pages[item["page"] - 1]
        lines = page.splitlines()
        year = int(item["source"].split("_")[1])
        if (not lines or "Posto conteggio:" not in lines[0] or len(lines) < 2
                or not lines[1].strip().endswith(str(year))
                or not re.search(r"\s{2,}0*" + str(item["counter"]) + r"\s{2,}", lines[1])):
            raise ValueError("Ticino counter/year/page mismatch for FID %s" % item["fid"])
        count = re.search(r"Traffico feriale medio TFM:\s+(\d+)\b", page)
        days = re.search(r"Giorni in funzione:\s+(\d+)\b", page)
        if not count or not days or int(count[1]) <= 0 or not 0 < int(days[1]) <= 366:
            raise ValueError("Missing annual weekday mean or coverage for FID %s" % item["fid"])
        result[item["fid"]] = dict(weekday_total=int(count[1]), operating_days=int(days[1]))
    return result


def astra_weekday_counts(path, wanted):
    """TJMO is the ASTRA mean for Monday–Friday, excluding public holidays."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        result, station = {}, None
        for row in workbook["TJM"].iter_rows(values_only=True):
            if row[0] is not None and str(row[0]).strip().isdigit():
                station = str(row[0]).strip()
            if station in wanted and row[5] == "TJMO":
                if station in result:
                    raise ValueError("Duplicate ASTRA TJMO station %s" % station)
                result[station] = {"annual": row[18], "months": {month: row[month + 5] for month in range(1, 13)}}
        if set(result) != wanted:
            raise ValueError("Missing ASTRA 2023 weekday station(s): %s" % (wanted - set(result)))
        return result
    finally:
        workbook.close()


def doubs_daily_counts(path, wanted):
    """Retain two-direction annual traffic (TMJA), including weekends."""
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source, delimiter=";"):
            if not row["annee"]:
                continue
            key = row["axe"], row["pr_abs"], int(row["annee"])
            if key in wanted:
                if key in result:
                    raise ValueError("Duplicate Doubs count %s" % (key,))
                result[key] = float(row["tv_tmja_2_sens"])
    if set(result) != wanted:
        raise ValueError("Missing Doubs count(s): %s" % (wanted - set(result)))
    return result


def vaud_weekday_counts(path, wanted):
    """Direction 0 is both ways; DWV is the observed weekday vehicle mean."""
    result = {}
    with sqlite3.connect(path) as connection:
        rows = connection.execute("""SELECT MLocNr, datedebutmesure, datefinmesure, dwv
            FROM indicateurcomptage WHERE direction = 0 AND classification = 'nonClassifie.autre'""")
        for counter, start, end, value in rows:
            key = counter, start, end
            if key in wanted:
                if key in result:
                    raise ValueError("Duplicate Vaud count %s" % (key,))
                result[key] = value
    if set(result) != wanted:
        raise ValueError("Missing Vaud count(s): %s" % (wanted - set(result)))
    return result
