"""Collect a reproducible official road-exit reference outside the repository.

Run with ``python -m data.cross_border.collect_road_exit_points --help``.
This is an input-preparation utility, not a network request made by a Synpp stage.
"""

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from data.cross_border.generate_od import read_road_exit_names
from data.cross_border.road_exit_points import crossing_name

COLLECTION = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swisstlmregio"

# Exact name equivalences for the 2021 survey. Other offices must match the
# complete name after removing the parenthesised customs number; no fuzzy match
# or municipality-centroid geocoding is used. Guard aliases against code changes.
OFFICE_ALIASES = {
    1013: ("Riehen Lörracherstrasse (150)", "Basel Ost - Güg Riehen"),
    1014: ("Basel Freiburgerstrasse (149)", "Basel Ost - Güg Freiburgerstrasse"),
    1015: ("Basel Hiltalingerstrasse (148)", "Basel Ost - Güg Hiltalingerstrasse"),
    1016: ("Riehen Grenzacherstrasse (155)", "Basel Ost - Güg Grenzacherstrasse"),
    1017: ("Riehen Weilstrasse (152)", "Basel Ost - Güg Weilstrasse"),
    1018: ("Riehen Inzlingerstrasse (früher 153)", "Basel Ost - Güg Inzlingerstrasse"),
    1023: ("Waldshut-Koblenz (204)", "Koblenz"),
    1089: ("Chiasso-Strada (418)", "Chiasso"),
    1111: ("Gd St.Bernard-Hospice (früher 515)", "Hospice Grand St-Bernard"),
    1125: ("Soral II", "Soral 2"),
    1131: ("Chancy II", "Chancy 2"),
    1133: ("Conrières", "Cornières"),
    1134: ("Soral I", "Soral 1"),
    1144: ("Bossy (Collex)", "Collex-Bossy"),
    1146: ("Chancy I", "Chancy 1"),
    1181: ("Basel Lysbüchel (138)", "Basel-Lysbüchel - Güg Lysbüchel"),
    1183: ("Allschwil (140)", "Leimental - Güg Allschwil"),
    1185: ("Basel Burgfelderstrasse (143)", "Basel West - Güg Burgfelderstrasse"),
    1186: ("Kohlenstrasse; ex Hüningerstr. (144)", "Basel-Lysbüchel - Güg Kohlenstrasse"),
}


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--data-path", type = Path, required = True, help = "Pipeline data directory containing crossborder/.")
    parser.add_argument("--reference-year", type = int, default = 2021, help = "swissTLMRegio edition; default matches the 2021 survey.")
    parser.add_argument("--archive", type = Path, help = "Reuse an already downloaded swissTLMRegio shapefile ZIP.")
    parser.add_argument("--output", type = Path, help = "Default: DATA_PATH/crossborder/road_exit_points_official.gpkg.")
    args = parser.parse_args()
    output = args.output or args.data_path / "crossborder" / "road_exit_points_official.gpkg"
    if output.exists():
        parser.error("Output exists; use a new --output path and review it before replacing the current reference.")
    names = read_road_exit_names(args.data_path / "crossborder" / "AuGQPV_2021" / "AGQPV21_finale_Auswertungsdatenbank_Codebuch.xlsx")
    if not names:
        parser.error("The 2021 survey codebook is required to collect its exit reference.")

    # Keep the large download temporary; retain the small original point layer.

    with tempfile.TemporaryDirectory(prefix = "eqasim-road-exits-") as directory:
        archive, metadata = acquire_archive(args.reference_year, args.archive, Path(directory))
        offices = extract_offices(archive, Path(directory))
        reference, audit = match_offices(names, offices, args.reference_year, metadata["archive_url"])
        output.parent.mkdir(parents = True, exist_ok = True)
        reference.to_file(output, layer = "road_exits", driver = "GPKG")
        offices.to_file(output, layer = "source_customs_offices", driver = "GPKG")
        metadata.update(retrieved_utc = datetime.now(timezone.utc).isoformat(), source_rows = len(offices),
                        matched_survey_codes = len(reference), survey_year = 2021,
                        product_page = "https://www.swisstopo.admin.ch/en/landscape-model-swisstlmregio")
        output.with_suffix(".sources.json").write_text(json.dumps(metadata, indent = 2) + "\n")
        audit.to_csv(output.with_suffix(".matching.csv"), index = False)
    print("Collected %d official offices and %d survey-code matches in %s" % (len(offices), len(reference), output))


def acquire_archive(year, existing, temporary):
    """Use a dated official release and verify its published SHA-256 checksum."""
    import requests
    item_url = "%s/items/swisstlmregio_%d" % (COLLECTION, year)
    response = requests.get(item_url, timeout = 30)
    response.raise_for_status()
    item = response.json()
    asset = item["assets"]["swisstlmregio_%d_2056.shp.zip" % year]
    archive = existing if existing is not None else temporary / "swisstlmregio.zip"
    if existing is None:
        with requests.get(asset["href"], stream = True, timeout = (30, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as stream:
                for block in response.iter_content(chunk_size = 1024 * 1024):
                    stream.write(block)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = asset["checksum:multihash"]
    if not expected.startswith("1220") or digest.lower() != expected[4:].lower():
        raise ValueError("The archive does not match swisstopo's published SHA-256 checksum.")
    return archive, dict(reference_year = year, catalog_item_url = item_url, archive_url = asset["href"], sha256 = digest)


def extract_offices(archive_path, temporary):
    """Extract only the four small CustomsOffice shapefile components."""
    target = temporary / "customs"
    target.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        for suffix in [".shp", ".shx", ".dbf", ".prj"]:
            files = [name for name in archive.namelist() if Path(name).name.lower() == "swisstlmregio_customsoffice" + suffix]
            if len(files) != 1:
                raise ValueError("Expected one CustomsOffice%s component in the official archive." % suffix)
            (target / ("customs" + suffix)).write_bytes(archive.read(files[0]))
    return gpd.read_file(target / "customs.shp").to_crs(2056)


def match_offices(names, offices, year, source_url):
    """Associate exact individual names or explicitly reviewed name aliases."""
    offices = offices.to_crs(2056)
    grouped = {key: group for key, group in offices.groupby(offices["NAMN"].map(crossing_name))}
    rows, audit = [], []
    for code, name in sorted(names.items()):
        method = "exact_name"
        target = re.sub(r"\s*\([^)]*\)", "", name)
        if code in OFFICE_ALIASES:
            expected, target = OFFICE_ALIASES[code]
            if crossing_name(expected) != crossing_name(name):
                audit.append(dict(survey_exit_code = code, survey_exit_name = name, status = "alias_name_changed"))
                continue
            method = "reviewed_name_alias"
        found = grouped.get(crossing_name(target))
        if found is None or len(found) != 1:
            audit.append(dict(survey_exit_code = code, survey_exit_name = name,
                              status = "office_missing" if found is None else "office_name_ambiguous"))
            continue
        office = found.iloc[0]
        rows.append(dict(survey_exit_code = code, survey_exit_name = name, source_name = office.NAMN,
                         source_id = office.UUID, reference_year = year, source_url = source_url,
                         source_date = str(office.DATEOFCHNG), name_match = method, geometry = office.geometry))
        audit.append(dict(survey_exit_code = code, survey_exit_name = name, status = method, source_id = office.UUID))
    reference = gpd.GeoDataFrame(rows, columns = ["survey_exit_code", "survey_exit_name", "source_name", "source_id",
        "reference_year", "source_url", "source_date", "name_match", "geometry"], geometry = "geometry", crs = 2056)
    return reference, pd.DataFrame(audit)


if __name__ == "__main__":
    main()
