"""Explicitly archive unchanged BFS projection exports; never run inside Synpp."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

from data.statpop.projections.raw import HOUSEHOLD_FILE, POPULATION_TABLE, read_households, read_population

POPULATION_URL = ("https://www.pxweb.bfs.admin.ch/api/v1/de/"
                  + POPULATION_TABLE + "/" + POPULATION_TABLE + ".px")
HOUSEHOLD_URL = ("https://disseminate.stats.swiss/rest/data/"
                 "CH1.SCEN,DF_SCEN_MENAGES,1.0.0/AM-00-2025.._T..A.P")


def download(output_path, start_year=2024, end_year=2055):
    """Create a new snapshot directory, preserving response bytes and queries."""
    if start_year > end_year:
        raise ValueError("start-year must not exceed end-year")
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "incomplete", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "population_scenario": "AR-00-2025", "household_scenario": "AM-00-2025",
                "requested_years": [start_year, end_year], "files": {}}

    # Download only when explicitly requested. Pipeline runs use these files offline.

    try:
        with requests.Session() as session:
            metadata = session.get(POPULATION_URL, timeout=(15, 120))
            save_response(metadata, root / (POPULATION_TABLE + ".metadata.json"), manifest)
            metadata = metadata.json()
            if "AR-00-2025" not in metadata.get("title", ""):
                raise ValueError("Unexpected population scenario; review BFS table metadata")

            # Four-year chunks stay below the PXWeb cell limit. Save each response
            # separately: concatenation/decoding belongs to the raw pipeline stage.

            years = list(range(start_year, end_year + 1))
            for begin in range(0, len(years), 4):
                selected = years[begin:begin + 4]
                query = population_query(metadata, selected)
                name = "%s_%d_%d" % (POPULATION_TABLE, selected[0], selected[-1])
                (root / (name + ".query.json")).write_text(json.dumps(query, ensure_ascii=False, indent=2))
                response = session.post(POPULATION_URL, json=query, timeout=(15, 120))
                save_response(response, root / (name + ".csv"), manifest)
                print("Archived population years %d–%d" % (selected[0], selected[-1]), flush=True)

            # SDMX retains its official column codes and metadata. MEN and PER
            # are both preserved; the household stage explicitly selects MEN.

            response = session.get(HOUSEHOLD_URL,
                params={"startPeriod": start_year, "endPeriod": end_year},
                headers={"Accept": "text/csv"}, timeout=(15, 120))
            save_response(response, root / HOUSEHOLD_FILE, manifest)
            print("Archived household reference export", flush=True)
        # A successful HTTP response alone does not establish a usable export.
        # Check the table schema before marking the snapshot complete.
        read_population(sorted(root.glob(POPULATION_TABLE + "*.csv")))
        read_households(root / HOUSEHOLD_FILE)
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def population_query(metadata, years):
    # Select disjoint categories, without national/sex/nationality/age totals.
    selections = {"Kanton": [str(i) for i in range(1, 27)],
                  "Staatsangehörigkeit (Kategorie)": ["1", "2"],
                  "Geschlecht": ["1", "2"], "Alter": [str(i) for i in range(1, 102)]}
    variables = {item["code"]: item for item in metadata["variables"]}
    if set(variables) != set(selections) | {"Jahr", "Beobachtungseinheit"}:
        raise ValueError("Unexpected BFS population dimensions")
    for name, labels in {"Jahr": [str(year) for year in years],
                         "Beobachtungseinheit": ["Bevölkerungsstand am 31. Dezember"]}.items():
        codes = dict(zip(variables[name]["valueTexts"], variables[name]["values"]))
        if not set(labels).issubset(codes):
            raise ValueError("Requested year-end projection categories unavailable: %s" % labels)
        selections[name] = [codes[label] for label in labels]
    for name, codes in selections.items():
        if not set(codes).issubset(variables[name]["values"]):
            raise ValueError("BFS population category codes changed: %s" % name)
    return {"query": [{"code": item["code"], "selection": {
                "filter": "item", "values": selections[item["code"]]}}
                for item in metadata["variables"]], "response": {"format": "csv"}}


def save_response(response, path, manifest):
    response.raise_for_status()
    # Preserve the original payload; do not decode, rename columns or rewrite rows.
    payload = response.content
    if not payload:
        raise ValueError("Empty BFS response for %s" % response.url)
    path.write_bytes(payload)
    manifest["files"][path.name] = {
        "url": response.url, "content_type": response.headers.get("Content-Type"),
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, help="New directory; existing snapshots are never overwritten")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2055)
    args = parser.parse_args()
    download(args.output_path, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
