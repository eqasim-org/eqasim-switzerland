"""Download SITG counts and review large 2025 differences using shared detector IDs.

Run from the repository root with uv run -m scripts.update_transcality_sitg.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from analysis.counts.cantons.transcality import (
    DAILY_FLOW_OVERRIDES, EXCLUDED_DETECTOR_IDS, _read_matches,
    prepare_transcality_counts,
)
from analysis.counts.matching.road_matching import angular_difference

URL = "https://vector.sitg.ge.ch/arcgis/rest/services/OTC_COMPTAGE_TRAFIC/FeatureServer/0"


def download(directory, stamp):
    session = requests.Session()

    def get(endpoint, **params):
        response = session.get(URL + endpoint, params={"f": "json", **params}, timeout=60)
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise ValueError(result["error"])
        return result

    metadata = get("")
    ids = get("/query", where="1=1", returnIdsOnly="true")["objectIds"]
    features = []
    for start in range(0, len(ids), 500):
        result = get("/query", objectIds=",".join(map(str, ids[start:start + 500])),
                     outFields="*", outSR=2056, returnGeometry="true")
        if result.get("exceededTransferLimit"):
            raise ValueError("SITG truncated a download batch")
        features.extend(result["features"])
    if {f["attributes"]["OBJECTID"] for f in features} != set(ids) or len(features) != len(ids):
        raise ValueError("Incomplete or duplicated SITG download")
    (directory / f"layer_metadata_{stamp}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (directory / f"counts_{stamp}.json").write_text(json.dumps(
        {"source": URL, "spatialReference": {"wkid": 2056}, "features": features},
        ensure_ascii=False, indent=2))
    attributes = pd.DataFrame([f["attributes"] for f in features])
    attributes.to_csv(directory / f"counts_{stamp}.csv", index=False)
    points = [f.get("geometry") or {} for f in features]
    geometry = gpd.points_from_xy([p.get("x", np.nan) for p in points],
                                 [p.get("y", np.nan) for p in points])
    result = gpd.GeoDataFrame(attributes, geometry=geometry, crs=2056)
    result.to_file(directory / f"counts_{stamp}.gpkg", layer="counts", driver="GPKG")
    return result


def review(counts_directory, snapshot, minimum_relative, minimum_absolute):
    matched = _read_matches(counts_directory / "matched_points.csv")
    # Always compare against the baseline, so re-running does not remove
    # corrections merely because their previously corrected totals now agree.
    baseline = prepare_transcality_counts(counts_directory, apply_sitg_overrides=False)
    protected = {item for group in DAILY_FLOW_OVERRIDES for item in group} | EXCLUDED_DETECTOR_IDS
    snapshot = snapshot.copy()
    snapshot["detector_id"] = snapshot.NO_SIREDO.map(lambda value: str(float(value)) if pd.notna(value) else "")
    snapshot["ambiguous_id"] = snapshot.detector_id.duplicated(keep=False) | snapshot.detector_id.isin(["", "0.0"])
    merged = matched.merge(baseline[["detector_ids", "flow", "flow_lower", "flow_upper"]],
                           left_on="detid", right_on="detector_ids", validate="one_to_one")
    merged = merged.rename(columns={"geometry": "transcality_geometry", "angle": "transcality_angle"})
    merged = merged.merge(snapshot.drop(columns="geometry").assign(sitg_geometry=list(snapshot.geometry)),
                          left_on="detid", right_on="detector_id", validate="many_to_many")
    merged["distance_m"] = [a.distance(b) if a is not None and b is not None else np.nan
                            for a, b in zip(merged.transcality_geometry, merged.sitg_geometry)]
    merged["angle_difference"] = [angular_difference(a, 90 - b) if pd.notna(a) and pd.notna(b) else np.nan
                                  for a, b in zip(merged.transcality_angle, merged.ANGLE)]
    merged["count_year"] = pd.to_numeric(merged.TJOM_ANNEE, errors="coerce")
    merged["daily_flow"] = merged.TJOM
    merged["absolute_difference"] = (merged.daily_flow - merged.flow).abs()
    merged["relative_difference"] = merged.absolute_difference / merged.flow
    merged["eligible_2025"] = (merged.count_year.eq(2025) & merged.daily_flow.gt(0) & merged.flow.gt(0)
                               & ~merged.ambiguous_id & merged.distance_m.le(75)
                               & merged.angle_difference.le(45) & ~merged.detid.isin(protected))
    merged["selected"] = (merged.eligible_2025 & merged.relative_difference.ge(minimum_relative)
                           & merged.absolute_difference.ge(minimum_absolute))
    columns = ["detector_id", "source", "flow", "flow_lower", "flow_upper", "daily_flow", "count_year",
               "TJM", "TJM_ANNEE", "absolute_difference", "relative_difference", "distance_m",
               "angle_difference", "OBJECTID", "NOM_VOIE", "DIRECTION", "eligible_2025", "selected"]
    result = merged[columns].sort_values("detector_id")
    directory = counts_directory / "sitg"
    result.to_csv(directory / "transcality_comparison_2025.csv", index=False)
    result[result.selected].to_csv(directory / "transcality_overrides_2025.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-directory", required=True, type=Path, help="Transcality input directory")
    parser.add_argument("--snapshot-date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--reuse-snapshot", action="store_true")
    parser.add_argument("--minimum-relative-difference", type=float, default=.25)
    parser.add_argument("--minimum-absolute-difference", type=float, default=1000)
    args = parser.parse_args()
    if not (np.isfinite([args.minimum_relative_difference, args.minimum_absolute_difference]).all()
            and args.minimum_relative_difference >= 0 and args.minimum_absolute_difference >= 0):
        parser.error("Difference thresholds must be finite and nonnegative")
    directory = args.counts_directory / "sitg"
    directory.mkdir(exist_ok=True)
    snapshot = (gpd.read_file(directory / f"counts_{args.snapshot_date}.gpkg") if args.reuse_snapshot
                else download(directory, args.snapshot_date))
    result = review(args.counts_directory, snapshot,
                    args.minimum_relative_difference, args.minimum_absolute_difference)
    provenance = {"source": URL, "snapshot": f"counts_{args.snapshot_date}.gpkg", "feature_count": len(snapshot),
                  "measurement_year": 2025, "measure": "TJOM (weekday daily traffic)",
                  "minimum_relative_difference": args.minimum_relative_difference,
                  "minimum_absolute_difference": args.minimum_absolute_difference,
                  "maximum_distance_m": 75, "maximum_angle_difference_degrees": 45,
                  "selected_detectors": result.loc[result.selected, "detector_id"].tolist(),
                  "selection_updated_utc": datetime.now(timezone.utc).isoformat()}
    (directory / "selection_metadata.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    print(f"Saved {len(snapshot)} SITG points; compared {len(result)} shared-ID records; selected {result.selected.sum()} corrections.")


if __name__ == "__main__":
    main()
