"""Persistence helpers for per-canton count comparison results."""

import os


def save_count_results(city, matched, flows, output_directory):
    grouped_matches = matched.groupby("id").agg({
        "geometry": "first",
        "link_id": list,
        "road_geometry": list,
        "distance": list,
    }).reset_index()

    results = flows.merge(grouped_matches, on="id", how="left")
    results["city"] = city.lower()
    output_path = os.path.join(output_directory, f"results_flows_{city.lower()}.pkl")
    results.to_pickle(output_path)
    return output_path
