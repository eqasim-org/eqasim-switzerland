"""Retain survey demand while distinguishing located endpoints from proxies."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def recover_locations(observations, context):
    """Recover missing coordinates without using a proxy as a known residence.

    Swiss municipality codes and NUTS IDs are used only when their country
    agrees with the reported endpoint country. Otherwise a foreign country
    polygon provides a directional proxy. Missing non-European road origins
    can start at their known interview group. A missing through-destination
    needs a country polygon or a located reported exit; otherwise it stays
    unresolved, with its calibrated weight retained in the audit.
    """
    result = observations.copy()
    for side in ("start", "end"):
        x, y = side + "_x", side + "_y"
        valid = result[x].between(-180, 180) & result[y].between(-90, 90)
        result.loc[~valid, [x, y]] = np.nan
        result[side + "_location_source"] = np.where(valid, "reported", "unresolved")
    result["location_issue"] = ""
    if result[["start_x", "start_y", "end_x", "end_y"]].notna().all(axis = None):
        result["location_usable"] = True
        return result

    # Reuse existing spatial inputs, including the historical municipality-ID
    # mapping. Country-level proxies deliberately do not claim a town/address.
    nuts = context.stage("data.spatial.nuts").to_crs(2056).drop_duplicates("nuts_id").set_index("nuts_id")
    municipalities, mapping = context.stage("data.spatial.municipalities")
    municipalities = municipalities.to_crs(2056).set_index("municipality_id")
    municipality_ids = mapping.set_index("deprecated_municipality_id")["municipality_id"].to_dict()
    points = context.stage("data.cross_border.interview_places").to_crs(2056)
    exits = context.stage("data.cross_border.road_exit_points").to_crs(2056)
    border = context.stage("data.spatial.swiss_border").iloc[0]
    rng = np.random.RandomState(context.config("random_seed"))

    for side, country_field, place_field in [("start", "origin_country_raw", "origin_place"),
                                              ("end", "destination_country_raw", "destination_place")]:
        missing = result[side + "_location_source"].eq("unresolved")
        for idx in result.index[missing]:
            row = result.loc[idx]
            country = str(row[country_field]).strip()
            nuts_country = {"GB": "UK", "GR": "EL"}.get(country, country)
            place = str(row[place_field]).strip()
            geometry, source = None, "unresolved"

            # A GISCO string from a different country is not a usable location.
            if country == "CH" and place.startswith("CH_CH") and place[5:].isdigit():
                municipality = municipality_ids.get(int(place[5:]), int(place[5:]))
                if municipality in municipalities.index:
                    geometry = municipalities.loc[municipality, "geometry"].representative_point()
                    source = "municipality_code"
            elif country != "CH" and place.startswith(country + "_"):
                code = place.split("_", 1)[1]
                if code.startswith(nuts_country) and code in nuts.index:
                    geometry = nuts.loc[code, "geometry"].representative_point()
                    source = "nuts_code"

            if geometry is None and country != "CH" and nuts_country in nuts.index:
                geometry = nuts.loc[nuts_country, "geometry"].representative_point()
                source = "country_proxy"

            # No town is invented for an overseas road origin. Its explicit
            # crossing proxy is excluded from residence-based area filtering
            # and from perimeter routing that requires a real foreign location.
            if geometry is None and country != "CH" and row["trip_mode"] == "car" and side == "start":
                candidates = points.loc[points["label"].eq("road") & points["interview_place"].eq(row["interview_place"])]
                if len(candidates):
                    weights = candidates["importance"]
                    weights = weights if np.isfinite(weights).all() and weights.sum() > 0 else None
                    geometry = candidates.sample(n = 1, weights = weights, random_state = rng).geometry.iloc[0]
                    source = "crossing_proxy"

            if geometry is None and country != "CH" and row["trip_mode"] == "car" and side == "end":
                code = row.get("survey_exit_code")
                candidates = exits.loc[exits["survey_exit_code"].eq(code).fillna(False)]
                if len(candidates):
                    geometry = outward_proxy(candidates.geometry.iloc[0], border, context.config("border_offset"))
                    source = "reported_exit_proxy"

            if geometry is not None:
                geographic = gpd.GeoSeries([geometry], crs = 2056).to_crs(4326).iloc[0]
                result.loc[idx, [side + "_x", side + "_y"]] = [geographic.x, geographic.y]
                result.loc[idx, side + "_location_source"] = source

    result["location_usable"] = result[["start_x", "start_y", "end_x", "end_y"]].notna().all(axis = 1)
    result.loc[~result["location_usable"], "location_issue"] = "missing_endpoint_without_supported_recovery"
    return result


def outward_proxy(anchor, border, border_offset):
    """An explicit routing direction outside a known exit, not a residence.

    Choose the point furthest outside Switzerland on a circle around the exit.
    The downstream directed network still has to visit the reported exit. This
    proxy is only used when neither an actual destination nor country geometry
    is available, and remains labelled in the survey and generated-agent audits.
    """
    distance = max(2 * border_offset, 100000)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint = False)
    candidates = [Point(anchor.x + distance * np.cos(a), anchor.y + distance * np.sin(a)) for a in angles]
    return max(candidates, key = lambda point: point.distance(border))


def recover_unlocated_swiss_end(observations, side, municipality_missing, context):
    """Use an explicit Swiss municipality ID when its reported point falls outside.

    This handles border-point rounding without moving already located endpoints
    or guessing a Swiss destination from another respondent's journey.
    """
    result = observations.copy()
    if not municipality_missing.any():
        return result
    municipalities, mapping = context.stage("data.spatial.municipalities")
    municipalities = municipalities.to_crs(4326).set_index("municipality_id")
    ids = mapping.set_index("deprecated_municipality_id")["municipality_id"].to_dict()
    field = "origin_place" if side == "start" else "destination_place"
    for idx in result.index[municipality_missing]:
        code = str(result.loc[idx, field]).strip()
        if not code.startswith("CH_CH") or not code[5:].isdigit():
            continue
        municipality = ids.get(int(code[5:]), int(code[5:]))
        if municipality not in municipalities.index:
            continue
        point = municipalities.loc[municipality, "geometry"].representative_point()
        result.loc[idx, [side + "_x", side + "_y"]] = [point.x, point.y]
        result.loc[idx, side + "_location_source"] = "municipality_code"
    return result
