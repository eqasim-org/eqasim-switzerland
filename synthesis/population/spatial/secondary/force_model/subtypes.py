"""Local facility preference relative to pooled 'other', estimated from MZ 2015."""
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from data.activity_purposes import MZ2015_OTHER_SUBTYPES as PURPOSES

OPTION = "secondary_activities.force_model.split_other"
STAGE = "synthesis.population.spatial.secondary.force_model.subtypes"
REFIT = "secondary_activities.force_model.refit_facility_preferences"
FIT_STAGE = "synthesis.population.spatial.secondary.force_model.facility_preferences"
COEFFICIENTS = Path(__file__).resolve().parents[5] / "calibration/secondary_locations/mz2015_facility_preferences.csv"
SECTORS = {
    "school_daycare": ("851", "852", "853", "8891"),
    "other_education": ("854", "855", "856"), "health_social": ("86", "87", "88"),
    "personal_repair": ("95", "96"), "postal_finance_admin": ("53", "64", "84"),
    "retail": ("47",), "food_leisure": ("56", "90", "91", "92", "93"),
}


def configure(context):
    if context.config(REFIT, default=False):
        context.stage(FIT_STAGE)
    context.stage("synthesis.population.destinations")
    context.stage("data.statent.statent")
    context.stage("data.gtfs.cleaned")
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Score every distinct eligible location once in Swiss projected coordinates.

    destinations = context.stage("synthesis.population.destinations").to_crs(2056)
    other = destinations.loc[destinations.offers_other.fillna(False), "geometry"]
    xy = np.unique(np.column_stack((other.x, other.y)), axis=0)
    border = context.stage("data.spatial.swiss_border").to_crs(2056).geometry.union_all()
    shapely.prepare(border)
    inside = shapely.intersects_xy(border, *xy.T)

    # Foreign locations retain the pooled preference because STATENT covers Switzerland.

    stops = Path(context.path("data.gtfs.cleaned")) / "output/stops.txt"
    sites = facility_catalog(context.stage("data.statent.statent"), stops, border)
    coefficients = (context.stage(FIT_STAGE) if context.config(REFIT)
                    else pd.read_csv(COEFFICIENTS, index_col="sector"))
    coefficients = coefficients.loc[:, list(PURPOSES.values())]
    log_weights = np.zeros((inside.sum(), len(PURPOSES)))
    selected_xy = xy[inside]
    for sector, points in sites.groupby("sector"):
        distance = cKDTree(points[["x", "y"]]).query(selected_xy)[0]
        log_weights += np.exp(-distance[:, None] / 50.) * coefficients.loc[sector].to_numpy()
    index = pd.MultiIndex.from_arrays(xy.T, names=["x", "y"])
    scores = pd.DataFrame(1., index=index, columns=coefficients.columns)
    scores.loc[inside] = np.exp(log_weights)
    return scores


def facility_catalog(jobs, stops_file, border):
    jobs = jobs.dropna(subset=["x", "y"])
    codes = jobs.noga.astype(str)
    sector = np.full(len(jobs), "other_establishment", dtype=object)
    for label, prefixes in reversed(SECTORS.items()):
        sector[codes.str.startswith(prefixes)] = label
    establishments = jobs[["x", "y"]].assign(sector=sector)
    stops = pd.read_csv(stops_file, usecols=["stop_lat", "stop_lon"]).dropna().drop_duplicates()
    projection = Transformer.from_crs(4326, 2056, always_xy=True)
    x, y = projection.transform(stops.stop_lon, stops.stop_lat)
    inside = shapely.intersects_xy(border, x, y)
    stops = pd.DataFrame(dict(x=x[inside], y=y[inside], sector="pt_stop"))
    return pd.concat([establishments, stops], ignore_index=True).drop_duplicates()


def align_weights(scores, xy):
    index = pd.MultiIndex.from_arrays(np.asarray(xy).T, names=scores.index.names)
    values = scores.loc[index]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Incomplete facility preference weights")
    return values


def selected_purposes(value):
    if isinstance(value, bool):
        return tuple(PURPOSES.values()) if value else ()
    if not isinstance(value, (list, tuple)) or any(type(code) is not int or code not in PURPOSES for code in value):
        raise ValueError("force_model.split_other expects a boolean or a list of MZ purpose codes from [5, 9, 10]")
    return tuple(name for code, name in PURPOSES.items() if code in value)


def assignment_purposes(trips, donor, selected=None):
    # Use processed donor trip IDs only after the raw survey code travelled through loop splitting.
    if "other_subtype_code" not in donor:
        raise ValueError("Rebuild data.microcensus.trips to retain MZ 2015 other subtypes")
    keys = ["mz_person_id", "trip_id"]
    lookup = donor.rename(columns={"person_id": keys[0]}).set_index(keys)
    if not lookup.index.is_unique:
        raise ValueError("Duplicate MZ donor/trip keys")
    result = trips.copy()
    mask = result.following_purpose.eq("other")
    donor_keys = pd.MultiIndex.from_frame(result.loc[mask, keys])
    codes = lookup.other_subtype_code.reindex(donor_keys)
    if codes.isna().any():
        raise ValueError("Missing processed MZ donor trip for an other destination")
    names = codes.map(PURPOSES)
    allowed = tuple(PURPOSES.values()) if selected is None else selected
    result.loc[mask, "following_purpose"] = names.where(names.isin(allowed), "other").to_numpy()
    return result


def validate(context):
    import hashlib
    if context.config(REFIT):
        return "survey"
    return hashlib.sha256(COEFFICIENTS.read_bytes()).hexdigest()
