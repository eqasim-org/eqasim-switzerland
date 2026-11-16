"""Regional Geneva airport passenger ground-access demand (full population)."""

import calendar
import hashlib
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from data.airports import common, geneva_times
from data.airports.common import apportion, atomic_write, export_signature, file_digest, normalized

logger = logging.getLogger("synpp")
MODEL_VERSION = 4


def configure(context):
    context.config("data_path")
    context.stage("data.airports.model")
    context.config("override_geneva_ods", default=False, volatile=True)
    context.config("geneva_airport_pdf", default="airports/geneva/geneva_airport.pdf")
    context.config("geneva_airport_survey_csv", default=None)
    context.config("geneva_airport_web_data", default=None)
    context.config("geneva_airport_time_data", default=None)
    context.config("geneva_airport_cache", default="airports/geneva/geneva_ods.parquet")
    context.config("geneva_airport_month", default=None)
    context.config("geneva_airport_demand_factor", default=1.0)


def execute(context):

    # Identify input contents, model code and demand settings before reusing trips.

    pdf, survey, web, cache = input_paths(context)
    timing_path = time_data_path(context)
    settings = demand_settings(context)
    allocation_model = context.stage("data.airports.model")
    token = signature(pdf, survey, survey.with_suffix(".json"), web, timing_path,
                      Path(__file__), Path(common.__file__), Path(geneva_times.__file__),
                      settings={**settings, "allocation_model": allocation_model.signature})
    trips = None if context.config("override_geneva_ods") else load_processed_data(token, cache)
    if trips is not None:
        logger.info("Geneva airport: reusing %s", cache)
        return trips

    # Read the reviewed survey and the offline snapshot of public traffic totals.

    observations = process_pdf(pdf, survey)
    traffic = process_json(web)
    ods = build_ods(observations)
    trips = allocation_model.allocate(build_trips(ods, traffic, **settings), "GVA")

    # Persist auditable probabilities, trips and assumptions alongside the cache.

    timing = geneva_times.load_snapshot(timing_path)
    time_distribution = geneva_times.build_time_distribution(timing)
    summary = demand_summary(trips, traffic, settings, token)
    summary["municipal_allocation"] = {"model_signature": allocation_model.signature,
        "parameters": allocation_model.parameters, "swiss_resolution": "municipality",
        "foreign_resolution": "original regional groups; no foreign census/hotel inputs",
        "observed_regional_totals_preserved": True}
    summary["timing"] = {key: value for key, value in timing.items() if key not in ("flights", "query")}
    summary["timing"]["flight_count"] = len(geneva_times.flight_frame(timing))
    summary["timing"]["snapshot_path"] = str(timing_path)
    summary["timing"]["day_aggregation"] = "Equal-weight normalized calendar-date curves for named days, weekday, weekend and all."
    save_ods(cache, token, trips, ods, summary, time_distribution)
    logger.info("Geneva airport: %.1f daily passenger trips across %d directional/profile ODs",
                trips.daily_trips.sum(), len(trips))
    return trips


def validate(context):
    """Invalidate Synpp's own result when external input contents change."""
    pdf, survey, web, cache = input_paths(context)
    token = signature(pdf, survey, survey.with_suffix(".json"), web, time_data_path(context),
                      Path(__file__), Path(common.__file__), Path(geneva_times.__file__), settings=demand_settings(context))
    token += export_signature({**cache_paths(cache), 'manifest': cache.with_suffix('.signature.json')})
    return token if not context.config("override_geneva_ods") else token + str(uuid.uuid4())


def input_paths(context):
    root = Path(context.config("data_path"))

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    airport = root / "airports/geneva"
    survey = context.config("geneva_airport_survey_csv")
    survey = resolve(survey) if survey else airport / "geneva_survey.csv"
    web = context.config("geneva_airport_web_data")
    web = resolve(web) if web else airport / "data_from_websites/geneva_airport.json"
    return resolve(context.config("geneva_airport_pdf")), survey, web, resolve(context.config("geneva_airport_cache"))


def time_data_path(context):
    value = context.config("geneva_airport_time_data")
    return geneva_times.snapshot_file(value, data_path=context.config("data_path"))


def demand_settings(context):
    month = context.config("geneva_airport_month")
    factor = context.config("geneva_airport_demand_factor")
    if month is not None and (isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12):
        raise ValueError("geneva_airport_month must be null or an integer from 1 to 12")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor) or factor < 0:
        raise ValueError("geneva_airport_demand_factor must be finite and non-negative")
    return {"month": month, "demand_factor": float(factor)}


def signature(*files, settings=None):
    """Content-based identity, independent of timestamps and process hash seeds."""
    payload = {"model_version": MODEL_VERSION, "files": [file_digest(f) for f in files],
               "settings": settings or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def process_pdf(pdf_file, survey_csv=None):
    """Load the reviewed CSV only when it belongs to this exact supplied PDF.

    The supplied PDF has no text layer. Repeated OCR would add latency and risk
    silently changing percentages. A replacement PDF requires a reviewed CSV
    and a sibling JSON containing its pdf_sha256 and survey_year.
    """
    survey_csv = Path(survey_csv) if survey_csv else Path(pdf_file).parent / "geneva_survey.csv"
    metadata = json.loads(survey_csv.with_suffix(".json").read_text(encoding="utf-8"))
    if file_digest(pdf_file) != metadata["pdf_sha256"]:
        raise ValueError("Geneva PDF changed: review its regional tables, update the survey CSV "
                         "and its sibling JSON pdf_sha256 before rebuilding demand.")
    observations = pd.read_csv(survey_csv, keep_default_na=False)
    required = {"table", "region_id", "region_name", "country", "region_type", "members", "percent", "pdf_page"}
    if not required.issubset(observations.columns):
        raise ValueError(f"Survey CSV missing columns: {sorted(required - set(observations.columns))}")
    if observations.duplicated(["table", "region_id"]).any():
        raise ValueError("Duplicate survey table/region observations")
    observations["percent"] = pd.to_numeric(observations.percent, errors="raise")
    if not np.isfinite(observations.percent).all() or not observations.percent.between(0, 100).all():
        raise ValueError("Survey percentages must be finite and between 0 and 100")
    observations.attrs["survey_year"] = int(metadata["survey_year"])
    return observations


def process_json(json_file):
    traffic = json.loads(Path(json_file).read_text(encoding="utf-8"))
    if traffic.get("schema_version") != 1:
        raise ValueError("Unsupported Geneva public-data schema")
    year = traffic["year"]
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2200:
        raise ValueError("Traffic year must be a calendar year")
    for key in ("annual_passengers", "annual_arrivals", "annual_departures",
                "direct_transit_passengers", "connecting_passengers"):
        value = traffic[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid non-negative traffic count: {key}")
    if not traffic["annual_passengers"] or not traffic["annual_arrivals"] or not traffic["annual_departures"]:
        raise ValueError("Annual and directional totals must be positive")
    if traffic["connecting_multiplier"] not in (1, 2):
        raise ValueError("connecting_multiplier must be 1 or 2; see source counting assumptions")
    monthly = np.asarray(traffic["monthly_passengers"], dtype=float)
    if monthly.shape != (12,) or not np.isfinite(monthly).all() or (monthly < 0).any() or monthly.sum() <= 0:
        raise ValueError("Expected twelve finite non-negative monthly traffic weights")
    if not np.isclose(monthly.sum(), traffic["annual_passengers"], rtol=.01):
        raise ValueError("Monthly and annual traffic totals differ by over 1%; check source years")
    if not np.isclose(traffic["annual_arrivals"] + traffic["annual_departures"],
                      traffic["annual_passengers"], rtol=.01):
        raise ValueError("Directional and annual traffic totals differ by over 1%")
    if local_passengers(traffic) < 0:
        raise ValueError("Airside exclusions exceed annual passenger traffic")
    if not traffic.get("sources") or any(not s.get("url", "").startswith("https://") for s in traffic["sources"]):
        raise ValueError("Public data requires source URLs")
    return traffic


def regional_weights(observations, table, regions):
    """Reconcile rounded regional bars to the survey's country headline shares."""
    rows = observations.loc[observations.table.eq(table)].set_index("region_id")
    raw = rows.loc[regions.region_id, "percent"].to_numpy(float)
    countries = observations.loc[observations.table.eq(table + "_country")].set_index("region_id")
    country_ids = list(dict.fromkeys(regions.country))
    country_weights = normalized(countries.loc[country_ids, "percent"])
    result = np.zeros(len(regions))
    for country, weight in zip(country_ids, country_weights):
        mask = regions.country.eq(country).to_numpy()
        result[mask] = normalized(raw[mask]) * weight
    return result


def build_ods(observations):
    """Fit resident/visitor regional priors to observed pooled regional margins."""
    regions = observations.loc[observations.table.eq("pooled") & observations.region_type.ne("unknown"),
                               ["region_id", "region_name", "country", "region_type", "members"]].reset_index(drop=True)
    if regions.empty:
        raise ValueError("Missing pooled regional observations")
    target = regional_weights(observations, "pooled", regions)
    profiles = observations.loc[observations.table.eq("profile")].set_index("region_id").percent
    profile_weights = normalized([profiles["outgoing"], profiles["incoming"] + profiles["other"]])
    if (profile_weights <= 0).any():
        raise ValueError("Both resident and visitor profiles need positive survey weights")
    prior = np.vstack([regional_weights(observations, table, regions)
                       for table in ("outgoing", "incoming_other")])
    joint = prior * profile_weights[:, None]
    for _ in range(10000):
        joint *= np.divide(target, joint.sum(axis=0), out=np.ones_like(target), where=joint.sum(axis=0) > 0)
        joint *= (profile_weights / joint.sum(axis=1))[:, None]
        if np.max(np.abs(joint.sum(axis=0) - target)) < 1e-12:
            break
    else:
        raise ValueError("Incompatible regional and profile margins in Geneva survey")
    output = []
    for i, profile in enumerate(("outgoing", "incoming_other")):
        frame = regions.copy()
        frame["profile"] = profile
        frame["prior_region_probability"] = prior[i]
        frame["profile_probability"] = profile_weights[i]
        frame["region_probability"] = joint[i] / profile_weights[i]
        frame["joint_probability"] = joint[i]
        frame["pooled_region_probability"] = target
        frame["survey_year"] = observations.attrs["survey_year"]
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def local_passengers(traffic):
    return (traffic["annual_passengers"] - traffic["direct_transit_passengers"]
            - traffic["connecting_multiplier"] * traffic["connecting_passengers"])


def build_trips(ods, traffic, month=None, demand_factor=1.0):
    if not ods.survey_year.eq(traffic["year"]).all():
        logger.warning("Applying Geneva survey year %s to traffic year %s", ods.survey_year.iloc[0], traffic["year"])
    annual = local_passengers(traffic) * demand_factor
    monthly = normalized(traffic["monthly_passengers"])
    year = traffic["year"]
    days = (366 if calendar.isleap(year) else 365) if month is None else calendar.monthrange(year, month)[1]
    period_weight = 1.0 if month is None else monthly[month - 1]
    direction_weights = normalized([traffic["annual_departures"], traffic["annual_arrivals"]])
    output = []
    for direction, weight in zip(("to_airport", "from_airport"), direction_weights):
        frame = ods.copy()
        frame["direction"] = direction
        frame["origin_region"] = frame.region_id if direction == "to_airport" else "GVA"
        frame["destination_region"] = "GVA" if direction == "to_airport" else frame.region_id
        frame["direction_probability"] = weight
        frame["annual_trips"] = annual * weight * frame.joint_probability
        frame["period_trips"] = frame.annual_trips * period_weight
        frame["daily_trips"] = frame.period_trips / days
        frame["traffic_year"] = year
        frame["month"] = 0 if month is None else month
        frame["period_days"] = days
        frame["demand_factor"] = demand_factor
        output.append(frame)
    trips = pd.concat(output, ignore_index=True)
    trips["annual_trips_integer"] = apportion(trips.annual_trips)
    trips["period_trips_integer"] = apportion(trips.period_trips)
    trips["daily_trips_integer"] = apportion(trips.daily_trips)
    return trips


def demand_summary(trips, traffic, settings, token):
    return {"signature": token, "model_version": MODEL_VERSION, "unit": "passenger ground-access movements",
            "settings": settings, "traffic_year": traffic["year"],
            "annual_local_passengers_unscaled": local_passengers(traffic),
            "annual_local_passengers_if_connecting_multiplier_1": traffic["annual_passengers"] - traffic["direct_transit_passengers"] - traffic["connecting_passengers"],
            "annual_trips": float(trips.annual_trips.sum()), "daily_trips": float(trips.daily_trips.sum()),
            "sources": traffic["sources"], "assumptions": traffic["assumptions"],
            "method": "Country-reconciled survey priors raked to pooled region and profile margins; same catchment in both directions.",
            "warnings": ["Annual-average or monthly-average day, without workday or hourly inference.",
                         "Unknown locations redistributed within known countries; unresolved countries included in OTHER.",
                         "Resident/visitor profile is independent of ground direction: both profiles use both directions.",
                         "Connecting multiplier is an explicit modelling assumption; confirm with airport."]}


def cache_paths(path):
    path = Path(path)
    return {"trips": path, "ods": path.with_suffix(".probabilities.csv"),
            "csv": path.with_suffix(".csv"), "regional": path.with_suffix(".regional_od.csv"),
            "summary": path.with_suffix(".summary.json"),
            "times": path.with_suffix(".time_distribution.csv"),
            "municipal": path.with_suffix(".municipality_od.csv")}


def load_processed_data(signature_files, processed_ods_file):
    paths = cache_paths(processed_ods_file)
    manifest = Path(processed_ods_file).with_suffix(".signature.json")
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata["signature"] != signature_files:
            return None
        if any(file_digest(path) != metadata["artifacts"][key] for key, path in paths.items()):
            return None
        return pd.read_parquet(paths["trips"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_ods(processed_ods_file, signature_files, trips, ods, summary, time_distribution):
    paths = cache_paths(processed_ods_file)
    atomic_write(paths["times"], lambda p: time_distribution.to_csv(p, index=False))
    atomic_write(paths["trips"], lambda p: trips.to_parquet(p, index=False))
    atomic_write(paths["ods"], lambda p: ods.to_csv(p, index=False))
    atomic_write(paths["csv"], lambda p: trips.to_csv(p, index=False))
    keys = ["origin_region", "destination_region", "region_id", "region_name", "country",
            "region_type", "members", "direction", "traffic_year", "month", "period_days"]
    counts = ["annual_trips", "period_trips", "daily_trips", "annual_trips_integer",
              "period_trips_integer", "daily_trips_integer"]
    coarse = trips.copy()
    if "spatial_resolution" in coarse:
        coarse["origin_region"] = coarse.region_id.where(coarse.direction.eq("to_airport"), "GVA")
        coarse["destination_region"] = coarse.region_id.where(coarse.direction.eq("from_airport"), "GVA")
    regional = coarse.groupby(keys, as_index=False, sort=False)[counts].sum()
    municipal_keys = [key for key in ["origin_region", "destination_region", "municipality_id",
                     "municipality_name", "canton_id", "spatial_resolution", "direction", "traffic_year", "month", "period_days"] if key in trips]
    municipal = trips.groupby(municipal_keys, as_index=False, sort=False, dropna=False)[counts].sum()
    atomic_write(paths["municipal"], lambda p: municipal.to_csv(p, index=False))
    atomic_write(paths["regional"], lambda p: regional.to_csv(p, index=False))
    atomic_write(paths["summary"], lambda p: p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))
    metadata = {"signature": signature_files, "artifacts": {key: file_digest(path) for key, path in paths.items()}}
    atomic_write(Path(processed_ods_file).with_suffix(".signature.json"),
                 lambda p: p.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8"))


def read_ods(processed_ods_file, signature_files):
    return load_processed_data(signature_files, processed_ods_file)
