"""Reusable PT results outside the stage directory, published only on success."""
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow as pa

import matsim.runtime.eqasim as eqasim

logger = logging.getLogger("synpp")


def route_trips(context, requests):
    # Canonical ordering makes cache identity and later seeded draws independent
    # of the order in which Java workers finish their routes.
    requests = requests.loc[:, (
        'identifier',
        'origin_x',
        'origin_y',
        'destination_x',
        'destination_y',
        'departure_time',
    )].copy()
    requests["identifier"] = requests["identifier"].astype(str)
    requests = requests.sort_values("identifier", kind="stable").reset_index(drop=True)
    if requests["identifier"].duplicated().any():
        raise ValueError("Departure routing requires unique identifiers")
    if requests.empty:
        return empty_routes()

    # Include requests and routing inputs, but not DMC coefficients: changing
    # utilities must not repeat routing. Seed changes are checked explicitly too.
    config = Path(context.stage("matsim.config.pt")).resolve()
    jar = Path(context.path("matsim.runtime.eqasim")) / context.stage("matsim.runtime.eqasim")
    key = cache_key(requests, config, jar, context.config("random_seed"))
    directory = Path(context.working_directory) / 'departure_time_routing'
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / f"{key}.parquet"
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            result = align_routes(requests, cached, complete=True)
            values = result.loc[:, (
                'access_egress_time_min',
                'in_vehicle_time_min',
                'transfers',
                'waiting_time_min',
                'initial_waiting_time_min',
                'distance_km',
            )].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid cached route values")
            logger.info("Reusing %d departure-time routes from %s", len(result), cache_path)
            return result
        except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
            logger.warning("Cannot reuse departure routing cache %s: %s; routing again", cache_path, exc)

    # Unique temporary files avoid exposing partial CSV/Parquet output after an
    # OOM or router error. A completed cache survives deletion of context.path().
    with tempfile.TemporaryDirectory(prefix=f".{key}.", dir=directory) as temporary:
        input_path = Path(temporary) / "requests.csv"
        output_path = Path(temporary) / "routes.csv"
        requests.to_csv(input_path, index=False)
        logger.info("Routing %d departure alternatives; cache destination: %s", len(requests), cache_path)
        eqasim.run(context, 'org.eqasim.core.tools.routing.RunBatchPublicTransportRouter', [
            "--config-path", str(config), "--input-path", str(input_path),
            "--output-trips-path", str(output_path),
            "--batch-size", "2048", "--chunk-size", "131072",
            "--eqasim-configurator", 'org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator', "--threads", str(context.config("threads")),
        ])
        try:
            raw = pd.read_csv(output_path, dtype={"identifier": str})
        except pd.errors.EmptyDataError:
            raw = pd.DataFrame(columns=["identifier", *(
                'access_travel_time_min',
                'egress_travel_time_min',
                'transfer_waiting_time_min',
                'transfer_travel_time_min',
                'in_vehicle_time_total_min',
                'in_vehicle_distance_total_km',
                'transfers',
                'initial_waiting_time_min',
            )])
        result = prepare_routes(requests, raw)
        temporary_parquet = Path(temporary) / "routes.parquet"
        result.to_parquet(temporary_parquet, index=False, compression="zstd")
        os.replace(temporary_parquet, cache_path)
    return result


def cache_key(requests, config, jar, seed):
    # Read the small config; use size/mtime/inode for large network/schedule/JAR
    # files so checking a cache does not reread gigabytes of routing inputs.
    config_bytes = config.read_bytes()
    inputs = {config, jar.resolve()}
    inputs.update(path.resolve() for path in config.parent.iterdir() if path.is_file())
    for parameter in ET.fromstring(config_bytes).iter("param"):
        value = parameter.get("value", "")
        if value and len(value) < 4096:
            path = Path(value)
            path = path if path.is_absolute() else config.parent / path
            try:
                if path.is_file():
                    inputs.add(path.resolve())
            except OSError:
                # Most MATSim parameters are values rather than file paths.
                continue
    signatures = []
    for path in sorted(inputs):
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino))
    digest = hashlib.sha256(config_bytes)
    digest.update(json.dumps({"version": 1, "seed": int(seed),
                              "router": 'org.eqasim.core.tools.routing.RunBatchPublicTransportRouter', "configurator": 'org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator',
                              "inputs": signatures}, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(requests, index=False).to_numpy().tobytes())
    digest.update(str(tuple(map(str, requests.dtypes))).encode())
    return digest.hexdigest()


def align_routes(requests, routes, *, complete=False):
    if routes["identifier"].isna().any() or routes["identifier"].duplicated().any():
        raise ValueError("Null or duplicate departure routing identifiers")
    if not routes["identifier"].isin(requests["identifier"]).all():
        raise ValueError("Unknown departure routing identifiers")
    if complete and len(routes) != len(requests):
        raise ValueError("Incomplete departure routing cache")
    return requests[["identifier"]].merge(routes, on="identifier", how="left", validate="one_to_one")


def prepare_routes(requests, raw):
    # Missing/invalid routes are complete, reusable results of a successful run.
    # Cache them too, rather than routing them again on every restart.
    raw = raw.loc[:, ["identifier", *(
        'access_travel_time_min',
        'egress_travel_time_min',
        'transfer_waiting_time_min',
        'transfer_travel_time_min',
        'in_vehicle_time_total_min',
        'in_vehicle_distance_total_km',
        'transfers',
        'initial_waiting_time_min',
    )]].astype(
        {column: float for column in (
            'access_travel_time_min',
            'egress_travel_time_min',
            'transfer_waiting_time_min',
            'transfer_travel_time_min',
            'in_vehicle_time_total_min',
            'in_vehicle_distance_total_km',
            'transfers',
            'initial_waiting_time_min',
        )})
    result = align_routes(requests, raw)
    values = result.loc[:, (
        'access_travel_time_min',
        'egress_travel_time_min',
        'transfer_waiting_time_min',
        'transfer_travel_time_min',
        'in_vehicle_time_total_min',
        'in_vehicle_distance_total_km',
        'transfers',
        'initial_waiting_time_min',
    )].to_numpy(dtype=float)
    invalid = (~np.isfinite(values) | (values < 0)).any(axis=1)
    result["access_egress_time_min"] = result["access_travel_time_min"] + result["egress_travel_time_min"]
    result["waiting_time_min"] = result["transfer_waiting_time_min"] + result["transfer_travel_time_min"]
    result["in_vehicle_time_min"] = result["in_vehicle_time_total_min"]
    result["distance_km"] = result["in_vehicle_distance_total_km"]
    result = result.loc[:, (
        'identifier',
        'access_egress_time_min',
        'in_vehicle_time_min',
        'transfers',
        'waiting_time_min',
        'initial_waiting_time_min',
        'distance_km',
    )].copy()
    result.loc[invalid, (
        'access_egress_time_min',
        'in_vehicle_time_min',
        'transfers',
        'waiting_time_min',
        'initial_waiting_time_min',
        'distance_km',
    )] = 1000000.0
    logger.info("Departure routing completed: %d alternatives, %d unroutable", len(result), invalid.sum())
    return result


def empty_routes():
    return pd.DataFrame({column: pd.Series(dtype=str if column == "identifier" else float)
                         for column in (
                             'identifier',
                             'access_egress_time_min',
                             'in_vehicle_time_min',
                             'transfers',
                             'waiting_time_min',
                             'initial_waiting_time_min',
                             'distance_km',
                         )})
