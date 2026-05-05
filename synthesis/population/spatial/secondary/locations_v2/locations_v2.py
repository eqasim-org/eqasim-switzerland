import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import torch

from .locations_v2_helpers import (_load_wrapper, _build_level_attributes, _prepare_destination_level2_index,
                                      _prepare_primary_locations, _prepare_person_attributes, _build_coarse_X, _safe_xy,
                                      _euclidean, _build_detailed_X, _sample_company_in_l2, _coarse_company_mask, SECONDARY_SET)
from .NNModel import MNLWrapper, MediumLevel1Wrapper, DetailedLevel2Wrapper

logger = logging.getLogger("synpp")

_WORKER_STATE = {}


def _maybe_optimize_wrapper_model(wrapper, enable_compile=False, compile_mode="reduce-overhead"):
    # Keep model in float32 on CPU; this is already the dominant precision in this pipeline.
    wrapper.model = wrapper.model.float()
    wrapper.model.eval()

    if not enable_compile:
        return wrapper

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available in this PyTorch build; continuing without compile")
        return wrapper

    try:
        wrapper.model = torch.compile(wrapper.model, mode=compile_mode)
    except Exception as e:
        logger.warning("torch.compile failed (%s); continuing with eager model", e)

    return wrapper


def _extract_model_path(stage_output):
    if isinstance(stage_output, os.PathLike):
        stage_output = os.fspath(stage_output)
    if isinstance(stage_output, str) and stage_output.endswith(".pt"):
        return stage_output
    if isinstance(stage_output, tuple):
        for item in stage_output:
            if isinstance(item, os.PathLike):
                item = os.fspath(item)
            if isinstance(item, str) and item.endswith(".pt"):
                return item
    return None


def _load_wrapper_prefer_artifact(stage_output, wrapper_cls, stage_name):
    model_path = _extract_model_path(stage_output)
    if model_path is not None and os.path.exists(model_path):
        logger.info("Using %s artifact from %s", wrapper_cls.__name__, model_path)
        return wrapper_cls.load(model_path), model_path

    if model_path is not None and not os.path.exists(model_path):
        logger.warning(
            "Model artifact declared by stage '%s' does not exist at %s; falling back to stage wrapper object",
            stage_name,
            model_path,
        )

    wrapper = _load_wrapper(stage_output, wrapper_cls)
    logger.info("Using %s object returned by stage '%s'", wrapper_cls.__name__, stage_name)
    return wrapper, None


def _init_locations_v2_worker(
    coarse_model_path,
    medium_model_path,
    detailed_model_path,
    coarse_wrapper_obj,
    medium_wrapper_obj,
    detailed_wrapper_obj,
    coarse_attrs,
    level2_attrs,
    destination_l2_index,
    destination_fallback,
    worker_torch_threads,
    enable_mkldnn,
    enable_compile,
    compile_mode,
):  
    torch.backends.mkldnn.enabled = bool(enable_mkldnn)

    if worker_torch_threads is not None and int(worker_torch_threads) > 0:
        torch.set_num_threads(int(worker_torch_threads))
        torch.set_num_interop_threads(int(worker_torch_threads))

    if coarse_model_path is not None:
        coarse_wrapper = MNLWrapper.load(coarse_model_path)
    else:
        coarse_wrapper = coarse_wrapper_obj
    if medium_model_path is not None:
        medium_wrapper = MediumLevel1Wrapper.load(medium_model_path)
    else:
        medium_wrapper = medium_wrapper_obj
    if detailed_model_path is not None:
        detailed_wrapper = DetailedLevel2Wrapper.load(detailed_model_path)
    else:
        detailed_wrapper = detailed_wrapper_obj

    coarse_wrapper = _maybe_optimize_wrapper_model(coarse_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)
    medium_wrapper = _maybe_optimize_wrapper_model(medium_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)
    detailed_wrapper = _maybe_optimize_wrapper_model(detailed_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)

    coarse_mask = _coarse_company_mask(coarse_wrapper, coarse_attrs)

    _WORKER_STATE["coarse_wrapper"] = coarse_wrapper
    _WORKER_STATE["medium_wrapper"] = medium_wrapper
    _WORKER_STATE["detailed_wrapper"] = detailed_wrapper
    _WORKER_STATE["coarse_attrs"] = coarse_attrs
    _WORKER_STATE["level2_attrs"] = level2_attrs
    _WORKER_STATE["coarse_mask"] = coarse_mask
    _WORKER_STATE["destination_l2_index"] = destination_l2_index
    _WORKER_STATE["destination_fallback"] = destination_fallback


def _assign_chunk(df_trips_chunk, df_meta_chunk, seed):
    coarse_wrapper = _WORKER_STATE.get("coarse_wrapper")
    medium_wrapper = _WORKER_STATE.get("medium_wrapper")
    detailed_wrapper = _WORKER_STATE.get("detailed_wrapper")
    coarse_attrs = _WORKER_STATE.get("coarse_attrs")
    level2_attrs = _WORKER_STATE.get("level2_attrs")
    coarse_mask = _WORKER_STATE.get("coarse_mask")
    destination_l2_index = _WORKER_STATE.get("destination_l2_index")
    destination_fallback = _WORKER_STATE.get("destination_fallback")

    rng = np.random.RandomState(seed)
    locations_records = []
    convergence_records = []

    if len(df_trips_chunk) == 0 or len(df_meta_chunk) == 0:
        return locations_records, convergence_records

    df_meta_chunk = df_meta_chunk.set_index("person_id")
    person_groups = df_trips_chunk.groupby("person_id", sort=False)

    for person_id, grp in person_groups:
        if person_id not in df_meta_chunk.index:
            continue

        info = df_meta_chunk.loc[person_id]
        home_x = float(info["home_x"])
        home_y = float(info["home_y"])
        work_x = float(info["work_x"])
        work_y = float(info["work_y"])
        edu_x = float(info["edu_x"])
        edu_y = float(info["edu_y"])
        has_work = np.isfinite(work_x) and np.isfinite(work_y)

        age = float(info["age"])
        sex = float(info["sex"])
        employed = float(info["employed"])
        income_class = float(info["income_class"])
        car_availability = float(info["car_availability"])

        current_x, current_y = home_x, home_y
        if len(grp) > 0:
            first_preceding = str(grp.iloc[0].get("preceding_purpose", "home"))
            if first_preceding == "work" and has_work:
                current_x, current_y = work_x, work_y
            elif first_preceding == "education" and np.isfinite(edu_x) and np.isfinite(edu_y):
                current_x, current_y = edu_x, edu_y

        daily_total = 0.0
        daily_longest = _euclidean(current_x, current_y, home_x, home_y)

        person_trip_count = len(grp)
        for local_idx, row in enumerate(grp.itertuples(index=False)):
            following_purpose = str(getattr(row, "following_purpose")) if hasattr(row, "following_purpose") else str(getattr(row, "purpose"))
            trip_index = int(getattr(row, "trip_index"))

            if following_purpose == "home":
                next_x, next_y = home_x, home_y
            elif following_purpose == "work" and has_work:
                next_x, next_y = work_x, work_y
            elif following_purpose == "education" and np.isfinite(edu_x) and np.isfinite(edu_y):
                next_x, next_y = edu_x, edu_y
            elif following_purpose in SECONDARY_SET:
                chain_daily_longest = float(getattr(row, "daily_longest_distance_from_home", daily_longest))
                chain_daily_total = float(getattr(row, "daily_crowfly_total", daily_total))
                chain_consumed = float(getattr(row, "crowfly_consumed_before_trip", daily_total))
                chain_trip_pos = float(getattr(row, "trip_position_class", float(local_idx) / max(1, person_trip_count - 1)))

                chain_daily_longest = chain_daily_longest if np.isfinite(chain_daily_longest) and chain_daily_longest >= 0.0 else daily_longest
                chain_daily_total = chain_daily_total if np.isfinite(chain_daily_total) and chain_daily_total >= 0.0 else daily_total
                chain_consumed = chain_consumed if np.isfinite(chain_consumed) and chain_consumed >= 0.0 else daily_total
                chain_trip_pos = chain_trip_pos if np.isfinite(chain_trip_pos) else (float(local_idx) / max(1, person_trip_count - 1))

                feature_inputs = {
                    "home_x": home_x,
                    "home_y": home_y,
                    "work_x": work_x,
                    "work_y": work_y,
                    "origin_x": current_x,
                    "origin_y": current_y,
                    "has_work": has_work,
                    "age": age,
                    "daily_longest": chain_daily_longest,
                    "daily_total": chain_daily_total,
                    "consumed_before": chain_consumed,
                    "trip_position": chain_trip_pos,
                    "income_class": income_class,
                    "sex": sex,
                    "employed": employed,
                    "car_availability": car_availability,
                }

                Xc = _build_coarse_X(coarse_wrapper, coarse_attrs, feature_inputs, following_purpose)
                coarse_h3 = coarse_wrapper.predict_from_X(Xc, max_utility=False, rng=rng, candidate_mask=coarse_mask)[0]

                level1 = medium_wrapper.predict_level1(
                    level0_h3=coarse_h3,
                    home_x=home_x,
                    home_y=home_y,
                    work_x=work_x,
                    work_y=work_y,
                    origin_x=current_x,
                    origin_y=current_y,
                    age=age,
                    daily_longest_distance_from_home=chain_daily_longest,
                    daily_crowfly_total=chain_daily_total,
                    crowfly_consumed_before_trip=chain_consumed,
                    trip_position_class=chain_trip_pos,
                    sex=sex,
                    employed=employed,
                    car_availability=car_availability,
                    income_class=income_class,
                    purpose=following_purpose,
                    max_utility=False,
                    rng=rng,
                )
                level1_h3 = level1["level1_h3"]

                children = detailed_wrapper.children_by_level1.get((coarse_h3, level1_h3), [])
                if len(children) == 0:
                    convergence_records.append((False, 1))
                    continue

                Xd, md = _build_detailed_X(detailed_wrapper, children, level2_attrs, feature_inputs, following_purpose)
                idx_arr, probs = detailed_wrapper.predict_from_X(Xd, md, max_utility=False, rng=rng, return_probabilities=True)
                company_mask_l2 = np.array([level2_attrs.get(h, {}).get("num_statent", 0.0) > 0.0 for h in children], dtype=bool)
                if np.any(company_mask_l2):
                    p = probs[0, :len(children)] * company_mask_l2.astype(np.float64)
                    p = p / np.clip(p.sum(), 1e-12, None)
                    l2_idx = int(rng.choice(len(children), p=p))
                else:
                    l2_idx = int(idx_arr[0])
                l2_h3 = children[l2_idx]

                destination_id, geom = _sample_company_in_l2(
                    following_purpose,
                    l2_h3,
                    destination_l2_index,
                    destination_fallback,
                    rng,
                )
                if geom is None:
                    convergence_records.append((False, 1))
                    continue

                next_x, next_y = float(geom.x), float(geom.y)
                locations_records.append((person_id, trip_index, destination_id, geom))
                convergence_records.append((True, 1))
            else:
                next_x, next_y = current_x, current_y

            leg = _euclidean(current_x, current_y, next_x, next_y)
            daily_total += leg
            daily_longest = max(daily_longest, _euclidean(next_x, next_y, home_x, home_y))
            current_x, current_y = next_x, next_y

    return locations_records, convergence_records

def configure(context):
    context.stage("data.constants")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.destinations")

    context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
    context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")
    context.stage("synthesis.population.spatial.secondary.locations_v2.coarse_model")
    context.stage("synthesis.population.spatial.secondary.locations_v2.medium_model")
    context.stage("synthesis.population.spatial.secondary.locations_v2.detailed_model")

    context.config("random_seed")
    context.config("locations_v2_num_processes", 8)
    context.config("locations_v2_chunk_size_persons", 10000)
    context.config("locations_v2_worker_torch_threads", 2)
    context.config("locations_v2_enable_mkldnn", True)
    context.config("locations_v2_enable_torch_compile", True)
    context.config("locations_v2_torch_compile_mode", "reduce-overhead")


def execute(context):
    logger.info("Starting locations_v2 execution")
    constants = context.stage("data.constants")
    crs = constants.LV95
    rng = np.random.RandomState(context.config("random_seed"))
    enable_mkldnn = bool(context.config("locations_v2_enable_mkldnn"))
    enable_compile = bool(context.config("locations_v2_enable_torch_compile"))
    compile_mode = str(context.config("locations_v2_torch_compile_mode"))

    torch.backends.mkldnn.enabled = enable_mkldnn

    logger.info("Loading the models")
    coarse_stage = context.stage("synthesis.population.spatial.secondary.locations_v2.coarse_model")
    medium_stage = context.stage("synthesis.population.spatial.secondary.locations_v2.medium_model")
    detailed_stage = context.stage("synthesis.population.spatial.secondary.locations_v2.detailed_model")

    coarse_wrapper, coarse_model_path = _load_wrapper_prefer_artifact(
        coarse_stage,
        MNLWrapper,
        "synthesis.population.spatial.secondary.locations_v2.coarse_model",
    )
    medium_wrapper, medium_model_path = _load_wrapper_prefer_artifact(
        medium_stage,
        MediumLevel1Wrapper,
        "synthesis.population.spatial.secondary.locations_v2.medium_model",
    )
    detailed_wrapper, detailed_model_path = _load_wrapper_prefer_artifact(
        detailed_stage,
        DetailedLevel2Wrapper,
        "synthesis.population.spatial.secondary.locations_v2.detailed_model",
    )

    coarse_wrapper = _maybe_optimize_wrapper_model(coarse_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)
    medium_wrapper = _maybe_optimize_wrapper_model(medium_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)
    detailed_wrapper = _maybe_optimize_wrapper_model(detailed_wrapper, enable_compile=enable_compile, compile_mode=compile_mode)

    logger.info("Loading the H3 data and building attributes")
    h3_data, h3_geo, _ = context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
    coarse_attrs = _build_level_attributes(h3_data, h3_geo["level_0"], coarse_wrapper.all_h3)
    level2_attrs = _build_level_attributes(h3_data, h3_geo["level_2"])
    coarse_mask = _coarse_company_mask(coarse_wrapper, coarse_attrs)

    destination_l2_index, destination_fallback = _prepare_destination_level2_index(context)

    df_primary = _prepare_primary_locations(context)
    df_person = _prepare_person_attributes(context)
    df_meta = df_primary.merge(df_person, on="person_id", how="left")
    df_meta = df_meta.set_index("person_id")

    # Precompute numeric coordinates once to reduce repeated Point -> xy conversions in hot loops.
    df_meta_numeric = df_meta.reset_index().copy()
    home_xy = np.array([_safe_xy(v) for v in df_meta_numeric["home"].values], dtype=np.float64)
    work_xy = np.array([_safe_xy(v) for v in df_meta_numeric["work"].values], dtype=np.float64)
    edu_xy = np.array([_safe_xy(v) for v in df_meta_numeric["education"].values], dtype=np.float64)
    df_meta_numeric["home_x"] = home_xy[:, 0]
    df_meta_numeric["home_y"] = home_xy[:, 1]
    df_meta_numeric["work_x"] = work_xy[:, 0]
    df_meta_numeric["work_y"] = work_xy[:, 1]
    df_meta_numeric["edu_x"] = edu_xy[:, 0]
    df_meta_numeric["edu_y"] = edu_xy[:, 1]
    df_meta_numeric = df_meta_numeric[[
        "person_id", "home_x", "home_y", "work_x", "work_y", "edu_x", "edu_y",
        "age", "sex", "employed", "income_class", "car_availability",
    ]]

    df_trips = context.stage("synthesis.population.trips").copy()
    if "mz_person_id" not in df_trips.columns or "trip_id" not in df_trips.columns:
        raise RuntimeError("synthesis.population.trips must provide mz_person_id and trip_id for mz_chains feature lookup")

    mz_chains = context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")[[
        "person_id",
        "trip_id",
        "daily_longest_distance_from_home",
        "daily_crowfly_total",
        "crowfly_consumed_before_trip",
        "trip_position_class",
    ]].rename(columns={"person_id": "mz_person_id"})

    df_trips = df_trips.merge(mz_chains, how="left", on=["mz_person_id", "trip_id"])
    df_trips = df_trips.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    logger.info("Assigning secondary locations with locations_v2")
    num_processes = int(context.config("locations_v2_num_processes"))
    chunk_size_persons = max(1, int(context.config("locations_v2_chunk_size_persons")))
    worker_torch_threads = int(context.config("locations_v2_worker_torch_threads"))

    unique_persons = df_trips["person_id"].drop_duplicates().tolist()
    person_chunks = [unique_persons[i:i + chunk_size_persons] for i in range(0, len(unique_persons), chunk_size_persons)]

    locations_records = []
    convergence_records = []

    if num_processes <= 1:
        _WORKER_STATE.clear()
        _WORKER_STATE["coarse_wrapper"] = coarse_wrapper
        _WORKER_STATE["medium_wrapper"] = medium_wrapper
        _WORKER_STATE["detailed_wrapper"] = detailed_wrapper
        _WORKER_STATE["coarse_attrs"] = coarse_attrs
        _WORKER_STATE["level2_attrs"] = level2_attrs
        _WORKER_STATE["coarse_mask"] = coarse_mask
        _WORKER_STATE["destination_l2_index"] = destination_l2_index
        _WORKER_STATE["destination_fallback"] = destination_fallback

        with context.progress(label="Assigning secondary locations with locations_v2", total=len(person_chunks)) as progress:
            for chunk_idx, person_chunk in enumerate(person_chunks):
                trips_chunk = df_trips[df_trips["person_id"].isin(person_chunk)].copy()
                meta_chunk = df_meta_numeric[df_meta_numeric["person_id"].isin(person_chunk)].copy()
                recs, conv = _assign_chunk(trips_chunk, meta_chunk, seed=int(context.config("random_seed")) + chunk_idx)
                locations_records.extend(recs)
                convergence_records.extend(conv)
                progress.update()
    else:
        with context.progress(label="Assigning secondary locations with locations_v2", total=len(person_chunks)) as progress:
            with ProcessPoolExecutor(
                max_workers=num_processes,
                mp_context=get_context("spawn"),
                initializer=_init_locations_v2_worker,
                initargs=(
                    coarse_model_path,
                    medium_model_path,
                    detailed_model_path,
                    None if coarse_model_path is not None else coarse_wrapper,
                    None if medium_model_path is not None else medium_wrapper,
                    None if detailed_model_path is not None else detailed_wrapper,
                    coarse_attrs,
                    level2_attrs,
                    destination_l2_index,
                    destination_fallback,
                    worker_torch_threads,
                    enable_mkldnn,
                    enable_compile,
                    compile_mode,
                ),
            ) as executor:
                futures = {}
                base_seed = int(context.config("random_seed"))
                for chunk_idx, person_chunk in enumerate(person_chunks):
                    trips_chunk = df_trips[df_trips["person_id"].isin(person_chunk)].copy()
                    meta_chunk = df_meta_numeric[df_meta_numeric["person_id"].isin(person_chunk)].copy()
                    futures[executor.submit(_assign_chunk, trips_chunk, meta_chunk, base_seed + chunk_idx)] = chunk_idx

                for future in as_completed(futures):
                    recs, conv = future.result()
                    locations_records.extend(recs)
                    convergence_records.extend(conv)
                    progress.update()

    if len(locations_records) == 0:
        df_locations = gpd.GeoDataFrame(
            pd.DataFrame(columns=["person_id", "trip_index", "destination_id"]),
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    else:
        df_locations = pd.DataFrame.from_records(
            locations_records,
            columns=["person_id", "trip_index", "destination_id", "geometry"],
        )
        df_locations = gpd.GeoDataFrame(df_locations, geometry="geometry", crs=crs)
        df_locations = df_locations.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    if len(convergence_records) == 0:
        df_convergence = pd.DataFrame(columns=["valid", "size"])
    else:
        df_convergence = pd.DataFrame.from_records(convergence_records, columns=["valid", "size"])

    logger.info("locations_v2 success rate: %f", df_convergence["valid"].mean() if len(df_convergence) else 0.0)
    return df_locations, df_convergence

