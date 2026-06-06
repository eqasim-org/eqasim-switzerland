import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from shapely.geometry import Point
import queue
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import torch

from .location_helpers import (_prepare_primary_locations, _prepare_person_attributes, _euclidean, SECONDARY_SET, _get_first_location)
from .model_wrappers import HierarchicalLocationChoiceModel

logger = logging.getLogger("synpp")

_WORKER_STATE = {}
_WORKER_PROGRESS_QUEUE = None
_WORKER_PROGRESS_FLUSH_EVERY = 100
_CHUNCK_SIZE_PERSONS = 10000

def configure(context):
    context.stage("data.constants")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")
    context.stage("synthesis.population.destinations")

    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("synthesis.population.spatial.secondary_nn.regional_model")
    context.stage("synthesis.population.spatial.secondary_nn.subregional_model")
    context.stage("synthesis.population.spatial.secondary_nn.local_model")

    context.config("random_seed")
    context.config("threads")

def execute(context):
    logger.info("Starting secondary_nn location assignment")
    constants = context.stage("data.constants")
    crs = constants.LV95    
    torch.backends.mkldnn.enabled = True
    
    ########## Loading NN models ##########
    logger.info("\t Loading the models")
    wrapper = HierarchicalLocationChoiceModel.build(context, optimize=False)    
    wrapper.assert_ready_for_prediction()

    ########## Loading population data ##########
    logger.info("\t Loading population data")    
    df_primary = _prepare_primary_locations(context)
    df_person = _prepare_person_attributes(context)
    df_meta = df_primary.merge(df_person, on="person_id", how="left")    

    # Precompute numeric coordinates once to reduce repeated Point -> xy conversions in hot loops.
    df_meta = df_meta.reset_index()
    df_meta[["home_x","home_y", "work_x", "work_y", "edu_x", "edu_y"]] = np.nan
    has_work_mask = df_meta["work"].notna()
    has_education_mask = df_meta["education"].notna()
    df_meta["home_x"] = gpd.GeoSeries(df_meta["home"]).x
    df_meta["home_y"] = gpd.GeoSeries(df_meta["home"]).y    
    df_meta.loc[has_work_mask, "work_x"] = gpd.GeoSeries(df_meta.loc[has_work_mask, "work"]).x
    df_meta.loc[has_work_mask, "work_y"] = gpd.GeoSeries(df_meta.loc[has_work_mask, "work"]).y    
    df_meta.loc[has_education_mask, "edu_x"] = gpd.GeoSeries(df_meta.loc[has_education_mask, "education"]).x
    df_meta.loc[has_education_mask, "edu_y"] = gpd.GeoSeries(df_meta.loc[has_education_mask, "education"]).y
    df_meta["has_work"] = has_work_mask
    df_meta["has_education"] = has_education_mask
    
    df_meta = df_meta[["person_id","home_x", "home_y", "work_x", "work_y", "edu_x", "edu_y", "age", 
                       "sex", "employed", "income_class", "car_availability","has_work", "has_education",
                       "work_destination_id","education_destination_id", "home_destination_id"]]
    
    # remove nans from meta data, they can provoke an error or loss of performance with numba
    df_meta.loc[~df_meta["has_work"], ["work_x","work_y"]] = 0.0
    df_meta.loc[~df_meta["has_education"], ["edu_x","edu_y"]] = 0.0    

    # get the trips and enrich them
    df_trips = context.stage("synthesis.population.trips").copy()
    if "mz_person_id" not in df_trips.columns or "trip_id" not in df_trips.columns:
        raise RuntimeError("synthesis.population.trips must provide mz_person_id and trip_id for mz_chains feature lookup")

    mz_chains = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")[["person_id", "trip_id", "daily_longest_distance_from_home",
                                    "daily_crowfly_total", "crowfly_consumed_before_trip", "trip_position_class",
                                    "departure_time_normalized", "daily_longest_distance_from_work",
                                    "activity_duration_h", "target_distance", "trip_origin_distance_from_home", "activity_chain"]].rename(columns={"person_id": "mz_person_id"})

    df_trips = df_trips.merge(mz_chains, how="left", on=["mz_person_id", "trip_id"])
    df_trips = df_trips.sort_values(by=["person_id", "trip_id"]).reset_index(drop=True)

    # chunk by person_id to ensure all trips of a person are processed together, and to enable efficient parallelization without shared state
    logger.info("Assigning secondary locations with secondary_nn")
    worker_torch_threads = 2
    num_processes = int(context.config("threads")/worker_torch_threads)    
    chunk_size_persons = max(1, int(_CHUNCK_SIZE_PERSONS))    
    progress_flush_persons = max(1, int(_WORKER_PROGRESS_FLUSH_EVERY))

    unique_persons = df_trips["person_id"].unique()
    person_chunk_ids = (np.arange(len(unique_persons), dtype=np.int32) // chunk_size_persons)
    person_to_chunk = pd.Series(person_chunk_ids, index=unique_persons)
    chunk_person_counts = np.bincount(person_chunk_ids)

    df_trips = df_trips.assign(_chunk_id=df_trips["person_id"].map(person_to_chunk).astype(np.int32))
    df_meta = df_meta.assign(_chunk_id=df_meta["person_id"].map(person_to_chunk))
    df_meta = df_meta[df_meta["_chunk_id"].notna()].copy()
    df_meta["_chunk_id"] = df_meta["_chunk_id"].astype(np.int32)

    # ensure these distances are not infinite or NaN, as that would break the models; we can set them to 0 since the models will learn to ignore them when the corresponding features are missing
    df_trips["daily_longest_distance_from_home"] = df_trips["daily_longest_distance_from_home"].replace([np.inf, -np.inf], np.nan).fillna(df_trips["daily_longest_distance_from_home"].median())
    df_trips["daily_crowfly_total"] = df_trips["daily_crowfly_total"].replace([np.inf, -np.inf], np.nan).fillna(df_trips["daily_crowfly_total"].median())
    df_trips["daily_longest_distance_from_work"] = df_trips["daily_longest_distance_from_work"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_trips["crowfly_consumed_before_trip"] = df_trips["crowfly_consumed_before_trip"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_trips["departure_time_normalized"] = df_trips["departure_time_normalized"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    df_trips["activity_duration_h"] = df_trips["activity_duration_h"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_trips["target_distance"] = df_trips["target_distance"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_trips["trip_origin_distance_from_home"] = df_trips["trip_origin_distance_from_home"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df_trips["activity_chain"] = df_trips["activity_chain"].apply(lambda v: v.astype(np.float32))

    # ensure cols are in the right type for the models
    assert df_meta.isna().sum().sum()==0, "Meta data contains NaNs"
    assert df_trips.drop(columns=["activity_chain"]).isna().sum().sum()==0, "trips data (excluding object columns) contains NaNs"
    df_trips = df_trips.astype({"person_id": "int64", "preceding_purpose": str, "following_purpose": str, "trip_index": int, "daily_longest_distance_from_home": float, 
                                "daily_crowfly_total": float, "daily_longest_distance_from_work": float, "crowfly_consumed_before_trip": float,
                                "trip_position_class": float, "departure_time_normalized": float, "activity_duration_h": float,
                                "target_distance": float, "trip_origin_distance_from_home": float})
    df_meta = df_meta.astype({"person_id": "int64", "home_x": float, "home_y": float, "work_x": float, "work_y": float, "edu_x": float, "edu_y": float, "age": float, "sex": float, 
                              "employed": float, "income_class": float, "car_availability": float, "has_work": bool, "has_education": bool, "_chunk_id": int})

    trips_by_chunk = {int(chunk_id): grp.drop(columns=["_chunk_id"]).copy() for chunk_id, grp in df_trips.groupby("_chunk_id", sort=True)}
    meta_by_chunk = {int(chunk_id): grp.drop(columns=["_chunk_id"]).copy() for chunk_id, grp in df_meta.groupby("_chunk_id", sort=True)}
    chunk_ids = sorted(trips_by_chunk.keys())
    empty_meta_chunk = df_meta.drop(columns=["_chunk_id"]).iloc[0:0].copy()

    locations_records = []
    convergence_records = []
    
    ########### HERE WE DECIDE WHETHER TO USE MULTIPROCESSING OR NOT BASED ON THE CONFIGURATION ###########    
    if num_processes <= 1:
        base_seed = int(context.config("random_seed"))
        _WORKER_STATE.clear()
        init_args = (wrapper, worker_torch_threads, True, True, "max-autotune", None, progress_flush_persons)
        _init_location_worker(*init_args) 

        with context.progress(label="Assigning secondary locations with locations_v2", total=len(unique_persons)) as progress:
            for chunk_idx, chunk_id in enumerate(chunk_ids):
                trips_chunk = trips_by_chunk[chunk_id]
                meta_chunk = meta_by_chunk.get(chunk_id, empty_meta_chunk)
                recs, conv = _assign_person_chunk(trips_chunk, meta_chunk, seed=base_seed + chunk_idx)
                locations_records.extend(recs)
                convergence_records.extend(conv)
                progress.update(int(chunk_person_counts[chunk_id]))
    else:
        mp_context = get_platform_mp_context()
        compile_in_worker = _should_compile_in_worker(mp_context)
        progress_queue = mp_context.Queue()
        base_seed = int(context.config("random_seed"))
        # Keep at most 2× num_processes tasks in-flight to bound memory usage while ensuring
        # all workers stay busy. Slots are refilled one-for-one as completions come in.
        max_inflight = num_processes * 2
        chunk_queue = list(enumerate(chunk_ids))   # [(chunk_idx, chunk_id), ...]
        submit_cursor = 0
        if not compile_in_worker:
            logger.info("\t Compiling models in parent before forking workers")
            wrapper.optimize_wrapper(enable_compile=True, compile_mode="max-autotune")
        init_args = (wrapper, worker_torch_threads, True, compile_in_worker, "max-autotune", progress_queue, progress_flush_persons)

        with context.progress(label="Assigning secondary locations with locations_v2", total=len(unique_persons)) as progress:
            with ProcessPoolExecutor(max_workers=num_processes, mp_context=mp_context, initializer=_init_location_worker,
                initargs=init_args) as executor:
                pending = {}  # future -> chunk_id

                # seed the pipeline
                while submit_cursor < len(chunk_queue) and len(pending) < max_inflight:
                    chunk_idx, chunk_id = chunk_queue[submit_cursor]
                    submit_cursor += 1
                    pending[executor.submit(_assign_person_chunk,
                                            trips_by_chunk[chunk_id],
                                            meta_by_chunk.get(chunk_id, empty_meta_chunk),
                                            base_seed + chunk_idx)] = chunk_id

                while pending:
                    _drain_progress_queue(progress, progress_queue)
                    done, _ = wait(list(pending.keys()), timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_id = pending.pop(future)
                        recs, conv = future.result()
                        locations_records.extend(recs)
                        convergence_records.extend(conv)
                        # refill the freed slot immediately so workers stay busy
                        if submit_cursor < len(chunk_queue):
                            next_idx, next_chunk_id = chunk_queue[submit_cursor]
                            submit_cursor += 1
                            pending[executor.submit(_assign_person_chunk,
                                                    trips_by_chunk[next_chunk_id],
                                                    meta_by_chunk.get(next_chunk_id, empty_meta_chunk),
                                                    base_seed + next_idx)] = next_chunk_id
                    _drain_progress_queue(progress, progress_queue)

                _drain_progress_queue(progress, progress_queue)

        progress_queue.close()
        progress_queue.join_thread()

    # Build the dataframe and geodataframe for the assigned locations
    df_locations = pd.DataFrame.from_records(
        locations_records,
        columns=["person_id", "trip_index", "destination_id", "geometry"],
    )
    df_locations = gpd.GeoDataFrame(df_locations, geometry="geometry", crs=crs)
    df_locations = df_locations.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)
    # adjust the trip index for compatibility with the rda stage
    df_locations["trip_index"] = df_locations["trip_index"].astype(int) + 1 

    if len(convergence_records) == 0:
        df_convergence = pd.DataFrame(columns=["valid", "size"])
    else:
        df_convergence = pd.DataFrame.from_records(convergence_records, columns=["valid", "size"])

    logger.info("locations_v2 success rate: %f", df_convergence["valid"].mean() if len(df_convergence) else 0.0)
    return df_locations, df_convergence





















######################################################################
def get_platform_mp_context():
    """
    Get the appropriate multiprocessing context for the current platform.
    Linux prefers fork for speed; spawn remains required on Windows and macOS.
    """
    # import platform
    # if platform.system() == "Windows":
    #     return get_context("spawn")
    # elif platform.system() == "Darwin":
    #     return get_context("spawn")
    # else:  # Linux and other Unix-like systems
    #     return get_context("fork")
    return get_context("spawn")

def _should_compile_in_worker(mp_context):
    return mp_context.get_start_method() != "fork"
    
def _drain_progress_queue(progress, progress_queue):
    if progress_queue is None:
        return 0

    drained = 0
    while True:
        try:
            inc = progress_queue.get_nowait()
        except queue.Empty:
            break

        if inc is None:
            continue
        drained += int(inc)

    if drained > 0:
        progress.update(drained)
    return drained

def _init_location_worker(wrapper, worker_torch_threads, enable_mkldnn, 
                              enable_compile, compile_mode, progress_queue, progress_flush_every):  
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_PROGRESS_FLUSH_EVERY

    torch.backends.mkldnn.enabled = bool(enable_mkldnn)

    if worker_torch_threads is not None and int(worker_torch_threads) > 0:
        torch.set_num_threads(int(worker_torch_threads))
        torch.set_num_interop_threads(int(worker_torch_threads))

    # torch.compile artifacts do not survive pickling, so spawn workers must compile their own copy.
    # Fork workers can inherit the compiled wrapper from the parent process.
    if enable_compile:
        wrapper.optimize_wrapper(enable_compile=True, compile_mode=compile_mode)

    _WORKER_STATE["wrapper"] = wrapper
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_PROGRESS_FLUSH_EVERY = max(1, int(progress_flush_every))

def _assign_person_chunk(df_trips_chunk, df_meta_chunk, seed):
    wrapper = _WORKER_STATE.get("wrapper")

    rng = np.random.RandomState(seed)
    locations_records = []
    convergence_records = []

    if len(df_trips_chunk) == 0 or len(df_meta_chunk) == 0:
        return locations_records, convergence_records
    
    meta_cols = ["home_x", "home_y", "work_x", "work_y", "edu_x", "edu_y", "age", "sex", "employed", "income_class", 
                "car_availability", "has_work", "has_education",
                "work_destination_id","education_destination_id","home_destination_id"]
    meta_lookup = df_meta_chunk.set_index("person_id")[meta_cols].to_dict("index")
    person_groups = df_trips_chunk.groupby("person_id", sort=False)
    pending_progress = 0

    for person_id, grp in person_groups:
        info = meta_lookup.get(person_id)
        if info is None:
            continue

        home_x, home_y, work_x, work_y, edu_x, edu_y = info["home_x"], info["home_y"], info["work_x"], info["work_y"], info["edu_x"], info["edu_y"]
        has_work, has_education = info["has_work"], info["has_education"]        
        age, sex, employed, income_class, car_availability = info["age"], info["sex"], info["employed"], info["income_class"], info["car_availability"]
        work_destination_id, education_destination_id, home_destination_id = info["work_destination_id"], info["education_destination_id"], info["home_destination_id"]
        
        # This not only get the current coords, but also add a trip from primary location if the first trip is not from primary
        grp, (current_x, current_y), added_a_trip, origin_id = _get_first_location(grp, home_x, home_y, work_x, work_y, edu_x, edu_y, has_work, has_education,
                                                                                        work_destination_id, education_destination_id, home_destination_id)
        
        person_trip_count = len(grp)        
        following_purpose_arr = grp["following_purpose"].to_numpy()
        trip_index_arr = grp["trip_index"].to_numpy()
        departure_time_arr = grp["departure_time_normalized"].to_numpy()
        preceding_purpose_arr = grp["preceding_purpose"].to_numpy()
        activity_duration_arr = grp["activity_duration_h"].to_numpy()
        target_distance_arr = grp["target_distance"].to_numpy()
        
        chain_daily_longest = grp["daily_longest_distance_from_home"].iloc[0]  # static per person; same value for all trips
        chain_daily_total = grp["daily_crowfly_total"].iloc[0]  # static per person; same value for all trips
        chain_daily_longest_work = grp["daily_longest_distance_from_work"].iloc[0]  # static per person; same value for all trips
        activity_chain_vector = grp["activity_chain"].iloc[0]  # static per person; same array for all trips

        consumed_fore_trip_start = 0.0
        trip_pos = 0.0
        destination_id = origin_id
        trip_pos_inc = 1/max(1, person_trip_count - 1)        

        for local_idx in range(person_trip_count):
            following_purpose = following_purpose_arr[local_idx]
            trip_index = trip_index_arr[local_idx]

            if following_purpose == "home":
                next_x, next_y = home_x, home_y
                destination_id = home_destination_id

            elif following_purpose == "work" and has_work:
                next_x, next_y = work_x, work_y
                destination_id = work_destination_id

            elif following_purpose == "education" and has_education:
                next_x, next_y = edu_x, edu_y
                destination_id = education_destination_id

            elif following_purpose in SECONDARY_SET:
                target_distance = target_distance_arr[local_idx]
                if target_distance<10.0:# less than 10 meters, it doesn't move from current location
                    next_x, next_y = current_x, current_y                    
                    geom = Point(next_x, next_y)
                    locations_records.append((person_id, trip_index, destination_id, geom))
                    convergence_records.append((True, 1))
                else:
                    departure_time = departure_time_arr[local_idx]
                    origin_purpose = preceding_purpose_arr[local_idx]
                    activity_duration_h = activity_duration_arr[local_idx]                
                    # Predict first level coarse H3 cell
                    destination_id, geom = wrapper.predict(person_id=person_id, home_x=home_x, home_y=home_y, work_x=work_x, work_y=work_y, origin_x=current_x, origin_y=current_y, 
                                                        age=age, sex=sex, employed=employed, car_availability=car_availability, income_class=income_class, 
                                                        daily_longest_distance_from_home=chain_daily_longest, daily_crowfly_total=chain_daily_total,
                                                        daily_longest_distance_from_work=chain_daily_longest_work,
                                                        crowfly_consumed_before_trip=consumed_fore_trip_start, trip_position_class=trip_pos,
                                                        departure_time_normalized=departure_time,
                                                        activity_duration_h=activity_duration_h,
                                                        target_distance=target_distance,
                                                        activity_chain_vector=activity_chain_vector,
                                                        origin_purpose=origin_purpose,
                                                        purpose=following_purpose, has_work=has_work, has_education=has_education, rng=rng)

                    # sample a destination point within the predicted level 2 cell                 

                    # get coords and append to the list
                    next_x, next_y = float(geom.x), float(geom.y)
                    locations_records.append((person_id, trip_index, destination_id, geom))
                    convergence_records.append((True, 1))
            else:
                raise ValueError(f"Unexpected following purpose {following_purpose} for person_id {person_id} trip_index {trip_index}")                       
            
            if not (added_a_trip and local_idx==0):
                # if the first trip was added as a synthetic primary trip, we don't want to advance the position in the day or the consumed distance, since that trip doesn't really exist and shouldn't affect the features of the subsequent trips; for all other trips, including the first one if it was not added synthetically, we do want to advance these features as usual
                trip_pos += trip_pos_inc
                consumed_fore_trip_start += _euclidean(current_x, current_y, next_x, next_y)
            current_x, current_y = next_x, next_y

        if _WORKER_PROGRESS_QUEUE is not None:
            pending_progress += 1
            if pending_progress >= _WORKER_PROGRESS_FLUSH_EVERY:
                _WORKER_PROGRESS_QUEUE.put(pending_progress)
                pending_progress = 0

    if _WORKER_PROGRESS_QUEUE is not None and pending_progress > 0:
        _WORKER_PROGRESS_QUEUE.put(pending_progress)

    return locations_records, convergence_records






