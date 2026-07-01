import geopandas as gpd
import numpy as np
import pandas as pd
import os
from synthesis.population.spatial.secondary.components import CustomDistanceSampler, CustomDiscretizationSolver, CandidateIndex, CustomFreeChainSolver
from synthesis.population.spatial.secondary.problems import find_assignment_problems
from synthesis.population.spatial.secondary.rda import AssignmentSolver, DiscretizationErrorObjective, \
    GravityChainSolver, AngularTailSolver, GeneralRelaxationSolver
import logging

logger = logging.getLogger("synpp")

NUMBER_CANDIDATES = 20
ALPHA_PROBABILITIES = 0.8

def configure(context):
    context.stage("data.constants")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")

    context.stage("synthesis.population.spatial.secondary.distance_distributions")
    context.stage("synthesis.population.destinations")

    context.config("random_seed")
    context.config("threads")


def prepare_locations(context):
    df_home = context.stage("synthesis.population.spatial.home.locations").rename(columns={"geometry": "home"})
    df_work = context.stage("synthesis.population.spatial.primary.locations")[0].rename(columns={"geometry": "work"})
    df_education = context.stage("synthesis.population.spatial.primary.locations")[1].rename(columns={"geometry": "education"})

    df_locations = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = pd.merge(df_locations, df_home[["household_id", "home"]], how="left", on="household_id")
    df_locations = pd.merge(df_locations, df_work[["person_id", "work"]], how="left", on="person_id")
    df_locations = pd.merge(df_locations, df_education[["person_id", "education"]], how="left", on="person_id")

    return df_locations[["person_id", "home", "work", "education"]].sort_values(by="person_id")


def prepare_destinations(context):
    df_destinations = context.stage("synthesis.population.destinations")

    identifiers = df_destinations["destination_id"].values
    locations   = np.vstack(df_destinations["geometry"].apply(lambda x: np.array([x.x, x.y])).values)
    number_employees = df_destinations["number_employees"].values

    data = {}

    for purpose in ("shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"):
        f = df_destinations["offers_%s" % purpose].values

        data[purpose] = dict(
            identifiers=identifiers[f],
            locations=locations[f],
            ovgk=df_destinations["ovgk"].values[f],
            number_employees=np.nan_to_num(number_employees[f], nan=1.0, posinf=1.0, neginf=1.0)
        )

    return data


def resample_cdf(cdf, factor):
    if factor >= 0.0:
        cdf = cdf * (1.0 + factor * np.arange(1, len(cdf) + 1) / len(cdf))
    else:
        cdf = cdf * (1.0 + abs(factor) - abs(factor) * np.arange(1, len(cdf) + 1) / len(cdf))

    cdf /= cdf[-1]
    return cdf


def resample_distributions(distributions, factors):
    for mode, mode_distributions in distributions.items():
        for distribution in mode_distributions["distributions"]:
            distribution["cdf"] = resample_cdf(distribution["cdf"], factors[mode])


def execute(context):
    c    = context.stage("data.constants")
    crs  = c.CH1903_PLUS

    # Load car availability
    df_car_availability     = context.stage("synthesis.population.enriched")[["person_id","car_availability","driving_license"]]
    df_car_availability["car_availability"] = ((df_car_availability["car_availability"].astype(int)==1) &
                                               (df_car_availability["driving_license"]).astype(int)==1).astype(bool)        

    # Load trips and primary locations
    df_trips                = context.stage("synthesis.population.trips").sort_values(by=["person_id", "trip_index"])

    # Border-crossing agents are located directly from data.cross_border.swiss_residents_od
    # (see synthesis.population.spatial.locations), using the actual border-crossing point they
    # were matched to in synthesis.population.trips. "border" is not a destination purpose known
    # to this solver (see prepare_destinations), so these agents must be excluded here.
    is_cb_trip   = (df_trips["preceding_purpose"] == "border") | (df_trips["following_purpose"] == "border")
    cb_person_ids = set(df_trips.loc[is_cb_trip, "person_id"])
    df_trips     = df_trips[~df_trips["person_id"].isin(cb_person_ids)]

    df_trips                = pd.merge(df_trips, df_car_availability[["person_id","car_availability"]], how="left", on="person_id")
    df_trips["travel_time"] = df_trips["arrival_time"] - df_trips["departure_time"]
    df_primary              = prepare_locations(context)

    # Prepare data
    distance_distributions = context.stage("synthesis.population.spatial.secondary.distance_distributions")
    destinations = prepare_destinations(context)

    # Resampling for calibration
    resample_distributions(distance_distributions, dict(
        car=0.8, car_passenger=1.0, pt=1.0, bike=0.0, walk=0.0, walk_loop=0.0, bike_loop=0.0, car_loop=0.8, pt_loop=1.0, remote_walk = 0.0
    ))

    # Segment into subsamples (pt agents with car availability / without car availability in the same batch)
    processes = max(1, min(context.config("threads"), 24))

    unique_person_ids_carAvail = df_trips[df_trips["car_availability"] == True]["person_id"].unique()
    unique_person_ids_noCar = df_trips[df_trips["car_availability"] == False]["person_id"].unique()
    
    number_of_persons = len(unique_person_ids_carAvail) + len(unique_person_ids_noCar)
    processes_carAvail = int((len(unique_person_ids_carAvail)/number_of_persons) * processes)
    unique_person_ids_carAvail = np.array_split(unique_person_ids_carAvail, processes_carAvail)
    unique_person_ids_noCar = np.array_split(unique_person_ids_noCar, processes - processes_carAvail)
    unique_person_ids = unique_person_ids_carAvail + unique_person_ids_noCar
    assert len(unique_person_ids) == processes

    rng = np.random.RandomState(context.config("random_seed"))
    random_seeds = rng.randint(10000, size=processes)

    # Create batch problems for parallelization
    batches = []

    for index in range(processes):
        batch_trips = df_trips[df_trips["person_id"].isin(unique_person_ids[index])]
        batches.append((
            batch_trips,
            df_primary[df_primary["person_id"].isin(unique_person_ids[index])],
            random_seeds[index],
            crs,
            batch_trips["car_availability"].iloc[0] # all persons in batch have same car availability
        ))

    # Run algorithm in parallel
    with context.progress(label="Assigning secondary locations to persons", total=number_of_persons):
        with context.parallel(processes = processes, data=dict(
                distance_distributions = distance_distributions,
                destinations = destinations
        )) as parallel:
            df_locations, df_convergence = [], []

            for df_locations_item, df_convergence_item in parallel.imap_unordered(process, batches):
                df_locations.append(df_locations_item)
                df_convergence.append(df_convergence_item)

    df_locations = pd.concat(df_locations).sort_values(by=["person_id", "trip_index"])
    df_convergence = pd.concat(df_convergence)

    logger.info("Success rate: %f", df_convergence["valid"].mean())

    # df_locations.to_csv("/cluster/project/cmdp/asallard/analysis/Crossborder/MZ/secondary_destinations.csv", index=False)

    return df_locations, df_convergence


def process(context, arguments):
    df_trips, df_primary, random_seed, crs, car_availability = arguments

    # Set up RNG
    rng = np.random.RandomState(random_seed)

    # get destinations candidates
    destinations = context.data("destinations").copy()
    if not car_availability:
        # we add this condition here because people without car availability tend to
        # choose secondary location where public transport is more accessible (ovgk A, B, C, D)
        for k,v in destinations.items():
            mask = (v["ovgk"]=='A') | (v["ovgk"]=='B') | (v["ovgk"]=='C') | (v["ovgk"]=='D')
            v["identifiers"] = v["identifiers"][mask]
            v["locations"]   = v["locations"][mask]
            v["number_employees"] = v["number_employees"][mask]

    # drop ovgk as it is not needed in the candidate index
    for v in destinations.values():
        del v["ovgk"]            

    # Set up discretization solver
    candidate_index = CandidateIndex(destinations, number_candidates = NUMBER_CANDIDATES, alpha_probabilities = ALPHA_PROBABILITIES, random=rng)
    discretization_solver = CustomDiscretizationSolver(candidate_index)

    # Set up distance sampler
    distance_distributions = context.data("distance_distributions")
    distance_sampler = CustomDistanceSampler(
        maximum_iterations=1000,
        random=rng,
        distributions=distance_distributions)

    # Set up relaxation solver; currently, we do not consider tail problems.
    chain_solver = GravityChainSolver(
        random=rng, eps=75.0, lateral_deviation=10.0, alpha=0.1
    )

    tail_solver = AngularTailSolver(random = rng)
    free_solver = CustomFreeChainSolver(rng, candidate_index)

    relaxation_solver = GeneralRelaxationSolver(chain_solver, tail_solver, free_solver)

    # Set up assignment solver
    thresholds = dict(
        car=300.0, car_passenger=300.0, pt=300.0,
        bike=200.0, walk=200.0,
        bike_loop=200.0, walk_loop=200.0,
        car_loop=300.0, pt_loop=300.0,
        remote_walk = 200.0
    )

    assignment_objective = DiscretizationErrorObjective(thresholds=thresholds)
    assignment_solver = AssignmentSolver(
        distance_sampler=distance_sampler,
        relaxation_solver=relaxation_solver,
        discretization_solver=discretization_solver,
        objective=assignment_objective,
        maximum_iterations=200
    )

    person_ids = []
    trip_indices = []
    destination_ids = []
    x_coordinates = []
    y_coordinates = []
    df_convergence = []

    last_person_id = None

    for problem in find_assignment_problems(df_trips, df_primary):
        result = assignment_solver.solve(problem)

        starting_activity_index = problem["activity_index"]

        identifiers = np.asarray(result["discretization"]["identifiers"])
        locations = np.asarray(result["discretization"]["locations"])
        number_of_locations = len(identifiers)

        if number_of_locations > 0:
            person_ids.append(np.full(number_of_locations, problem["person_id"]))
            trip_indices.append(starting_activity_index + np.arange(number_of_locations))
            destination_ids.append(identifiers)
            x_coordinates.append(locations[:, 0])
            y_coordinates.append(locations[:, 1])

        df_convergence.append((
            result["valid"], problem["size"]
        ))

        if problem["person_id"] != last_person_id:
            last_person_id = problem["person_id"]
            context.progress.update()

    if len(person_ids) > 0:
        person_ids = np.concatenate(person_ids)
        trip_indices = np.concatenate(trip_indices)
        destination_ids = np.concatenate(destination_ids)
        x_coordinates = np.concatenate(x_coordinates)
        y_coordinates = np.concatenate(y_coordinates)
    else:
        person_ids = np.array([], dtype=np.int64)
        trip_indices = np.array([], dtype=np.int64)
        destination_ids = np.array([], dtype=object)
        x_coordinates = np.array([], dtype=float)
        y_coordinates = np.array([], dtype=float)

    df_locations = pd.DataFrame(dict(
        person_id=person_ids,
        trip_index=trip_indices,
        destination_id=destination_ids
    ))

    df_locations = gpd.GeoDataFrame(
        df_locations,
        geometry=gpd.points_from_xy(x_coordinates, y_coordinates),
        crs=crs
    )
    assert not df_locations["geometry"].isna().any()

    df_convergence = pd.DataFrame.from_records(df_convergence, columns=["valid", "size"])
    return df_locations, df_convergence
