import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from synthesis.population.spatial.secondary.components import CustomDistanceSampler, CustomDiscretizationSolver, CandidateIndex, CustomFreeChainSolver, DestinationAcceptance
from synthesis.population.spatial.secondary.problems import find_assignment_problems
from synthesis.population.spatial.secondary.opening_hours import encode_schedules
from synthesis.population.spatial.secondary.rda import AssignmentSolver, DiscretizationErrorObjective, \
    GravityChainSolver, AngularTailSolver, GeneralRelaxationSolver
import logging
import copy
from collections.abc import Mapping
from synthesis.population.spatial.secondary.fallback_repair import SwissFallbackRepair

logger = logging.getLogger("synpp")

NUMBER_CANDIDATES = 20
ALPHA_PROBABILITIES = 0.8
FRANCE_PURPOSES = ("shop", "leisure", "other")

def configure(context):
    context.stage("data.constants")
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.enriched")

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")

    context.stage("synthesis.population.spatial.secondary.distance_distributions")
    context.config("consider_opening_hours", default=True)
    tolerance = context.config("opening_hours_tolerance_minutes", default=10.)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("opening_hours_tolerance_minutes must be finite and non-negative")
    context.stage("synthesis.population.destinations_opening_hours" if context.config("consider_opening_hours")
                  else "synthesis.population.destinations", alias="secondary_destinations")

    context.config("consider_detour_factor_in_sl", default=True)
    if context.config("consider_detour_factor_in_sl"):
        context.stage("synthesis.population.spatial.secondary.detour_factors.factors")

    context.config("secondary_rda_france_acceptance", 1.0)
    context.config("secondary_rda_france_repair_attempts", 0)
    context.config("generate_outbound_flows", False)
    acceptance = france_acceptance(context.config("secondary_rda_france_acceptance"))
    repair_attempts = france_repair_attempts(context.config("secondary_rda_france_repair_attempts"))
    if repair_attempts:
        if not context.config("generate_outbound_flows") or not any(value < 1 for value in acceptance.values()):
            raise ValueError("French fallback repair requires outbound flows and active French proposal acceptance")
        context.stage("data.spatial.swiss_border")
    if context.config("generate_outbound_flows") and any(value < 1 for value in acceptance.values()):
        context.stage("data.locations_fr.secondary")

    context.config("random_seed")
    context.config("threads", volatile=True)


def execute(context):

    # Load coordinate constants.

    c    = context.stage("data.constants")
    crs  = c.CH1903_PLUS

    # Load car availability

    df_car_availability     = context.stage("synthesis.population.enriched")[["person_id","car_availability","driving_license"]].copy()
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

    # Duration of the activity reached by each trip; used by detour-factor heuristics.

    df_trips["activity_duration"] = df_trips.groupby("person_id")["departure_time"].shift(-1) - df_trips["arrival_time"]
    df_trips["activity_duration"] = df_trips["activity_duration"].fillna(0.0)
    df_trips.loc[df_trips["activity_duration"] < 0.0, "activity_duration"] = 0.0
    df_primary              = prepare_locations(context)

    # Prepare data

    distance_distributions = copy.deepcopy(context.stage("synthesis.population.spatial.secondary.distance_distributions"))
    destinations = prepare_destinations(context)

    # Resampling for calibration

    resample_distributions(distance_distributions, dict(
        car=0.8, car_passenger=1.0, pt=1.0, bike=0.0, walk=0.0, walk_loop=0.0, bike_loop=0.0, 
        car_loop=0.8, pt_loop=1.0, car_passenger_loop = 1.0, remote_walk = 0.0
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

    # Load detour factors if they are used

    detour_factor = None
    if context.config("consider_detour_factor_in_sl"):
        detour_factor = context.stage("synthesis.population.spatial.secondary.detour_factors.factors")

    # Run algorithm in parallel

    with context.progress(label="Assigning secondary locations to persons", total=number_of_persons):
        with context.parallel(processes = processes, data=dict(
                distance_distributions = distance_distributions,
                destinations = destinations,
                france_repair_attempts = france_repair_attempts(context.config("secondary_rda_france_repair_attempts")),
                detour_factor = detour_factor
                )) as parallel:
            df_locations, df_convergence = [], []

            for df_locations_item, df_convergence_item in parallel.imap_unordered(process, batches):
                df_locations.append(df_locations_item)
                df_convergence.append(df_convergence_item)

    df_locations = pd.concat(df_locations).sort_values(by=["person_id", "trip_index"])
    df_convergence = pd.concat(df_convergence)

    logger.info("Success rate: %f", df_convergence["valid"].mean())
    logger.info("RDA country preference: %d rejected valid proposals, %d valid fallbacks, %d invalid preference fallbacks",
                df_convergence["rejected_valid_candidates"].sum(),
                df_convergence["preference_fallback"].sum(),
                df_convergence["preference_invalid_fallback"].sum())
    logger.info("RDA French fallback repair: %d attempted chains, %d repaired, %d evaluated replacements",
                df_convergence["repair_attempted"].sum(), df_convergence["repair_success"].sum(),
                df_convergence["repair_trials"].sum())

    # Return stage results.

    return df_locations, df_convergence


def france_acceptance(value):
    """Accept scalar/purpose probabilities; omitted purposes are unrestricted."""
    values = dict(value) if isinstance(value, Mapping) else dict.fromkeys(FRANCE_PURPOSES, value)
    unknown = set(values) - set(FRANCE_PURPOSES)
    if unknown:
        raise ValueError("Unknown secondary_rda_france_acceptance purposes: %s" % sorted(unknown))
    result = dict.fromkeys(FRANCE_PURPOSES, 1.0)
    for purpose, probability in values.items():
        try:
            probability = float(probability)
        except (TypeError, ValueError) as error:
            raise ValueError("French proposal acceptance must be finite probabilities in [0, 1]") from error
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("French proposal acceptance must be finite probabilities in [0, 1]")
        result[purpose] = probability
    return result


def france_repair_attempts(value):
    """A zero default disables local fallback repair without changing RNG draws."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("secondary_rda_france_repair_attempts must be a nonnegative integer")
    return int(value)


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
    df_destinations = context.stage("secondary_destinations")
    acceptance = france_acceptance(context.config("secondary_rda_france_acceptance"))
    french_ids = set()
    if context.config("generate_outbound_flows") and any(value < 1 for value in acceptance.values()):
        french_ids = set(context.stage("data.locations_fr.secondary")["destination_id"])
        logger.info("French RDA proposal acceptance: %s", acceptance)
    is_french = df_destinations["destination_id"].isin(french_ids).to_numpy()
    repair_swiss = None
    if france_repair_attempts(context.config("secondary_rda_france_repair_attempts")):
        border = context.stage("data.spatial.swiss_border").to_crs(df_destinations.crs).geometry.union_all()
        shapely.prepare(border)
        repair_swiss = shapely.intersects(border, df_destinations.geometry.array) & ~is_french

    identifiers = df_destinations["destination_id"].values
    locations = np.column_stack((df_destinations.geometry.x.to_numpy(), df_destinations.geometry.y.to_numpy()))
    number_employees = df_destinations["number_employees"].values

    data = {}

    for purpose in ("shop", "leisure", "other", "work_secondary", "education_secondary", "home_secondary"):
        f = df_destinations["offers_%s" % purpose].values

        data[purpose] = dict(
            identifiers=identifiers[f],
            proposal_acceptance=np.where(is_french[f], acceptance.get(purpose, 1.0), 1.0),
            locations=locations[f],
            ovgk=df_destinations["ovgk"].values[f],
            number_employees=np.nan_to_num(number_employees[f], nan=1.0, posinf=1.0, neginf=1.0)
        )
        if context.config("consider_opening_hours") and purpose in ("shop", "leisure"):
            data[purpose]["opening_schedule"] = encode_schedules(df_destinations.loc[f, purpose + "_open_intervals"])
            data[purpose]["opening_tolerance_seconds"] = float(context.config("opening_hours_tolerance_minutes")) * 60.
        if repair_swiss is not None:
            data[purpose]["repair_swiss"] = repair_swiss[f]

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


def process(context, arguments):
    df_trips, df_primary, random_seed, crs, car_availability = arguments

    # Set up RNG
    rng = np.random.RandomState(random_seed)

    # get destinations candidates
    destinations = {purpose: values.copy() for purpose, values in context.data("destinations").items()}
    if not car_availability:
        # we add this condition here because people without car availability tend to
        # choose secondary location where public transport is more accessible (ovgk A, B, C, D)
        for k,v in destinations.items():
            mask = (v["ovgk"]=='A') | (v["ovgk"]=='B') | (v["ovgk"]=='C') | (v["ovgk"]=='D')
            for field, value in v.items():
                if np.ndim(value) > 0:
                    v[field] = value[mask]

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
        car=350.0, car_passenger=350.0, pt=320.0,
        bike=240.0, walk=200.0,
        bike_loop=200.0, walk_loop=200.0,
        car_loop=300.0, pt_loop=300.0,
        remote_walk = 200.0, detour_factor = 1.65
    )

    detour_factor = context.data("detour_factor")

    assignment_objective = DiscretizationErrorObjective(
        thresholds=thresholds, random=rng, detour_factor=detour_factor)
    proposal_acceptance = None
    if any(np.any(data.get("proposal_acceptance", np.array([1.0])) < 1) for data in destinations.values()):
        proposal_acceptance = DestinationAcceptance(
            destinations, np.random.RandomState((int(random_seed) + 1000003) % (2**32)))
    fallback_repair = None
    if any("repair_swiss" in data for data in destinations.values()):
        fallback_repair = SwissFallbackRepair(
            destinations, assignment_objective, random_seed, context.data("france_repair_attempts"),
            number_candidates=NUMBER_CANDIDATES, alpha_probabilities=ALPHA_PROBABILITIES)
    assignment_solver = AssignmentSolver(
        distance_sampler=distance_sampler,
        relaxation_solver=relaxation_solver,
        discretization_solver=discretization_solver,
        objective=assignment_objective,
        maximum_iterations=200,
        proposal_acceptance=proposal_acceptance,
        fallback_repair=fallback_repair
    )

    person_ids = []
    trip_indices = []
    destination_ids = []
    x_coordinates = []
    y_coordinates = []
    df_convergence = []

    last_person_id = None

    problem_trips = (df_trips if any("opening_schedule" in values for values in destinations.values())
                     else df_trips.drop(columns=["arrival_time", "departure_time"]))
    for problem in find_assignment_problems(problem_trips, df_primary):
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

        df_convergence.append(dict(
            person_id=problem["person_id"], activity_index=problem["activity_index"],
            size=problem["size"], **{key: result[key] for key in (
                "valid", "attempts", "valid_candidates", "rejected_valid_candidates",
                "preference_accepted", "preference_probability", "preference_fallback",
                "preference_invalid_fallback", "pre_repair_valid", "pre_repair_probability",
                "repair_attempted", "repair_trials", "repair_success", "repair_status"
            )}
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

    df_convergence = pd.DataFrame.from_records(df_convergence, columns=[
        "valid", "size", "person_id", "activity_index", "attempts", "valid_candidates",
        "rejected_valid_candidates", "preference_accepted", "preference_probability",
        "preference_fallback", "preference_invalid_fallback", "pre_repair_valid", "pre_repair_probability",
        "repair_attempted", "repair_trials", "repair_success", "repair_status"
    ])
    return df_locations, df_convergence


def validate(context):
    import hashlib
    from pathlib import Path
    root = Path(__file__).parent
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in
        ("opening_hours.py", "components.py", "problems.py", "rda.py"))).hexdigest()
