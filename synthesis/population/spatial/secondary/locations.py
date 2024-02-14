import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geo

from synthesis.population.spatial.secondary.components import CustomDistanceSampler, CustomDiscretizationSolver, CandidateIndex, CustomFreeChainSolver
from synthesis.population.spatial.secondary.problems import find_assignment_problems
from synthesis.population.spatial.secondary.rda import AssignmentSolver, DiscretizationErrorObjective, \
    GravityChainSolver, AngularTailSolver, GeneralRelaxationSolver


def configure(context):
    context.stage("synthesis.population.SNN_mobility")

    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.home.locations")
    context.stage("synthesis.population.spatial.primary.locations")

    context.stage("synthesis.population.spatial.secondary.distance_distributions")
    context.stage("synthesis.population.destinations_schedule")

    if context.config("run_snn"):
        context.config("run_snn")
        context.config("snn_heuristic")

    context.config("random_seed")
    context.config("threads")
    context.config("use_detailed_activities")
    context.config("output_path")


def prepare_locations(context):
    # Load persons and their primary locations
    df_home = context.stage("synthesis.population.spatial.home.locations")

    if (not context.config("run_snn")) or (context.config("run_snn") and context.config("snn_heuristic") == 0):
        df_work, df_education = context.stage("synthesis.population.spatial.primary.locations")

    else:
        df_work, df_education, df_work_from_home = context.stage("synthesis.population.spatial.primary.locations")
        df_work_from_home = df_work.rename(columns={"geometry": "work_from_home"})

    df_home = df_home.rename(columns={"geometry": "home"})
    df_work = df_work.rename(columns={"geometry": "work"})
    df_education = df_education.rename(columns={"geometry": "education"})

    df_locations = context.stage("synthesis.population.sampled")[["person_id", "household_id"]]
    df_locations = pd.merge(df_locations, df_home[["household_id", "home"]], how="left", on="household_id")
    df_locations = pd.merge(df_locations, df_work[["person_id", "work"]], how="left", on="person_id")
    df_locations = pd.merge(df_locations, df_education[["person_id", "education"]], how="left", on="person_id")

    if context.config("run_snn"):
        if context.config("snn_heuristic") != 0:
            df_locations = pd.merge(df_locations, df_work_from_home[["person_id", "work_from_home"]], how="left", on="person_id")
            return df_locations[["person_id", "home", "work", "education", "work_from_home"]].sort_values(by="person_id")

    return df_locations[["person_id", "home", "work", "education"]].sort_values(by="person_id")


def prepare_destinations(context):
    df_destinations = context.stage("synthesis.population.destinations_schedule")
    M = np.max(df_destinations["destination_id"].values.tolist()) + 1
    det_activities = context.config("use_detailed_activities")

    data = {}
    if det_activities:# == "true":
        print("We shouldn't be here")
        df_home = context.stage("synthesis.population.spatial.home.locations").copy()[["household_id", "geometry"]].rename({"household_id": "destination_id"}, axis = 1)
        df_home.loc[:, "destination_id"] = np.array(range(M, M + len(df_home), 1))
        df_home.loc[:, "offers_visits"] = True
        df_home.loc[:, "offers_work"] = False
        df_home.loc[:, "offers_education"] = False
        df_home.loc[:, "offers_leisure"] = False
        df_home.loc[:, "offers_grocery"] = False
        df_home.loc[:, "offers_other(S)"] = False
        df_home.loc[:, "offers_culture"] = False
        df_home.loc[:, "education_type"] = False
        df_home.loc[:, "offers_religion"] = False
        df_home.loc[:, "offers_gastronomy"] = False
        df_home.loc[:, "offers_sport"] = False
        df_home.loc[:, "offers_other(L)"] = False
        df_home.loc[:, "offers_other"] = False
        df_home.loc[:, "offers_volunteer"] = False
        df_home.loc[:, "offers_outdoor"] = False
        df_home.loc[:, "destination_x"] = df_home["geometry"].apply(lambda x: x.x).values
        df_home.loc[:, "destination_y"] = df_home["geometry"].apply(lambda x: x.y).values
        df_home.loc[:, "number_employees"] = 1
        df_home.loc[:, "open_0-3"] = 1
        df_home.loc[:, "open_3-6"] = 1
        df_home.loc[:, "open_6-9"] = 1
        df_home.loc[:, "open_9-12"] = 1
        df_home.loc[:, "open_12-15"] = 1
        df_home.loc[:, "open_15-18"] = 1
        df_home.loc[:, "open_18-21"] = 1
        df_home.loc[:, "open_21-24"] = 1
        df_home = pd.DataFrame(df_home)

        df_destinations = pd.concat([df_destinations, df_home])
        identifiers = df_destinations["destination_id"].values
        locations = np.vstack(df_destinations["geometry"].apply(lambda x: np.array([x.x, x.y])).values)
        nb_employees = df_destinations["number_employees"].values
        df_destinations["offers_outdoor"] = df_destinations['offers_outdoor'].fillna(False)
        df_destinations["offers_services"] = df_destinations['offers_services'].fillna(False)
        df_destinations["number_employees"] = df_destinations['number_employees'].fillna(1)

        open03 = df_destinations["open_0-3"].values
        open36 = df_destinations["open_3-6"].values
        open69 = df_destinations["open_6-9"].values
        open912 = df_destinations["open_9-12"].values
        open1215 = df_destinations["open_12-15"].values
        open1518 = df_destinations["open_15-18"].values
        open1821 = df_destinations["open_18-21"].values
        open2124 = df_destinations["open_21-24"].values

        output_path = context.config("output_path")
        #df_destinations.to_csv("%s/destinations_sec_activities.csv" % output_path, index = None)

        for purpose in ("grocery", "other(S)", "culture", "gastronomy", "religion", "sport", "other(L)", "other", "visits", "volunteer", "outdoor", "services"):
            f = df_destinations["offers_%s" % purpose].values
            data[purpose] = dict(
                identifiers=identifiers[f],
                locations=locations[f],
                number_employees = nb_employees[f],
                open03 = open03[f],
                open36 = open36[f],
                open69 = open69[f],
                open912 = open912[f],
                open1215 = open1215[f],
                open1518 = open1518[f],
                open1821 = open1821[f],
                open2124 = open2124[f]
            )

    else:
        identifiers = df_destinations["destination_id"].values
        locations = np.vstack(df_destinations["geometry"].apply(lambda x: np.array([x.x, x.y])).values)

        for purpose in ("shop", "leisure", "other", "start_out_of_home"):
            if purpose == "start_out_of_home":
                f = df_destinations["offers_other"].values
            else:
                f = df_destinations["offers_%s" % purpose].values
            data[purpose] = dict(
                identifiers=identifiers[f],
                locations=locations[f]
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
    # Load trips and primary locations
    df_trips = context.stage("synthesis.population.SNN_mobility")[1].sort_values(by=["person_id", "trip_index"])
    
    df_trips["travel_time"] = df_trips["arrival_time"] - df_trips["departure_time"]
    df_primary = prepare_locations(context)

    # Prepare data
    distance_distributions = context.stage("synthesis.population.spatial.secondary.distance_distributions")
    destinations = prepare_destinations(context)

    # Resampling for calibration
    resample_distributions(distance_distributions, dict(
        car=0.8, car_passenger=1.0, pt=1.0, bike=0.0, walk=0.0
    ))

    # Segment into subsamples
    processes = context.config("threads")

    unique_person_ids = df_trips["person_id"].unique()
    number_of_persons = len(unique_person_ids)
    unique_person_ids = np.array_split(unique_person_ids, processes)

    rng = np.random.RandomState(context.config("random_seed"))
    random_seeds = rng.randint(10000, size=processes)

    # Create batch problems for parallelization
    batches = []
    
    for index in range(processes):
        batches.append((
            df_trips[df_trips["person_id"].isin(unique_person_ids[index])],
            df_primary[df_primary["person_id"].isin(unique_person_ids[index])],
            random_seeds[index]
        ))

    # Run algorithm in parallel
    with context.progress(label="Assigning secondary locations to persons", total=number_of_persons):
        with context.parallel(processes=processes, data=dict(
                distance_distributions=distance_distributions,
                destinations=destinations
        )) as parallel:
            df_locations, df_convergence = [], []

            for df_locations_item, df_convergence_item in parallel.imap_unordered(process, batches):
                df_locations.append(df_locations_item)
                df_convergence.append(df_convergence_item)

    df_locations = pd.concat(df_locations).sort_values(by=["person_id", "activity_index"])
    df_convergence = pd.concat(df_convergence)

    #df_locations["destination_id"] = [x[0] if type(x) == np.ndarray else x for x in df_locations["location_id"].values]
    print("Success rate:", df_convergence["valid"].mean())
    assert not df_locations["geometry"].isna().any()

    return df_locations, df_convergence


def process(context, arguments):
    df_trips, df_primary, random_seed = arguments

    # Set up RNG
    random = np.random.RandomState(context.config("random_seed"))
    rng = np.random.RandomState(random_seed)

    # Set up discretization solver
    destinations = context.data("destinations")
    candidate_index = CandidateIndex(destinations)
    discretization_solver = CustomDiscretizationSolver(candidate_index)
    #discretization_solver = CustomDiscretizationSolver(destinations)

    # Set up distance sampler
    distance_distributions = context.data("distance_distributions")
    distance_sampler = CustomDistanceSampler(
        maximum_iterations=1000,
        random=rng,
        distributions=distance_distributions)

    # Set up relaxation solver; currently, we do not consider tail problems.
    chain_solver = GravityChainSolver(
        random=rng, eps=10.0, lateral_deviation=10.0, alpha=0.1
    )

    tail_solver = AngularTailSolver(random = random)
    free_solver = CustomFreeChainSolver(random, candidate_index)

    relaxation_solver = GeneralRelaxationSolver(chain_solver, tail_solver, free_solver)

    # Set up assignment solver
    thresholds = dict(
        car=200.0, car_passenger=200.0, pt=200.0,
        bike=100.0, walk=100.0
    )

    assignment_objective = DiscretizationErrorObjective(thresholds=thresholds)
    assignment_solver = AssignmentSolver(
        distance_sampler=distance_sampler,
        relaxation_solver=relaxation_solver,
        discretization_solver=discretization_solver,
        objective=assignment_objective,
        maximum_iterations=20
    )

    df_locations = []
    df_convergence = []

    last_person_id = None

    for problem in find_assignment_problems(df_trips, df_primary):
        ai = problem["activity_index"]
        result = assignment_solver.solve(problem)

        assert(ai == problem["activity_index"])

        starting_activity_index = problem["activity_index"]

        for index, (identifier, location) in enumerate(
                zip(result["discretization"]["identifiers"], result["discretization"]["locations"])):
            df_locations.append((
                problem["person_id"], starting_activity_index + index, identifier, geo.Point(location)
            ))

        df_convergence.append((
            result["valid"], problem["size"]
        ))

        if problem["person_id"] != last_person_id:
            last_person_id = problem["person_id"]
            context.progress.update()


    df_locations = pd.DataFrame.from_records(df_locations,
                                             columns=["person_id", "activity_index", "destination_id", "geometry"])
    
    df_locations = gpd.GeoDataFrame(df_locations, crs="epsg:2056")
    assert not df_locations["geometry"].isna().any()

    df_convergence = pd.DataFrame.from_records(df_convergence, columns=["valid", "size"])
    return df_locations, df_convergence
