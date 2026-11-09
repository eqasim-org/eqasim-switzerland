import numpy as np

FIELDS = ["person_id", "trip_index", "preceding_purpose", "following_purpose", "mode", "travel_time"]
FIXED_PURPOSES = ["home", "work", "education"]


def find_bare_assignment_problems(df):
    has_activity_duration = "activity_duration" in df.columns
    fields = FIELDS + (["activity_duration"] if has_activity_duration else [])
    has_times = "arrival_time" in df and "departure_time" in df
    if has_times:
        fields += ["arrival_time", "departure_time"]
    problem = None
    previous_person, previous_arrival = None, 0.

    for row in df[fields].itertuples(index=False):
        person_id, trip_index, preceding_purpose, following_purpose, mode, travel_time = row[:6]
        activity_duration = row[6] if has_activity_duration else np.nan
        if has_times:
            arrival, departure = row[-2:]
            if person_id != previous_person:
                previous_arrival = 0.

        if not problem is None and person_id != problem["person_id"]:
            # We switch person, but we're still tracking a problem. This is a tail!
            yield problem
            problem = None

        if problem is None:
            # Start a new problem
            problem = dict(
                person_id=person_id, trip_index=trip_index, purposes=[preceding_purpose],
                modes=[], travel_times=[], activity_durations=[]
            )
            if has_times:
                problem["activity_start_times"] = [previous_arrival]
                problem["activity_end_times"] = [departure]

        if has_times:
            problem["activity_end_times"][-1] = departure
            problem["activity_start_times"].append(arrival)
            problem["activity_end_times"].append(max(30 * 3600., arrival + 60.))
            previous_person, previous_arrival = person_id, arrival

        problem["purposes"].append(following_purpose)
        problem["modes"].append(mode)
        problem["travel_times"].append(travel_time)
        problem["activity_durations"].append(activity_duration)

        if problem["purposes"][-1] in FIXED_PURPOSES:
            # The current chain (or initial tail) ends with a fixed activity.
            yield problem
            problem = None
    
    if not problem is None:
        yield problem


LOCATION_FIELDS = ["person_id", "home", "work", "education"]


def find_assignment_problems(df, df_locations):
    """
        Enriches assignment problems with:
          - Locations of the fixed activities
          - Size of the problem
          - Reduces purposes to the variable ones
    """
    location_iterator = df_locations[LOCATION_FIELDS].itertuples(index=False)
    current_location = None

    for problem in find_bare_assignment_problems(df):
        origin_purpose = problem["purposes"][0]
        destination_purpose = problem["purposes"][-1]
        if "activity_start_times" in problem:
            selection = slice(1 if origin_purpose in FIXED_PURPOSES else None,
                              -1 if destination_purpose in FIXED_PURPOSES else None)
            for field in ("activity_start_times", "activity_end_times"):
                problem[field] = np.asarray(problem[field][selection], dtype=float)

        # Reduce purposes
        if origin_purpose in FIXED_PURPOSES and destination_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][1:-1]

        elif origin_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][1:]

        elif destination_purpose in FIXED_PURPOSES:
            problem["purposes"] = problem["purposes"][:-1]

        else:
            #raise RuntimeError("The presented 'problem' is neither a chain nor a tail")
            pass # Neither chain nor tail

        # Define size
        problem["size"] = len(problem["purposes"])

        if problem["size"] == 0:
            continue  # We can skip if there are no variable activities

        # Advance location iterator until we arrive at the current problem's person
        while current_location is None or current_location[0] != problem["person_id"]:
            current_location = next(location_iterator)

        # Define origin and destination locations if they have fixed purposes
        problem["origin"]      = None
        problem["destination"] = None

        if origin_purpose in FIXED_PURPOSES:
            problem["origin"] = current_location[LOCATION_FIELDS.index(origin_purpose)]  # Shapely POINT
            problem["origin"] = np.array([[problem["origin"].x, problem["origin"].y]])

        if destination_purpose in FIXED_PURPOSES:
            problem["destination"] = current_location[LOCATION_FIELDS.index(destination_purpose)]  # Shapely POINT
            problem["destination"] = np.array([[problem["destination"].x, problem["destination"].y]])

        if problem["origin"] is None:
            problem["activity_index"] = problem["trip_index"]
        else:
            problem["activity_index"] = problem["trip_index"] + 1

        yield problem
