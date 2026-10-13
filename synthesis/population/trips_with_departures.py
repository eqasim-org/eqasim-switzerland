"""Optional optimized departures, downstream of the original trip synthesis."""


def configure(context):
    context.stage("synthesis.population.trips")
    context.config("correct_departure_time", True)
    if context.config("correct_departure_time"):
        context.stage("synthesis.population.departure_times.trips_departures")


def execute(context):
    trips = context.stage("synthesis.population.trips")
    if not context.config("correct_departure_time"):
        return trips.copy()
    return apply_departures(trips, context.stage("synthesis.population.departure_times.trips_departures"))


def apply_departures(trips, departures):
    """Resolve departures by trip key, preserving row order, IDs and metadata.

    The optimizer supplies departures only. Retain sampled arrival times, as
    the MATSim writer previously did, and recompute the derived trip duration.
    Missing corrections retain the sampled departure; input frames are untouched.
    """
    corrections = departures[["person_id", "trip_index", "best_departure_time"]].copy()
    corrections["best_departure_time"] = corrections["best_departure_time"].astype("float64")
    if trips.duplicated(["person_id", "trip_index"]).any() or corrections.duplicated(["person_id", "trip_index"]).any():
        raise ValueError("Duplicate person/trip keys in departure-time inputs")
    values = corrections["best_departure_time"]
    if (values.isin([float("inf"), -float("inf")]) | values.lt(0)).any():
        raise ValueError("Optimized departure times must be finite, nonnegative seconds")
    corrected = trips[["person_id", "trip_index", "departure_time"]].merge(
        corrections, on=["person_id", "trip_index"], how="left", validate="1:1", sort=False,
    )
    result = trips.copy()
    # Assign positionally to preserve the original (possibly non-default) index.
    result["departure_time"] = corrected["best_departure_time"].fillna(
        corrected["departure_time"]
    ).to_numpy()
    result["trip_duration"] = result["arrival_time"] - result["departure_time"]
    return result


def align_activities(activities, trips):
    """Apply final trip departures to output activities without changing the stage.

    Arrival times remain unchanged by departure optimization, so only the preceding
    activity's end time and duration need updating. Preserve non-travellers and
    final activities, and never mutate the original activity table.
    """
    result = activities.copy()
    departures = trips[["person_id", "trip_index", "departure_time"]].set_index(["person_id", "trip_index"])["departure_time"]
    if not departures.index.is_unique:
        raise ValueError("Duplicate person/trip keys in output trips")
    keys = result[["person_id", "activity_index"]].set_index(["person_id", "activity_index"]).index
    values = departures.reindex(keys).to_numpy()
    present = keys.isin(departures.index)
    result.loc[present, "end_time"] = values[present]
    result["duration"] = result["end_time"] - result["start_time"]
    return result
