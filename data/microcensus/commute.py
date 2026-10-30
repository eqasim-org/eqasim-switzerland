import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")


def use_diary_home(commutes, trips, persons):
    """Anchor education distance to an observed home in the diary, when possible.

    Prefer the last home activity before the primary education activity, then
    the first home afterwards. This follows diary order, not proximity to school,
    and includes home_secondary activities. Household coordinates remain an
    explicit fallback and a separate comparison distance.
    """
    result = commutes.merge(persons[["person_id", "home_x", "home_y"]],
                            on="person_id", validate="one_to_one")
    result["commute_household_home_distance"] = result["commute_home_distance"]
    result["commute_home_source"] = "household_fallback"
    diaries = {person: diary.sort_values("trip_id", kind="stable")
               for person, diary in trips.loc[trips.person_id.isin(result.person_id)].groupby("person_id")}
    for row in result.itertuples():
        diary = diaries.get(row.person_id)
        if diary is None:
            continue
        events, anchor = [], None
        for position, trip in enumerate(diary.itertuples()):
            for offset, purpose, x, y in [(0, trip.origin_purpose, trip.origin_x, trip.origin_y),
                                          (1, trip.purpose, trip.destination_x, trip.destination_y)]:
                if isinstance(purpose, str) and purpose.startswith("home") and np.isfinite(x) and np.isfinite(y):
                    events.append((2 * position + offset, x, y))
            if trip.trip_id == row.commute_trip_id:
                # Origin-only education is selected when no education destination
                # was recorded. Its activity precedes the selected trip.
                anchor = 2 * position + (1 if trip.purpose == "education" else 0)
        if anchor is None or not events:
            continue
        before = [event for event in events if event[0] < anchor]
        after = [event for event in events if event[0] > anchor]
        selected = before[-1] if before else (after[0] if after else None)
        if selected is None:
            continue
        result.loc[row.Index, ["home_x", "home_y"]] = selected[1:]
        result.loc[row.Index, "commute_home_source"] = "diary_preceding_home" if before else "diary_following_home"
    result["commute_home_distance"] = np.hypot(result.home_x - result.commute_x, result.home_y - result.commute_y)
    return result.rename(columns={"home_x": "commute_home_x", "home_y": "commute_home_y"})


def execute(context):
    df_trips = context.stage("data.microcensus.trips")[0].copy()
    df_persons = context.stage("data.microcensus.persons")

    commutes = {}

    for primary_purpose in ["work", "education"]:
        # Find the maximum work duration per person
        commuters = list(set(df_trips[(df_trips["purpose"]==primary_purpose) | (df_trips["origin_purpose"]==primary_purpose)]["person_id"].values))

        df_max_time_per_person = (df_trips[(df_trips["purpose"] == primary_purpose)][["person_id", "activity_duration"]]
                                  .groupby("person_id")
                                  .max()
                                  .reset_index())

        # Find the trips with the maximum duration
        df_commute = pd.merge(df_trips[df_trips["purpose"] == primary_purpose][["person_id", "trip_id",
                                                                       "mode", "activity_duration",
                                                                       "destination_x", "destination_y"]],
                              df_max_time_per_person,
                              on=["person_id", "activity_duration"]
                              ).groupby("person_id").first().reset_index()[["person_id", "trip_id",
                                                                            "mode", "activity_duration",
                                                                            "destination_x", "destination_y"]]
        
        # Rename columns
        df_commute.columns = ["person_id", "commute_trip_id",
                              "commute_mode", "commute_activity_duration",
                              "destination_x", "destination_y"]

        # Find the commute distance
        df_commute = pd.merge(df_commute,
                              df_persons[["person_id", "home_x", "home_y"]],
                              on="person_id")

        df_commute["commute_home_distance"] = np.sqrt((df_commute["home_x"] - df_commute["destination_x"]) ** 2 +
                                                      (df_commute["home_y"] - df_commute["destination_y"]) ** 2)

        # Rename and clean columns
        df_commute["commute_x"] = df_commute["destination_x"]
        df_commute["commute_y"] = df_commute["destination_y"]
        df_commute = df_commute[["person_id", "commute_trip_id", "commute_mode", "commute_home_distance",
                                 "commute_activity_duration", "commute_x", "commute_y"]]
        
        # Some people might have primary_purpose only as origin_purpose of the first trip
        # Let's add their info
        remaining_commuters = [x for x in commuters if not x in df_commute["person_id"].values]
        
        df_commute_remaining = df_trips[(df_trips["origin_purpose"] == primary_purpose) & (df_trips["person_id"].isin(remaining_commuters))][
            ["person_id", "trip_id",
             "mode", "departure_time",
             "origin_x", "origin_y"]
            ].groupby("person_id").first().reset_index()[["person_id", "trip_id",
                "mode", "departure_time",
                "origin_x", "origin_y"]]
        
        df_commute_remaining.columns = ["person_id", "commute_trip_id",
                              "commute_mode", "commute_activity_duration",
                              "origin_x", "origin_y"]
        
        df_commute_remaining = pd.merge(df_commute_remaining,
                              df_persons[["person_id", "home_x", "home_y"]],
                              on="person_id")
        
        df_commute_remaining["commute_home_distance"] = np.sqrt((df_commute_remaining["home_x"] - df_commute_remaining["origin_x"]) ** 2 +
                                                      (df_commute_remaining["home_y"] - df_commute_remaining["origin_y"]) ** 2)
        
        df_commute_remaining["commute_x"] = df_commute_remaining["origin_x"]
        df_commute_remaining["commute_y"] = df_commute_remaining["origin_y"]
        df_commute_remaining = df_commute_remaining[["person_id", "commute_trip_id", "commute_mode", "commute_home_distance",
                                 "commute_activity_duration", "commute_x", "commute_y"]]
        
        df_commute = pd.concat([df_commute, df_commute_remaining])

        if primary_purpose == "education":
            df_commute = use_diary_home(df_commute, df_trips, df_persons)

        commutes.update({primary_purpose: df_commute})

    return commutes
