import pandas as pd
import numpy as np
import geopandas as gpd
import os
from shapely import wkt
from shapely.geometry import Point
from data.utils import coerce_boolean_series
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import logging

logger = logging.getLogger(__name__)

AVAILABILITY_MAP = {"none": "never", "all": "always", "some": "always"}

PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex",
                 "home_x", "home_y",
                 "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke",
                 "household_id", "is_car_passenger", 
                 "statpop_person_id", "statpop_household_id", "mz_person_id", "mz_head_id", 
                 "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "income_class", "income_per_capita",
                 "number_of_cars_class", "number_of_bikes_class"]


def normalize_for_comparison(df):
    """
    Brings a column to a comparable form so that the different spellings of the
    same value (e.g. True/False, "True"/"False", 1/0) compare equal.
    """

    if df.dtype == object:
        df = df.replace({"True": True, "False": False, "true": True, "false": False})

    if df.dtype == bool:
        df = df.astype(int)

    return df.reset_index(drop = True)


def drop_duplicate_columns(df, source_name):
    """
    Removes the copies pandas creates when a CSV header repeats a column name:
    the second occurrence of "x" is read as "x.1", the third as "x.2", etc.
    The eqasim-france export currently writes has_driving_license twice (once
    as True/False, once as 1/0), and without this the copy would be carried
    along by every stage reading this data set. Only the first occurrence is
    kept; a copy that disagrees with it is reported, since that would point at
    two genuinely different attributes rather than a duplicated header.
    """

    duplicate_columns = []

    for column in df.columns:
        base, separator, suffix = column.rpartition(".")

        if separator == "" or not suffix.isdigit() or base not in df.columns:
            continue

        if not normalize_for_comparison(df[base]).equals(normalize_for_comparison(df[column])):
            logger.warning("Column %s in %s is a duplicate of %s but has different values; keeping %s.", column, source_name, base, base)
        else:
            logger.info("Dropping %s from %s: duplicate header entry for %s.", column, source_name, base)

        duplicate_columns.append(column)

    return df.drop(columns = duplicate_columns)


def configure(context):
    context.config("include_external_population", default = False)

    if context.config("include_external_population"):
        context.config("external_population_folder")
        context.config("fr_sample_rate", default = 1.0)
        context.config("input_downsampling")

        context.stage("data.constants")
        context.stage("synthesis.population.enriched")


def execute(context):
    if not context.config("include_external_population"):
        return

    folder = context.config("external_population_folder")
    c      = context.stage("data.constants")

    assert any(f.endswith("_persons.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_households.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_homes.gpkg") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_trips.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_activities.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"

    persons_file = next(f for f in os.listdir(folder) if f.endswith("_persons.csv"))
    persons      = pd.read_csv(os.path.join(folder, persons_file), sep = ";")
    persons      = drop_duplicate_columns(persons, persons_file)

    person_columns = ["person_id", "household_id", "age", "sex", "employed", "has_driving_license", "has_pt_subscription", "census_person_id", "hts_id"]

    # eqasim-france now writes car_availability per person (e.g. boat commuters
    # are individually forced to "none" while their household record keeps the
    # untouched household value), so the person-level value is the correct one
    # and overrides the household one further down. Older exports without the
    # column keep falling back to the household value.
    has_person_car_availability = "car_availability" in persons.columns

    if has_person_car_availability:
        persons = persons.rename(columns = {"car_availability": "person_car_availability"})
        person_columns.append("person_car_availability")
    else:
        logger.warning("No person-level car_availability found in %s, falling back to the household-level value.", persons_file)

    persons = persons[person_columns]

    households_file = next(f for f in os.listdir(folder) if f.endswith("_households.csv"))
    households      = pd.read_csv(os.path.join(folder, households_file), sep = ";")[["household_id", "car_availability", "bike_availability", "number_of_vehicles", "number_of_bikes", "income"]]

    trips_file = next(f for f in os.listdir(folder) if f.endswith("_trips.csv"))
    trips      = pd.read_csv(os.path.join(folder, trips_file), sep = ";")[["person_id", "preceding_activity_index", "mode"]]

    acts_file = next(f for f in os.listdir(folder) if f.endswith("_activities.csv"))
    acts      = pd.read_csv(os.path.join(folder, acts_file), sep = ";")[["person_id", "activity_index", "start_time", "end_time",
                                                                          "is_first", "is_last", "purpose", "location_id", "geometry",
                                                                          "commune_id", "population_density", "employee_density", "companies_density",
                                                                          "municipality_type", "ovgk"]]
    acts["is_first"] = coerce_boolean_series(acts["is_first"], name="is_first")
    acts["is_last"] = coerce_boolean_series(acts["is_last"], name="is_last")
    # pd.read_csv treats the literal string "None" (the fallback value written
    # by data.spatial.ovgk for unclassified locations) as a missing value by
    # default - restore it so it doesn't end up as a raw NaN in population.xml.
    acts["ovgk"] = acts["ovgk"].fillna("None").astype("category")

    acts["geometry"] = acts["geometry"].apply(wkt.loads)
    acts = gpd.GeoDataFrame(acts, geometry="geometry", crs="EPSG:2154")
    acts = acts.to_crs("EPSG:2056")

    acts["destination_x"] = acts["geometry"].apply(lambda g: g.x)
    acts["destination_y"] = acts["geometry"].apply(lambda g: g.y)

    vehicles_file = next(f for f in os.listdir(folder) if f.endswith("_vehicles.csv"))
    vehicles      = pd.read_csv(os.path.join(folder, vehicles_file), sep = ";")
    vehicles      = drop_duplicate_columns(vehicles, vehicles_file)

    homes_file = next(f for f in os.listdir(folder) if f.endswith("_homes.gpkg"))
    homes      = gpd.read_file(os.path.join(folder, homes_file))[["household_id", "geometry"]]
    homes.crs  = "EPSG:2154"
    homes      = homes.to_crs("EPSG:2056")

    homes["home_x"] = homes.geometry.x
    homes["home_y"] = homes.geometry.y

    households = households.merge(homes[["household_id", "home_x", "home_y"]], on = "household_id", how = "left")
    households.loc[:, "car_availability"]  = households["car_availability"].map(AVAILABILITY_MAP)
    households.loc[:, "bike_availability"] = households["bike_availability"].map(AVAILABILITY_MAP)

    # Match the income_class shares of the Swiss population instead of using
    # fixed CHF thresholds: the poorest X% of French households are assigned
    # to the income classes that make up the poorest X% of the Swiss
    # population, regardless of the CHF boundaries this implies.
    df_enriched  = context.stage("synthesis.population.enriched")
    swiss_shares = df_enriched["income_class"].value_counts(normalize=True).sort_index()

    class_labels = swiss_shares.index.astype(int).to_numpy()
    cum_shares   = swiss_shares.cumsum().to_numpy()

    # Weight by household size so the resulting *population* shares (not
    # household shares) match the Swiss population shares.
    household_size = persons.groupby("household_id").size().rename("household_size")
    households      = households.merge(household_size, on = "household_id", how = "left")
    households["household_size"] = households["household_size"].fillna(1)

    order        = households["income"].sort_values(kind = "mergesort").index
    cum_population = households.loc[order, "household_size"].cumsum()
    total_population = cum_population.iloc[-1]

    thresholds = np.round(cum_shares * total_population).astype(int)
    thresholds[-1] = total_population  # guard against rounding

    class_idx = np.searchsorted(thresholds, cum_population.to_numpy(), side = "left")
    class_idx = np.clip(class_idx, 0, len(class_labels) - 1)

    households.loc[order, "income_class"] = class_labels[class_idx]
    households["income_class"] = households["income_class"].astype(int)

    plot_income_class_distribution(
        swiss_shares, c.INCOME_CLASS_MAP,
        output_path = os.path.join(context.path(), "income_class_comparison.png")
    )

    households["number_of_cars_class"]  = households["number_of_vehicles"]
    households["number_of_bikes_class"] = households["number_of_bikes"]

    persons = persons.merge(households, on = "household_id", how = "left")

    # The person-level car_availability wins over the household-level one; the
    # household value only fills in for persons without a valid own value.
    if has_person_car_availability:
        person_car_availability = persons["person_car_availability"].map(AVAILABILITY_MAP)

        number_of_overrides = int((person_car_availability.notna() & (person_car_availability != persons["car_availability"])).sum())
        logger.info("Overriding household-level car_availability with the person-level value for %d/%d external persons.", number_of_overrides, len(persons))

        persons["car_availability"] = person_car_availability.fillna(persons["car_availability"])
        persons = persons.drop(columns = ["person_car_availability"])

    # OECD-modified equivalence scale: 1 for the first adult, 0.5 for each
    # additional adult, 0.3 for each child (age < 14).
    persons["is_child"] = persons["age"] < 14
    num_children        = persons.groupby("household_id")["is_child"].transform("sum")
    num_adults          = persons["household_size"] - num_children
    equivalent_size      = 1 + 0.5 * (num_adults - 1) + 0.3 * num_children
    persons["income_per_capita"] = persons["income"] / equivalent_size

    car_passenger_ids = set(trips.loc[trips["mode"] == "car_passenger", "person_id"])
    persons["is_car_passenger"] = persons["person_id"].isin(car_passenger_ids)

    persons["statpop_person_id"]    = persons["census_person_id"]
    persons["statpop_household_id"] = persons["census_person_id"]

    persons["mz_person_id"] = persons["hts_id"]
    persons["mz_head_id"]   = persons["hts_id"]

    persons["driving_license"]       = persons["has_driving_license"]
    persons["subscriptions_ga"]      = False
    persons["subscriptions_halbtax"] = False
    persons["subscriptions_strecke"] = False
    persons["subscriptions_verbund"] = persons["has_pt_subscription"]

    acts.loc[acts["is_first"], "start_time"] = 0
    acts.loc[acts["is_last"], "end_time"]    = 30*3600
    acts.loc[:, "duration"] = acts["end_time"] - acts["start_time"]
    acts = acts.merge(trips.rename(columns = {"preceding_activity_index": "activity_index", "mode": "following_mode"}),  on = ["person_id", "activity_index"], how = "left")

    # correct loop modes attributes in persons dataframe
    loop_flags = (acts[acts["following_mode"].str.contains("loop", na=False)]
                    .assign(value=True)
                    .pivot_table(
                            index="person_id",
                            columns="following_mode",
                            values="value",
                            aggfunc="any",
                            fill_value=False,
                        )
                     .rename(columns=lambda x: f"has_{x}_trip")
                     )

    persons = persons.merge(loop_flags,left_on="person_id", right_index=True, how="left",)
    persons[loop_flags.columns] = persons[loop_flags.columns].fillna(False)
    persons = persons[PERSON_FIELDS]
    
    # continue
    valid_ids = acts.groupby("person_id")["geometry"].apply(
        lambda g: g.notna().all()
    )
    valid_ids = valid_ids[valid_ids].index

    persons = persons[persons["person_id"].isin(valid_ids)]

    # location_id already comes out of eqasim-france as a canonical id, so it
    # can be used directly instead of re-deriving a local "FR_<n>" id.
    acts["destination_id"]    = acts["location_id"]
    acts["municipality_id"]   = acts["commune_id"]

    # Adjust IDS
    id_person_max    = np.max(df_enriched["person_id"].values)
    id_household_max = np.max(df_enriched["household_id"].values)
    id_person_max    = max(id_person_max, id_household_max)  # just in case person_id and household_id are not on the same scale
    N                = id_person_max + 1

    # 1. Adjust household_id first, one new id per distinct original
    # household so real multi-member households (unlike cross-border
    # commuters) stay grouped together instead of becoming singletons.
    unique_household_ids = persons["household_id"].unique()
    household_id_map      = pd.Series(range(N, N + len(unique_household_ids)), index = unique_household_ids)
    persons["household_id"] = persons["household_id"].map(household_id_map)

    # 2. Adjust person_id in a disjoint range (one new id per person)
    M = N + len(unique_household_ids)
    persons["new_person_id"] = range(M, M + len(persons), 1)
    person_id_map            = persons.set_index("person_id")["new_person_id"]

    persons["person_id"] = persons["new_person_id"].values

    vehicles["owner_id"]    = vehicles["owner_id"].map(person_id_map).fillna(vehicles["owner_id"])
    vehicles["vehicle_id"]   = vehicles["owner_id"].astype(str) + ":" + vehicles["mode"]
    vehicles = vehicles[["owner_id", "vehicle_id", "age", "euro", "mode"]]

    acts["person_id"] = acts["person_id"].map(person_id_map).fillna(acts["person_id"])

    # 2. Destination id

    homes = acts[acts["purpose"] == "home"].copy()

    homes["destination_id"] = ["home" + str(person_id) for person_id in homes["person_id"].values.tolist()]

    home_coords = persons.groupby("person_id")[["home_x", "home_y"]].first()
    homes["destination_x"] = homes["person_id"].map(home_coords["home_x"])
    homes["destination_y"] = homes["person_id"].map(home_coords["home_y"])

    acts_not_home = acts[acts["purpose"] != "home"].copy()
    facility_locations = acts_not_home.groupby("destination_id")[["destination_x", "destination_y"]].first()
    acts_not_home["destination_x"] = acts_not_home["destination_id"].map(facility_locations["destination_x"])
    acts_not_home["destination_y"] = acts_not_home["destination_id"].map(facility_locations["destination_y"])

    acts = pd.concat([homes, acts_not_home])
    acts = acts.sort_values(by = ["person_id", "activity_index"])

    acts["geometry"] = acts.apply(
        lambda r: Point(r["destination_x"], r["destination_y"]), axis=1
    )

    #acts.to_csv("/cluster/project/cmdp/asallard/theacts.csv", index=False)

    # Fix missing vehicles
    modes = ["car", "car_passenger", "bike"]
    all_persons = persons["person_id"].unique()

    df_required = pd.MultiIndex.from_product(
        [all_persons, modes], names=["owner_id", "mode"]
    ).to_frame(index=False)

    df_required["vehicle_id"] = df_required["owner_id"].astype(str) + ":" + df_required["mode"]

    # Find missing
    existing = set(zip(vehicles["owner_id"], vehicles["mode"]))
    df_missing = df_required[~df_required.apply(
        lambda r: (r["owner_id"], r["mode"]) in existing, axis=1
    )].copy()

    # Add missing vehicles (age=0 or whatever default)
    df_missing["age"]  = 0
    df_missing["euro"] = 6

    vehicles = pd.concat([vehicles, df_missing], ignore_index=True)

    fr_sample_rate = context.config("fr_sample_rate")
    ch_sample_rate = context.config("input_downsampling")
    ratio          = ch_sample_rate / fr_sample_rate

    if ratio > 1:
        logger.warning("The requested sample size for the Swiss population exceeds the sample size used for the generation of the French population. We might find a solution for this at some point but as of now we are keeping the French population unchanged.")

    elif ratio < 1:
        print(f"FR sample rate: {fr_sample_rate}. CH sample rate: {ch_sample_rate}.")
        print(f"Downsampling with a ratio of {round(ratio, 2)}.")

        person_ids  = persons["person_id"].values.tolist()
        sampled_ids = np.random.choice(person_ids, size = int(len(person_ids) * ratio), replace = False).tolist()

        persons    = persons[persons["person_id"].isin(sampled_ids)]
        acts       = acts[acts["person_id"].isin(sampled_ids)]
        vehicles   = vehicles[vehicles["owner_id"].isin(sampled_ids)]

    return persons, acts, vehicles


def plot_income_class_distribution(swiss_shares, income_class_map, output_path = None):
    """
    Visualizes the income_class distribution as a single CHF-denominated
    histogram. The Swiss and French income_class distributions are identical
    by construction (see the matching logic around households["income_class"]
    above: French households are assigned classes to match the Swiss
    population shares), so only one histogram is drawn.

    income_class_map (data.constants's INCOME_CLASS_MAP) gives the midpoint of
    each 2'000 CHF-wide income bracket, e.g. class 1 -> 3'000 means the
    bracket 2'000-4'000. Classes 0 and 8 are open-ended ("less than 2'000"
    and "more than 16'000"), so their bracket only has one real edge.

    Parameters
    ----------
    swiss_shares : pd.Series
        Share of the population per income_class, e.g. the result of
        `df_enriched["income_class"].value_counts(normalize=True)`.
    income_class_map : dict
        Maps income_class to its CHF bracket midpoint, i.e.
        `data.constants`'s `INCOME_CLASS_MAP` (c.INCOME_CLASS_MAP).
    output_path : str, optional
        If given, the figure is saved there instead of being returned.
    """

    classes = np.sort(np.array(list(income_class_map.keys())))
    values  = np.array([income_class_map[c] for c in classes], dtype = float)

    # Reconstruct bracket edges from the midpoints: each bracket is 2'000 CHF
    # wide, except class 0 which starts at 0 (its value is already the edge).
    edges       = np.empty(len(values) + 1)
    edges[0]    = 0
    edges[1]    = values[0]
    edges[2:]   = values[1:] + 1000
    widths      = np.diff(edges)

    shares = swiss_shares.reindex(classes).fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize = (10, 5))
    ax.bar(edges[:-1], shares, width = widths, align = "edge", color = "steelblue", edgecolor = "white")

    ax.set_xlim(0, edges[-1])
    tick_labels     = [f"{int(e):,}" for e in edges]
    tick_labels[-2] += "+"  # class 8 is open-ended starting from this edge
    tick_labels[-1] = ""    # the outer edge is a nominal width, not a real threshold
    ax.set_xticks(edges)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Household income (CHF)")

    ax.set_ylabel("Share of population")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1))
    ax.set_title("Income class distribution (Swiss and French populations, matched by construction)")

    class_axis = ax.secondary_xaxis(-0.15)
    class_axis.set_xticks((edges[:-1] + edges[1:]) / 2)
    class_axis.set_xticklabels([str(c) for c in classes])
    class_axis.set_xlabel("Income class")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
        plt.close(fig)
        return None

    return fig, ax


