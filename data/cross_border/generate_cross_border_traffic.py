import os

import folium
import geopandas as gpd
import numpy as np

# Free, no-key WMTS basemap - see analysis/pt/interactive_map.py's module
# docstring for why plain OSM/cartodbpositron tiles are not used here.
_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

# Cap the number of sampled activity chains drawn on the map (one full
# trajectory per person is several markers plus a line, so this keeps the
# file a reasonable size - see build_map).
MAP_SAMPLE_SIZE = 1000

TRAJECTORY_COLORS = {
    "From-To": "#1f77b4",
    "Through": "#ff7f0e",
}
DEFAULT_TRAJECTORY_COLOR = "#666666"


def purpose_color(purpose):
    if purpose == "border":
        return "#d62728"
    if purpose in ("home", "other"):
        return "#7f7f7f"
    return "#2ca02c"


def configure(context):
    context.config("include_cross_border", default = False)
    context.config("random_seed")

    if context.config("include_cross_border"):
        context.stage("data.cross_border.population")
        context.stage("data.cross_border.activities")
        context.stage("data.cross_border.vehicles")
        context.stage("synthesis.population.enriched")

    if context.config("include_external_population", default = False):
        context.stage("data.external_population.read_outputs")


def execute(context):
    if context.config("include_cross_border"):
        # Copies: the id renumbering below rewrites these frames in place, and
        # they are the objects the upstream stages returned.
        population = context.stage("data.cross_border.population").copy()
        activities = context.stage("data.cross_border.activities").copy()
        vehicles   = context.stage("data.cross_border.vehicles").copy()

        # Fix IDs
        id_person_max    = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
        id_household_max = np.max(context.stage("synthesis.population.enriched").copy()["household_id"].values)

        if context.config("include_external_population"):
            ext_pers, _, _ = context.stage("data.external_population.read_outputs")
            ext_pers = ext_pers.copy()
            id_person_max    = np.max(ext_pers["person_id"].values)
            id_household_max = np.max(ext_pers["household_id"].values)

        id_person_max    = max(id_person_max, id_household_max)  # just in case person_id and household_id are not on the same scale
        N                = id_person_max + 1

        # Adjust person_id
        population["new_person_id"] = range(N, N + len(population), 1)
        person_id_map               = population.set_index("person_id")["new_person_id"]

        population["person_id"]    = population["new_person_id"].values
        population["household_id"] = population["new_person_id"].values

        vehicles["owner_id"]    = vehicles["owner_id"].map(person_id_map).fillna(vehicles["owner_id"])
        vehicles["vehicle_id"]  = vehicles["owner_id"].astype(str) + ":" + vehicles["mode"]
        vehicles = vehicles[["owner_id", "vehicle_id", "age", "euro", "mode"]]

        activities["person_id"] = activities["person_id"].map(person_id_map)

        assert activities["person_id"].notna().all(), (
            "Some cross-border activities belong to a person that is not in data.cross_border.population."
        )

        population = population.drop(columns = ["new_person_id"])

        build_map(
            activities, context.config("random_seed"),
            os.path.join(context.path(), "generate_cross_border_traffic_map.html"),
        )

        return population, activities, vehicles


def sample_person_ids(person_labels, sample_size, rng):
    """
    Up to sample_size person ids, split evenly across every distinct label
    present (e.g. "From-To" and "Through") so a label with fewer agents than
    its even share still shows up on the map, rather than a single global
    random sample where rare labels could end up with few or zero agents
    drawn purely by chance. Any quota left over because a label didn't have
    enough agents to fill it is handed to whichever other label(s) still have
    more available.
    """
    labels = person_labels["label"].dropna().unique()
    if len(labels) == 0:
        return np.array([], dtype = person_labels["person_id"].dtype)

    per_label_quota = sample_size // len(labels)
    sampled_ids = []

    for label in labels:
        ids = person_labels.loc[person_labels["label"] == label, "person_id"].values
        quota = min(per_label_quota, len(ids))
        if quota > 0:
            sampled_ids.extend(rng.choice(ids, size = quota, replace = False))

    remaining_quota = sample_size - len(sampled_ids)
    if remaining_quota > 0:
        for label in labels:
            ids = person_labels.loc[person_labels["label"] == label, "person_id"].values
            not_yet_sampled = np.setdiff1d(ids, sampled_ids)
            extra = min(remaining_quota, len(not_yet_sampled))
            if extra > 0:
                sampled_ids.extend(rng.choice(not_yet_sampled, size = extra, replace = False))
                remaining_quota -= extra
            if remaining_quota <= 0:
                break

    return np.array(sampled_ids)


def build_map(activities, random_seed, output_path):
    """
    One folium map (same swisstopo basemap as data.cross_border.network_projection's
    own map): for a sample of up to MAP_SAMPLE_SIZE cross-border agents - both
    "From-To" and "Through" labels guaranteed to be represented, see
    sample_person_ids - their full activity chain as a line connecting every
    activity in order, plus a small dot per activity colored by purpose
    (gray: home/other, red: border crossing, green: everything else). One
    layer per label, toggleable, colored differently, so "From-To" commuting
    patterns and "Through" transit routes can be told apart at a glance.
    """
    rng = np.random.RandomState(random_seed)

    person_labels = activities[["person_id", "label"]].drop_duplicates()
    sampled_ids = sample_person_ids(person_labels, MAP_SAMPLE_SIZE, rng)

    df_map = activities[activities["person_id"].isin(sampled_ids)].copy()

    if len(df_map) == 0:
        return

    df_map = df_map.sort_values(["person_id", "activity_index"]).reset_index(drop = True)

    points_wgs84 = gpd.GeoSeries(df_map["geometry"].values, crs = "EPSG:2056").to_crs("EPSG:4326")
    df_map["lon"] = points_wgs84.x.values
    df_map["lat"] = points_wgs84.y.values

    center = [df_map["lat"].mean(), df_map["lon"].mean()]
    m = folium.Map(location = center, zoom_start = 9, tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    layers = {
        label: folium.FeatureGroup(name = "%s trajectories" % label, show = True)
        for label in df_map["label"].dropna().unique()
    }

    for person_id, group in df_map.groupby("person_id"):
        label = group["label"].iloc[0]
        layer = layers.get(label)
        if layer is None:
            continue

        folium.PolyLine(
            locations = list(zip(group["lat"], group["lon"])),
            color = TRAJECTORY_COLORS.get(label, DEFAULT_TRAJECTORY_COLOR),
            weight = 1.5, opacity = 0.6,
        ).add_to(layer)

        for _, row in group.iterrows():
            popup = (
                "Person %s (%s)<br>Activity %d: %s<br>%.0fs - %.0fs"
                % (person_id, label, row["activity_index"], row["purpose"], row["start_time"], row["end_time"])
            )
            folium.CircleMarker(
                location = (row["lat"], row["lon"]),
                radius = 3,
                color = purpose_color(row["purpose"]), fill = True, fill_opacity = 0.85,
                popup = folium.Popup(popup, max_width = 260),
                tooltip = row["purpose"],
            ).add_to(layer)

    for layer in layers.values():
        layer.add_to(m)
    folium.LayerControl(collapsed = False).add_to(m)

    m.save(output_path)
