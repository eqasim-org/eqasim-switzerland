import numpy as np

def configure(context):
    context.config("include_cross_border", default = False)

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

        return population, activities, vehicles