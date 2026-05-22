import io, gzip

import numpy as np
import pandas as pd

import matsim.writers as writers

def configure(context):
    context.stage("synthesis.vehicles.vehicles")

    context.config("include_cross_border", default = False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")
        context.stage("synthesis.population.enriched")

    context.config("include_external_population", default = False)
    if context.config("include_external_population"):
        context.stage("data.external_population.read_outputs")

TYPE_FIELDS = ["type_id", "nb_seats", "length", "width", "pce", "mode"]
VEHICLE_FIELDS = ["vehicle_id", "type_id", "age", "euro"]

def execute(context):
    output_path = "%s/vehicles.xml.gz" % context.path()

    df_vehicle_types, df_vehicles, df_trucks, df_lcv = context.stage("synthesis.vehicles.vehicles")
    df_vehicles = pd.concat([df_vehicles, df_trucks, df_lcv])

    if context.config("include_external_population"):
        external_vehicles   = context.stage("data.external_population.read_outputs")[2].copy()

        external_vehicles["type_id"] = "default_" + external_vehicles["mode"]

        df_vehicles = pd.concat([df_vehicles, external_vehicles])

    if context.config("include_cross_border"):
        cross_border_vehicles = context.stage("data.cross_border.generate_cross_border_traffic")[2].copy()
        cross_border_persons  = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()
        id_person_max         = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
        N                     = id_person_max + 1

        cross_border_persons["new_person_id"]    = range(N, N + len(cross_border_persons), 1)

        id_map = cross_border_persons.set_index("person_id")["new_person_id"]
        cross_border_vehicles["person_id"]  = cross_border_vehicles["vehicle_id"].str.split(":").str[0]
        cross_border_vehicles["type_id"]    = cross_border_vehicles["vehicle_id"].str.split(":").str[1]
        cross_border_vehicles["person_id"]  = cross_border_vehicles["person_id"].map(id_map).fillna(cross_border_vehicles["person_id"])
        cross_border_vehicles["vehicle_id"] = cross_border_vehicles["person_id"].astype(str) + ":" + cross_border_vehicles["type_id"]

        cross_border_vehicles.loc[cross_border_vehicles["type_id"] == "car", "type_id"] = "default_car"
        cross_border_vehicles.loc[cross_border_vehicles["type_id"] == "car_passenger", "type_id"] = "default_car_passenger"

        del cross_border_vehicles["person_id"]

        df_vehicles = pd.concat([df_vehicles, cross_border_vehicles])

    with gzip.open(output_path, 'wb+') as writer:
        with io.BufferedWriter(writer, buffer_size = 2 * 1024**3) as writer:
            writer = writers.VehiclesWriter(writer)
            writer.start_vehicles()

            with context.progress(total = len(df_vehicle_types), label = "Writing vehicles types ...") as progress:
                for type in df_vehicle_types.to_dict(orient="records"):
                    writer.add_type(
                        type["type_id"],
                        length=type["length"],
                        width=type["width"],
                        pce = type["pce"], #this wasn't here before, but it is supported by the writer, so I added it here
                        nb_seats = type["nb_seats"],
                        mode = type["mode"], #this wasn't here before, but it is supported by the writer, so I added it here
                        maximum_velocity = type["maxVelocity"] if ("maxVelocity" in type) and (type["maxVelocity"] is not None) else None,
                        flow_efficiency_factor = type["flowEfficiencyFactor"] if ("flowEfficiencyFactor" in type) and (type["flowEfficiencyFactor"] is not None) else None,
                        engine_attributes = {
                            "HbefaVehicleCategory": type["hbefa_cat"],
                            "HbefaTechnology": type["hbefa_tech"],
                            "HbefaSizeClass": type["hbefa_size"],
                            "HbefaEmissionsConcept": type["hbefa_emission"]
                        } if not pd.isna(type.get("hbefa_cat",np.nan)) else {},
                    ) 
                    progress.update()

            with context.progress(total = len(df_vehicles), label = "Writing vehicles ...") as progress:
                for vehicle in df_vehicles.to_dict(orient="records"):

                    writer.add_vehicle(
                        vehicle["vehicle_id"],
                        vehicle["type_id"],
                        attributes = {
                            "age": vehicle["age"],
                            "euro": vehicle["euro"]
                        } if (("age" in vehicle) and (vehicle["age"] is not None)) else {}
                    )
                    progress.update()

            writer.end_vehicles()

    return output_path