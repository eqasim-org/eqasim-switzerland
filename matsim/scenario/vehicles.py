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

TYPE_FIELDS = ["type_id", "nb_seats", "length", "width", "pce", "mode"]
VEHICLE_FIELDS = ["vehicle_id", "type_id", "age", "euro"]

def execute(context):
    output_path = "%s/vehicles.xml.gz" % context.path()

    df_vehicle_types, df_vehicles, df_trucks = context.stage("synthesis.vehicles.vehicles")
    df_vehicles = pd.concat([df_vehicles, df_trucks])

    if context.config("include_cross_border"):
        cross_border_vehicles = context.stage("data.cross_border.generate_cross_border_traffic")[2].copy()
        cross_border_persons  = context.stage("data.cross_border.generate_cross_border_traffic")[0].copy()
        id_person_max         = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
        N                     = id_person_max + 1

        cross_border_persons["new_person_id"]    = range(N, N + len(cross_border_persons), 1)

        id_map = cross_border_persons.set_index("person_id")["new_person_id"]
        cross_border_vehicles["person_id"]    = cross_border_vehicles["vehicle_id"].str.split(":").str[0]
        cross_border_vehicles["vehicle_type"] = cross_border_vehicles["vehicle_id"].str.split(":").str[1]
        cross_border_vehicles["person_id"]    = cross_border_vehicles["person_id"].map(id_map).fillna(cross_border_vehicles["person_id"])
        cross_border_vehicles["vehicle_id"]   = cross_border_vehicles["person_id"].astype(str) + ":" + cross_border_vehicles["vehicle_type"]

        del cross_border_vehicles["person_id"]
        del cross_border_vehicles["vehicle_type"]

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
                        engine_attributes = {
                            "HbefaVehicleCategory": type["hbefa_cat"],
                            "HbefaTechnology": type["hbefa_tech"],
                            "HbefaSizeClass": type["hbefa_size"],
                            "HbefaEmissionsConcept": type["hbefa_emission"]
                        }
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
                        }
                    )
                    progress.update()

            writer.end_vehicles()

    return output_path