"""Write cached vehicle types and vehicles as MATSim vehicles.xml.gz."""

import gzip
import io
import pandas as pd
import matsim.writers as writers
from synthesis.prepared.vehicles import TYPE_FIELDS, VEHICLE_FIELDS


def configure(context):
    context.stage("synthesis.prepared.vehicles")


def execute(context):

    # Read the complete fleet with resolved type references and export attributes.

    data = context.stage("synthesis.prepared.vehicles")
    output = "%s/vehicles.xml.gz" % context.path()

    # Serialize type definitions first, then vehicles referencing those types.

    with gzip.open(output, "wb") as stream:
        with io.BufferedWriter(stream, buffer_size=2 * 1024**3) as buffered:
            writer = writers.VehiclesWriter(buffered)
            writer.start_vehicles()
            for row in context.progress(
                data.types.itertuples(index=False),
                total=len(data.types),
                label="Vehicle types",
            ):
                writer.add_type(
                    row.type_id,
                    length=row.length,
                    width=row.width,
                    pce=row.pce,
                    nb_seats=row.nb_seats,
                    mode=row.mode,
                    maximum_velocity=_optional(row.maximum_velocity),
                    flow_efficiency_factor=_optional(row.flow_efficiency_factor),
                    engine_attributes=_engine_attributes(row),
                )
            for row in context.progress(
                data.vehicles.itertuples(index=False),
                total=len(data.vehicles),
                label="Vehicles",
            ):
                writer.add_vehicle(
                    row.vehicle_id, row.type_id, attributes=_vehicle_attributes(row)
                )
            writer.end_vehicles()
    return output


def _optional(value):
    """Represent an absent optional XML field without serializing NaN."""
    return None if pd.isna(value) else value


def _engine_attributes(row):
    """Translate emission descriptors into MATSim/HBEFA XML attribute names."""
    if pd.isna(row.hbefa_cat):
        return {}
    return {
        "HbefaVehicleCategory": row.hbefa_cat,
        "HbefaTechnology": row.hbefa_tech,
        "HbefaSizeClass": row.hbefa_size,
        "HbefaEmissionsConcept": row.hbefa_emission,
    }


def _vehicle_attributes(row):
    return {} if row.age is None else {"age": row.age, "euro": row.euro}
