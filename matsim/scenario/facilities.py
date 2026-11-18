"""Write prepared facilities as MATSim facilities.xml.gz."""

import gzip
import io
import matsim.writers


def configure(context):
    context.stage("synthesis.prepared.facilities")


def execute(context):

    # Read final IDs, coordinates and activity options from the preparation stage.

    facilities = context.stage("synthesis.prepared.facilities").facilities
    output = "%s/facilities.xml.gz" % context.path()

    # Serialize facilities without interpreting raw population or business columns.

    with gzip.open(output, "wb") as stream:
        with io.BufferedWriter(stream, buffer_size=2 * 1024**3) as buffered:
            writer = matsim.writers.FacilitiesWriter(buffered)
            writer.start_facilities()
            for row in context.progress(
                facilities.itertuples(index=False), total=len(facilities)
            ):
                writer.start_facility(row.facility_id, row.x, row.y)
                for purpose in row.activity_options:
                    writer.add_activity(purpose)
                writer.end_facility()
            writer.end_facilities()
    return output
