import geopandas as gpd

from data.cross_border.reference_files import reference_directory, file_versions


def configure(context):
    context.config("data_path")
    context.config("cross_border_reference_path", None)


def execute(context):

    # Read the entry layer from the same selected release as the exit reference.

    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    crossings = gpd.read_file(root / "border_interview_places_road_and_rail.gpkg")

    # The source file no longer provides a stable identifier per point, so we derive one
    # from the row order instead. These ids are used as MATSim facility ids (through
    # synthesis.population.destinations) and as the location of the border activities in
    # data.cross_border.activities. "label" (road/pt) indicates which mode a point serves
    # and is used downstream (data.cross_border.generate_od / swiss_residents_od) to match
    # car trips to "road" points and pt trips to "pt" points.

    crossings = crossings.reset_index(drop=True)
    crossings["border_crossing_point_id"] = "BCP_" + crossings.index.astype(str)
    crossings["importance"] = crossings["importance"].astype(float)

    # Return the canonical entry points without adding exit-only offices.

    return crossings


def validate(context):
    root = reference_directory(context.config("data_path"), context.config("cross_border_reference_path"))
    return file_versions([root / "border_interview_places_road_and_rail.gpkg"])
