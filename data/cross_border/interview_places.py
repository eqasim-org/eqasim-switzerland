import geopandas as gpd


def configure(context):
    context.config("data_path")


def execute(context):
    data_path = context.config("data_path")
    crossings = f"{data_path}/crossborder/border_interview_places_road_and_rail.gpkg"
    crossings = gpd.read_file(crossings)

    # The source file no longer provides a stable identifier per point, so we derive one
    # from the row order instead. "label" (road/pt) indicates which mode a point serves and
    # is used downstream (data.cross_border.generate_od / swiss_residents_od) to match
    # car trips to "road" points and pt trips to "pt" points.
    crossings = crossings.reset_index(drop=True)
    crossings["border_crossing_point_id"] = "BCP_" + crossings.index.astype(str)
    crossings["importance"] = crossings["importance"].astype(float)

    return crossings