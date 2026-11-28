"""Required trip fields, preserving the intentional no-parking-duration sentinel."""


def drop_missing_required_trip_fields(trips):
    # Missing parking duration means no charge can be calculated; the parking
    # cost function explicitly accepts it, including on the final trip home.
    required = trips.columns.difference(["parking_duration_wo_travelTime_min"])
    return trips.dropna(subset=required)
