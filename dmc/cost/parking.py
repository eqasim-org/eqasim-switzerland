import numpy as np


def get_cost(df, context):
    parking_duration_min = df["parking_duration_wo_travelTime_min"].copy()
    is_last = df["is_last"]
    assert (parking_duration_min[is_last].isna()).all()

    parking_duration_min[~is_last] -= df.loc[~is_last, "car_travel_time_min"]
    df["actual_parking_duration_min"] = parking_duration_min.fillna(0.0)

    destination_urban = df.destination_municipality=="urban"
    destination_suburban = df.destination_municipality=="suburban"
    destination_home = df.destination_home
    destination_work = df.destination_work

    parking_cost = np.zeros(len(df))
    pay_parking_urban    = destination_urban & (~destination_home) & (parking_duration_min>60)
    pay_parking_suburban = destination_suburban & (~destination_home) & (parking_duration_min>60)

    parking_cost[pay_parking_urban]    = (parking_duration_min[pay_parking_urban]/60.0) * context.config("parking_cost_per_hour_CHF_urban")
    parking_cost[pay_parking_suburban] = (parking_duration_min[pay_parking_suburban]/60.0) * context.config("parking_cost_per_hour_CHF_suburban")
    parking_cost[destination_work] *= context.config("parking_price_reduction_for_work")

    parking_cost = np.clip(parking_cost, 0, 40)
    return parking_cost