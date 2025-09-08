

def get_cost(df, context):
    car_cost_per_km = context.config("car_cost_per_km") 
    return (car_cost_per_km * df["car_distance_km"])
