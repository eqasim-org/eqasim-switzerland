def configure(context):
    context.config("random_seed")

    context.stage("data.cross_border.destinations")
    context.stage("data.microcensus.persons")


PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", 
                 "home_x", "home_y",
                 "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke",
                 "household_id", "is_car_passenger", 
                 "statpop_person_id", "statpop_household_id", "mz_person_id", "mz_head_id", 
                 "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "income_class",
                 "number_of_cars_class", "number_of_bikes_class",
                 "origin_country", "destination_country", "origin_country_raw", "destination_country_raw"]


def execute(context):
    df         = context.stage("data.cross_border.destinations")
    mz_persons = context.stage("data.microcensus.persons")

    population = df[["cross_border_person_id", "mz_person_id", "origin_x", "origin_y",
                      "origin_country", "destination_country",
                      "origin_country_raw", "destination_country_raw"]].copy()

    population["household_id"]          = population["cross_border_person_id"].values 
    population["home_x"]                = population["origin_x"].values 
    population["home_y"]                = population["origin_y"].values 
    population["subscriptions_ga"]      = False
    population["subscriptions_halbtax"] = False

    population["has_walk_loop_trip"]          = False
    population["has_car_passenger_loop_trip"] = False
    population["has_car_loop_trip"]           = False
    population["has_bike_loop_trip"]          = False
    population["has_pt_loop_trip"]            = False

    population["mz_head_id"]           = population["mz_person_id"].values
    population["statpop_person_id"]    = 0
    population["statpop_household_id"] = 0

    mz_persons = mz_persons[["person_id", "age", "sex", "car_availability",
                             "employed", "driving_license", 
                             "subscriptions_verbund", "subscriptions_strecke",
                             "is_car_passenger", "income_class",
                             "number_of_cars_class", "number_of_bikes_class"]]
    
    population = population.merge(mz_persons,
                                  how = "left",
                                  left_on = "mz_person_id",
                                  right_on = "person_id")
    
    del population["person_id"]
    population["person_id"] = population["cross_border_person_id"]
    
    population = population[PERSON_FIELDS]

    return population