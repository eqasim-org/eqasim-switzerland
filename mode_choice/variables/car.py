import os
from mode_choice.dmc_defaults import Defaults
from .walk import walk_travel_time, walk_distance


def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.network.car_router")
    context.stage("mode_choice.network.network_processor")
    context.stage("mode_choice.network.road_network")
    
    context.config("data_path")
    context.config("dmc_simulation_data_path", 
                   default = os.path.join(context.config("data_path"), "simulation_data"))
    
    context.config("routing_batch_size", default=Defaults.DEFAULT_CAR_ROUTING_BATCH_SIZE)

    context.config("walk_speed_m_per_s", default=Defaults.DEFAULT_WALK_SPEED_M_PER_S) 
    context.config("walk_distance_factor", default=Defaults.DEFAULT_WALK_DISTANCE_FACTOR) # factor to account for indirect walk paths

# TODO: save the router object to avoid rebuilding it every time, because it takes time.
# this is not straightforward, because pandana graphs are not serializable using pickle.

def car_variables(context, df):
    # prepare the network processor
    network_processor_class, path_to_load_network = context.stage("mode_choice.network.network_processor")
    network_processor = network_processor_class.load(path_to_load_network)
    
    # get and build the router
    router_class = context.stage("mode_choice.network.car_router")
    router = router_class(network_processor = network_processor)
    router.build()

    # route the trips
    routed_trips = router.router_trips_dataframe(df, 
                                                 congestion=True, 
                                                 batch_size=context.config("routing_batch_size"))

    routed_trips["travel_time_min"] = routed_trips["total_travel_time"] / 60
    routed_trips["distance_km"] = routed_trips["total_distance"] / 1000
    
    # access_egress time in min
    euclidean_access_egress_distance = routed_trips["access_euc_distance"] + routed_trips["egress_euc_distance"]
    acess_egress_distance_km = walk_distance(euclidean_access_egress_distance / 1000, context)
    routed_trips["access_egress_time_min"] = walk_travel_time(acess_egress_distance_km, context)
    
    return routed_trips


def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
             ['person_id', 'trip_id', 'origin_x', 'origin_y',  'destination_x', 'destination_y',  
              'departure_time', 'euclidean_distance_km']
        ]

    # get travel times
    routed_trips = car_variables(context, trips)

    return routed_trips[["person_id","trip_id",
                         "travel_time_min","access_egress_time_min",
                         "distance_km"]]
    