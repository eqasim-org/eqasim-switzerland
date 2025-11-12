import os

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.network.car_router")
    context.stage("mode_choice.network.network_processor")
    context.stage("mode_choice.network.road_network")
    
    context.config("data_path")

    context.config("dmc_network_file", 
                   default=os.path.join(context.config("data_path"), "dmc", "switzerland_network.xml.gz"))
    context.config("dmc_congestion_file", 
                   default=os.path.join(context.config("data_path"), "dmc", "linkstats.txt.gz"))    
    context.config("dmc_graph_type", default="pandana")
    context.config("routing_batch_size", default=4096)

    context.config("walk_speed_m_per_s", default=1.3) 
    context.config("walk_distance_factor", default=1.3) # factor to account for indirect walk paths
    context.config("car_distance_factor", 1.4) # factor to account for indirect car paths

def execute(context):
    # get network and congestion file paths
    network_file = context.config("dmc_network_file")
    congestion_file = context.config("dmc_congestion_file")
    
    # get the road network
    road_network = context.stage("mode_choice.network.road_network")

    # prepare the network processor
    network_processor_class = context.stage("mode_choice.network.network_processor")
    network_processor = network_processor_class(
            network_file=network_file,
            network=road_network,
            congestion_file=congestion_file,
            graph_type=context.config("dmc_graph_type")
        )
    
    # get and build the router
    router_class = context.stage("mode_choice.network.car_router")
    router = router_class(network_processor = network_processor)
    router.build()

    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")

    # route the trips
    routed_trips = router.router_trips_dataframe(trips, 
                                                 congestion=True, 
                                                 batch_size=context.config("routing_batch_size"))

    routed_trips["travel_time_min"] = routed_trips["total_travel_time"] / 60

    # access_egress time in min
    access_egress_distance = routed_trips["access_euc_distance"] + routed_trips["egress_euc_distance"]
    walk_speed = context.config("walk_speed_m_per_s")
    walk_factor = context.config("walk_distance_factor")
    routed_trips["access_egress_time_min"] = (access_egress_distance * walk_factor) / (walk_speed * 60)

    ######################
    # finalize the output
    df = trips[["person_id","trip_index","crowfly_distance"]].copy()
    
    # Euclidean distance in km
    df["Euclidean_distance_km"] = df["crowfly_distance"] * 1e-3

    # router distance in km (this should be corrected once we make sure pandana is corrected)
    df["distance_km"] = df["Euclidean_distance_km"] * context.config("car_distance_factor")

    # merge with travel times
    df = df.merge(
        routed_trips[["person_id","trip_index","travel_time_min","access_egress_time_min"]],
        on=["person_id","trip_index"],
        how="left"
    )

    return df[["person_id","trip_index",
                "travel_time_min","access_egress_time_min",
                "distance_km"]]
    