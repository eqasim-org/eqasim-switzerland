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
    return routed_trips
    
    
    

    