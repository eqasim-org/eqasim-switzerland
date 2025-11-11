import pandas as pd
import numpy as np
import os

def config(context):
    context.stage("mode_choice.prepare_trips")
    context.stage("mode_choice.car.router")
    context.stage("mode_choice.network.network_processor")
    
    context.config("data_path")

    context.config("dmc_network_file", default=os.path.join(context.config("data_path"), "dmc", "switzerland_network.xml.gz"))
    context.config("dmc_congestion_file", default=os.path.join(context.config("data_path"), "dmc", "linkstats.txt.gz"))    
    context.config("dmc_graph_type", default="pandana")
    context.config("routing_batch_size", default=2048)


def excute(context):
    # get network and congestion file paths
    network_file = context.config("dmc_network_file")
    congestion_file = context.config("dmc_congestion_file")
    
    # prepare the router
    router_class = context.stage("mode_choice.car.router")
    network_processor_class = context.stage("mode_choice.network.network_processor")    
    network_processor = network_processor_class(
            network_file=network_file,
            congestion_file=congestion_file,
            graph_type=context.config("dmc_graph_type")
        )
    router = router_class(network_processor = network_processor)

    # read prepared trips
    trips = context.stage("mode_choice.prepare_trips")

    # route the trips
    routed_trips = router.router_trips_dataframe(trips, 
                                                 congestion=True, 
                                                 batch_size=context.config("routing_batch_size"))
    return routed_trips
    
    
    

    