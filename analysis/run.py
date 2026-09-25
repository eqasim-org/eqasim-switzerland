import logging
logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.mode_shares.run")
    context.stage("analysis.counts.run")
    context.stage("analysis.counts.cross_border_flow_cars")
    context.stage("analysis.counts.cross_border_flow_pt")
    context.stage("analysis.counts.pt_stop_agent_trips")
    #context.stage("analysis.travel_times.run")
    context.stage("analysis.pt_passenger_counts.TPG_comparison.comparison_passenger_counts_geneva")
    context.stage("analysis.pt_passenger_counts.transit_schedule.visualize")


def execute(context):
    runs = {
        "mode_shares" : context.stage("analysis.mode_shares.run")["path"],
        "counts" : context.stage("analysis.counts.run")["path"],
        "cross_border_flow_cars" : context.stage("analysis.counts.cross_border_flow_cars")["path"],
        "cross_border_flow_pt" : context.stage("analysis.counts.cross_border_flow_pt")["path"],
        "pt_stop_agent_trips" : context.stage("analysis.counts.pt_stop_agent_trips")["path"],
        #"travel_times" : context.stage("analysis.travel_times.run")["path"],
        "pt_comparison_geneva" : context.stage("analysis.pt_passenger_counts.TPG_comparison.comparison_passenger_counts_geneva")["path"],
        "transit_schedule_map" : context.stage("analysis.pt_passenger_counts.transit_schedule.visualize")["path"],
    }

    for stage, out in runs.items():
        logger.info(f"Analysis stage '{stage}' completed, output is saved to: {out}")
    
    return runs
    