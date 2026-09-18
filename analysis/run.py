import logging
logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.mode_shares.run")
    context.stage("analysis.counts.run")
    context.stage("analysis.counts.cross_border_flow_cars")
    context.stage("analysis.travel_times.run")
    context.stage("analysis.pt.comparison_passenger_counts_geneva")


def execute(context):
    runs = {
        "mode_shares" : context.stage("analysis.mode_shares.run")["path"],
        "counts" : context.stage("analysis.counts.run")["path"],
        "cross_border_flow_cars" : context.stage("analysis.counts.cross_border_flow_cars")["path"],
        "travel_times" : context.stage("analysis.travel_times.run")["path"],
        "pt_comparison_geneva" : context.stage("analysis.pt.comparison_passenger_counts_geneva")["path"],
    }

    for stage, out in runs.items():
        logger.info(f"Analysis stage '{stage}' completed, output is saved to: {out}")
    
    return runs
    