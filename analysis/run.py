import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.mode_shares.run")
    context.stage("analysis.counts.run")
    context.stage("analysis.travel_times.run")

def execute(context):
    runs = {
        "mode_shares" : context.stage("analysis.mode_shares.run")["path"],
        "counts" : context.stage("analysis.counts.run")["path"],
        "travel_times" : context.stage("analysis.travel_times.run")["path"],
    }

    for stage, out in runs.items():
        logger.info(f"Analysis stage '{stage}' completed, output is saved to: {out}")
    
    return runs
    