def configure(context):
    context.stage("sumo.config.microscopic")
    context.stage("sumo.config.mesoscopic")
    context.stage("sumo.config.macroscopic")
    context.stage("sumo.config.loaded")

def execute(context):
    return dict(
        microscopic = context.stage("sumo.config.microscopic"),
        mesoscopic  = context.stage("sumo.config.mesoscopic"),
        macroscopic = context.stage("sumo.config.macroscopic"),
        loaded      = context.stage("sumo.config.loaded"),
    )
