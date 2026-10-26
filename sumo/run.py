"""Select microscopic or mesoscopic execution of the cut SUMO scenario."""


def configure(context):
    from sumo.scenario.modes import require_mode_choice
    require_mode_choice(context)
    model = context.config("sumo_simulation_model", "microscopic")
    if model not in {"microscopic", "mesoscopic"}:
        raise ValueError("sumo_simulation_model must be 'microscopic' or 'mesoscopic'")
    context.stage("sumo.simulation.run_" + model)


def execute(context):

    # Return the selected regional simulation's cached output paths.
    return context.stage("sumo.simulation.run_" + context.config("sumo_simulation_model"))
