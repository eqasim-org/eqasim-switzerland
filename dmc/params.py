def configure(context):
    context.config("estimate_dmc", default=False)
    context.config("dmc_estimator", default="jax")

    if context.config("estimate_dmc"):
        estimator = str(context.config("dmc_estimator")).strip().lower()
        if estimator not in ("biogeme", "jax"):
            raise ValueError("dmc_estimator must be either 'biogeme' or 'jax'")
        context.stage("dmc.model_jax" if estimator == "jax" else "dmc.model")

    else:
        context.stage("calibration.dmc.default")


def execute(context):
    if context.config("estimate_dmc"):
        estimator = context.config("dmc_estimator")
        stage = "dmc.model_jax" if estimator == "jax" else "dmc.model"
        _, _, (mode_params_path, cost_params_path), _, _ = context.stage(stage)
    else:
        mode_params_path, cost_params_path = context.stage("calibration.dmc.default")

    return mode_params_path, cost_params_path
