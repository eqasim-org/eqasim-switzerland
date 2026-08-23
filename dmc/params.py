import yaml
import os

def configure(context):
    context.config("estimate_dmc", default=False)
    context.config("dmc_estimator", default="jax")
    context.stage("calibration.dmc.default")

    if context.config("estimate_dmc"):
        estimator = str(context.config("dmc_estimator")).strip().lower()
        if estimator not in ("biogeme", "jax"):
            raise ValueError("dmc_estimator must be either 'biogeme' or 'jax'")
        context.stage("dmc.model_jax" if estimator == "jax" else "dmc.model", alias="dmc_model")


def execute(context):
    if context.config("estimate_dmc"):
        _, _, (mode_params_path, cost_params_path), _, _ = context.stage("dmc_model")
        mode_params_path = calibrate_default(context, mode_params_path)
    else:
        mode_params_path, cost_params_path = context.stage("calibration.dmc.default")

    return mode_params_path, cost_params_path











########## functions ############
def calibrate_default(context, path_to_params):
    calibrated_params, _ = context.stage("calibration.dmc.default")

    # read both yml files
    with open(path_to_params, 'r') as f:
        model_params = yaml.safe_load(f)

    with open(calibrated_params, 'r') as f:
        cal_params = yaml.safe_load(f)

    # use the calibrated ASCs and cantons ASCs
    model_params["car.alpha_u"] = cal_params["car.alpha_u"]
    model_params["bike.alpha_u"] = cal_params["bike.alpha_u"]
    model_params["walk.alpha_u"] = cal_params["walk.alpha_u"]
    model_params["cp.alpha_u"] = cal_params["cp.alpha_u"]

    model_params["swissCanton.car"] = cal_params["swissCanton.car"]
    model_params["swissCanton.pt"] = cal_params["swissCanton.pt"]
    model_params["swissCanton.bike"] = cal_params["swissCanton.bike"]
    model_params["swissCanton.walk"] = cal_params["swissCanton.walk"]
    model_params["swissCanton.cp"] = cal_params["swissCanton.cp"]

    # save the yaml file
    estimator = context.config("dmc_estimator")
    os.rename(path_to_params, path_to_params.replace(".yml", f"_{estimator}.yml"))
    with open(path_to_params, 'w') as f:
        yaml.safe_dump(model_params, f, sort_keys=False)

    return path_to_params