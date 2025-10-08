import os



def get_calibration_args(context):
    additional_args = []
    if context.config("calibrate_betas_in_matsim"):
        cache_path = context.working_directory
        bounds = """car.alpha_u:5.0,
                    walk.alpha_u:5.0,
                    bike.alpha_u:5.0,
                    cp.alpha_u:5.0,
                    cp.travelTimeExponent:0.3,
                    walk.travelTimeExponent:0.3,
                    car.betaTravelTime_u_min:0.3"""
                
        additional_args.extend(["--config:eqasim:calibration.activate", "true",
                                "--config:eqasim:calibration.runCalibration", "true",
                                "--config:eqasim:calibration.optimizer", "cmaes",
                                "--config:eqasim:calibration.metric", "mse",
                                "--config:eqasim:calibration.betaMomentum", "0.4",
                                "--config:eqasim:calibration.eqasimCachePath", cache_path,
                                "--config:eqasim:calibration.bounds", bounds,
                                "--config:eqasim:calibration.distanceBins", "0,451,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000",
                                "--config:eqasim:calibration.maxEval", "4000",
                                "--config:eqasim:calibration.repoCommit", "b72fae6c3860169f5a837d4648dbf4b2dc7ac3a0",
                                "--config:eqasim:calibration.optimizerPath", "optimizer"
                                ])
    else:
        if context.config("calibrate_alphas_in_matsim"):
            level = context.config("alphaCalibration.level")
            filePath = context.config("alphaCalibration.filePath")

            additional_args.extend([
                "--config:eqasim:alphaCalibration.activate", "true",
                "--config:eqasim:alphaCalibration.beta", "0.2",
                "--config:eqasim:alphaCalibration.calibratedModes", "car,pt,walk,bike,car_passenger",
                "--config:eqasim:alphaCalibration.carModeShare", "0.423",
                "--config:eqasim:alphaCalibration.ptModeShare", "0.146",
                "--config:eqasim:alphaCalibration.walkModeShare", "0.256",
                "--config:eqasim:alphaCalibration.bikeModeShare", "0.083",
                "--config:eqasim:alphaCalibration.carPassengerModeShare", "0.092",
                "--config:eqasim:alphaCalibration.level", level,
                "--config:eqasim:alphaCalibration.filePath", filePath

            ])

    return additional_args







def get_delays_args(context):
    activate_tl_delays = context.config("activate_traffic_light_delays")
    activate_unsignalized_delaus = context.config("activate_unsignalized_intersections_delays")
    additional_args = []
    if activate_tl_delays or activate_unsignalized_delaus:
        additional_args.extend([
            "--config:eqasim:intersectionDelays.activate", "true",
            "--config:eqasim:intersectionDelays.activateTl", str(activate_tl_delays).lower(),
            "--config:eqasim:intersectionDelays.activateUnsignalized", str(activate_unsignalized_delaus).lower()
        ])
    return additional_args  