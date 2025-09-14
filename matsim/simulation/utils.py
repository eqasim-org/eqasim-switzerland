import os



def get_alpha_calibration_args(context):
    additional_args = []
    if context.config("calibrate_betas_in_matsim"):
        cache_path = os.path.dirname(os.path.dirname(context.path()))  
        bounds = """car.alpha_u:4.0,
                    walk.alpha_u:3.0,
                    bike.alpha_u:3.0,
                    cp.alpha_u:3.0,
                    cp.travelTimeExponent:0.2,
                    pt.inVehicleTimeExponent:0.5,
                    walk.travelTimeExponent:0.2,
                    car.travelTimeExponent:0.2,
                    bike.travelTimeExponent:0.2,
                    car.betaTravelTime_u_min:0.5,
                    walk.betaTravelTime_u_min:0.5,
                    bike.betaTravelTime_u_min:0.5,
                    pt.betaInVehicleTime_u_min:0.5,
                    cp.betaTravelTime_u_min:0.5,
                    pt.betaWaitingTime_u_min: 0.2,
                    pt.betaShortDistance_u:0.5"""
                
        additional_args.extend(["--config:eqasim:calibration.activate", "true",
                                "--config:eqasim:calibration.runCalibration", "true",
                                "--config:eqasim:calibration.optimizer", "Nelder-Mead",
                                "--config:eqasim:calibration.metric", "kl",
                                "--config:eqasim:calibration.betaMomentum", "0.5",
                                "--config:eqasim:calibration.eqasimCachePath", cache_path,
                                "--config:eqasim:calibration.bounds", bounds,
                                "--config:eqasim:calibration.distanceBins", "0,275,451,683,995,1513,2400,3853,5026,6674,9261,13788,22976,1000000",
                                "--config:eqasim:calibration.maxEval", "1000",
                                "--config:eqasim:calibration.repoCommit", "1de98c60cf0ca5c3f48d2342ea776f54cad05ffd",
                                ])
    else:
        if context.config("calibrate_alphas_in_matsim"):
            additional_args.extend([
                "--config:eqasim:alphaCalibration.activate", "true",
                "--config:eqasim:alphaCalibration.beta", "0.2",
                "--config:eqasim:alphaCalibration.calibratedModes", "car,pt,walk,bike,car_passenger",
                "--config:eqasim:alphaCalibration.carModeShare", "0.423",
                "--config:eqasim:alphaCalibration.ptModeShare", "0.146",
                "--config:eqasim:alphaCalibration.walkModeShare", "0.256",
                "--config:eqasim:alphaCalibration.bikeModeShare", "0.083",
                "--config:eqasim:alphaCalibration.carPassengerModeShare", "0.092"
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