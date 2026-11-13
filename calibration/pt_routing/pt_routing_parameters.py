
def configure(context):
    context.config("calibrate_pt_routing_params", default = False)

    if context.config("calibrate_pt_routing_params"):
        context.stage("calibration.pt_routing.calibrate_parameters")


def execute(context):

    # Use the fixed routing preferences unless parameter calibration is requested.

    if not context.config("calibrate_pt_routing_params"):
        params = {"travelTimeRail_u_h": -1.0,
                  "travelTimeBus_u_h": -1.3504,
                  "travelTimeTram_u_h": -1.0037,
                  "travelTimeSubway_u_h": -1.0037,
                  "travelTimeOther_u_h": -1.3504,
                  "waitTime_u_h": -1.29,
                  "walkTime_u_h": -1.35,
                  "perTransfer_u": 0.0,
                  "raptorPenalties:transfer_intermodal": 0.1520,
                  "raptorPenalties:transfer_rail": 0.1419,
                  "raptorPenalties:transfer_other": 0.1212}


    else:
        params = context.stage("calibration.pt_routing.calibrate_parameters")

    # Return the same parameter schema for routing and simulation config consumers.

    return params
