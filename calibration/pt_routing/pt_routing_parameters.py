
def configure(context):
    context.config("calibrate_pt_routing_params", default = False)

    if context.config("calibrate_pt_routing_params"):
        context.stage("calibration.pt_routing.calibrate_parameters")


def execute(context):
    if not context.config("calibrate_pt_routing_params"):
        params = {"travelTimeRail_u_h": -1.0, 
                  "travelTimeBus_u_h": -1.5836037239192158, 
                  "travelTimeTram_u_h": -1.1222785629604182, 
                  "travelTimeSubway_u_h": -1.1222785629604182, 
                  "travelTimeOther_u_h": -1.5836037239192158, 
                  "waitTime_u_h": -1.72, 
                  "walkTime_u_h": -1.97, 
                  "perTransfer_u": 0.0,
                  "raptorPenalties:transfer_intermodal": 0.21802854138692665,
                  "raptorPenalties:transfer_rail": 0.10055599178047361, 
                  "raptorPenalties:transfer_other": 0.1772974263245458}
        
    else:
        params = context.stage("calibration.pt_routing.calibrate_parameters")

    print(params)
        
    return params