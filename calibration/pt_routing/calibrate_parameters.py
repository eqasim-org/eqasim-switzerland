import pickle
import numpy as np

def configure(context):
    context.stage("calibration.pt_routing.calibrate")


def is_mode_vot(key):
    return ("rail" in key or "subway" in key or "bus" in key or "tram" in key or key == "other_u_h") and not("transfer" in key)

def execute(context):
    file_path = context.stage("calibration.pt_routing.calibrate")

    utilities = {}
    with open(file_path, "rb") as f:
        parameters = pickle.load(f)
        objectives = [item["objective"] for item in parameters]
        best_index = np.argmin(objectives)
        utilities = parameters[best_index]["utilities"]

    new_utilities = {}
    for k, v in utilities.items():
        if is_mode_vot(k):
            new_k = "travelTime" + k[0].upper() + k[1:]
        elif k == "wait_u_h":
            new_k = "waitTime_u_h"
        elif k == "walk_u_h":
            new_k = "walkTime_u_h"
        elif k == "transfer_u":
            new_k = "perTransfer_u"
        else:
            new_k = k
        new_utilities[new_k] = v

    return new_utilities