"""Shared scenario mode selection; truck includes freight and light commercials."""

SUPPORTED = {"car", "truck", "lcv", "taxi", "bike", "walk", "pt", "car_passenger", "motorcycle", "moped"}
CLASSES = {"car": "passenger", "truck": "truck", "lcv": "delivery", "taxi": "taxi",
           "bike": "bicycle", "walk": "pedestrian", "pt": "bus", "motorcycle": "motorcycle", "moped": "moped"}


def configure(context):
    context.config("sumo_modes", ["car", "truck"])
    selected(context.config("sumo_modes"))


def selected(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value or any(mode not in SUPPORTED for mode in value):
        raise ValueError(f"sumo_modes must be a nonempty list drawn from {sorted(SUPPORTED)}; None retains all modes")
    result = set(value)
    if "truck" in result:
        result.add("lcv")
    return result


def includes(mode, modes):
    if modes is None:
        return True
    return mode.removesuffix("_loop") in modes


def vehicle_mode(vehicle_class):
    if vehicle_class in {"truck", "trailer"}:
        return "truck"
    return {"passenger": "car", "delivery": "lcv", "bicycle": "bike", "bus": "pt", "coach": "pt"}.get(vehicle_class, vehicle_class)


def vehicle_classes(modes):
    if modes is None:
        return {"passenger", "truck", "delivery", "taxi", "bicycle", "motorcycle", "moped"}
    return {CLASSES[mode] for mode in modes if mode in CLASSES}


def plan_segments(activities, modes):
    """Keep contiguous selected legs without connecting across omitted movements."""
    if modes is None:
        return [activities]
    segments, start = [], None
    for index, activity in enumerate(activities[:-1]):
        keep = isinstance(activity.get("following_mode"), str) and includes(activity["following_mode"], modes)
        if keep and start is None:
            start = index
        if start is not None and (not keep or index == len(activities) - 2):
            stop = index + 2 if keep else index + 1
            segments.append(activities[start:stop])
            start = None
    return segments


def require_mode_choice(context):
    context.config("enable_mode_choice", False)
    if not context.config("enable_mode_choice"):
        raise ValueError("SUMO requires enable_mode_choice: true so selected modes and reusable road routes are available")
