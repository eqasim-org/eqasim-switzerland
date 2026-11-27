"""Realize macroscopic route shares and loaded edge clocks in a separate cache."""

import json
import tempfile
from pathlib import Path

from sumo.functions.native import processing


def _native_routes(context, assigned):
    allocation = context.stage("sumo.macro_scenario.allocation")
    cache = Path(context.path()).resolve()
    paths = {
        name: str(cache / filename)
        for name, filename in [
            ("output", "routes.arrow"),
            ("edges", "edges.arrow"),
            ("metrics", "trip_metrics.arrow"),
            ("report", "realization.json"),
        ]
    }
    with tempfile.TemporaryDirectory(prefix="realize_", dir=cache) as temporary:
        temporary = Path(temporary)
        pending = {
            name: str(temporary / Path(path).name) for name, path in paths.items()
        }
        processing.run(
            context,
            "realize",
            selected=allocation["selected"],
            groups=allocation["groups"],
            types=allocation["types"],
            network=assigned["network"],
            weights=assigned["weights"],
            metadata_edges=paths["edges"],
            **pending,
        )
        report = json.loads(Path(pending["report"]).read_text())
        if report["vehicles"] != allocation["vehicles"]:
            raise ValueError("Native realization changed the allocated vehicle count")
        for name in ("output", "edges", "metrics"):
            Path(pending[name]).replace(paths[name])
    report.update(
        {
            "published": True,
            "metrics": paths["metrics"],
            "distributions": allocation["distributions"],
            "post_assignment_routing": False,
            "route_choice": "seeded balanced integer realization of assigned distributions",
            "timing": "loaded edge costs from each vehicle's departure interval",
        }
    )
    Path(paths["report"]).write_text(json.dumps(report, indent=2) + "\n")
    return {
        **assigned,
        "routes": paths["output"],
        "edges": paths["edges"],
        "types": allocation["types"],
        "route_format": "arrow",
        "convergence": {
            "method": assigned["method"],
            "equilibrium_guaranteed": False,
            "realization": report,
        },
        "timing": "macroscopic estimated edge exit times",
    }
