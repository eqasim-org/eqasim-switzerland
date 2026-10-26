"""Paths to files copied into a cut scenario; no config parsing or copying."""
from pathlib import Path


def simulation_root(context):
    source = context.config("cutter.scenario_path")
    return str(Path(source).expanduser().resolve()) if source else context.path("matsim.simulation.run")


class FilePaths:
    def __init__(self, context):
        self.context = context
        source = context.config("cutter.scenario_path")
        self.root = Path(source).expanduser().resolve() if source else None

    def get(self, name):
        """Return a supporting file path, from the supplied folder or a stage."""
        if self.root is not None:
            patterns = {
                "global_shares": ("global_target_mode_shares.csv", "*global_mode_shares.csv"),
                "cantonal_shares": ("cantonal_target_mode_shares.csv", "*cantonal_mode_shares.csv"),
                "geometry": ("*detailed_network.csv", "*network_geometry.csv"),
                "counts": ("network_calibration_files/target_counts.csv",),
                "travel_times": ("network_calibration_files/target_travel_times.csv",),
            }.get(name, (name,))
            for pattern in patterns:
                matches = sorted(p for p in self.root.glob(pattern) if p.is_file())
                if len(matches) == 1:
                    return str(matches[0])
                if len(matches) > 1:
                    raise ValueError(f"Multiple files match {pattern} in {self.root}: {matches}")
            if name == "global_shares":
                return ""
            raise FileNotFoundError(f"Cannot find {name} in cutter.scenario_path={self.root}; expected {patterns}")

        if name in ("dmc_parameters.yml", "cost_parameters.yml"):
            return str(Path(self.context.path("matsim.simulation.prepare")) / name)
        stages = {
            "global_shares": ("data.microcensus.shares", 0),
            "cantonal_shares": ("data.microcensus.shares", 1),
            "SBB_all_distances.csv": ("data.pt_pricing.pt_pricing", 0),
            "gtfs_zones.csv": ("data.pt_pricing.pt_pricing", 1),
            "pricingDescription.xml": ("data.pt_pricing.pt_pricing", 2),
            "geometry": ("matsim.scenario.network.pt", "geometry"),
            "counts": ("analysis.counts.target", None),
            "travel_times": ("analysis.travel_times.APIs.target", None),
        }
        stage, key = stages[name]
        result = self.context.stage(stage)
        return result if key is None else result[key]

    def regions(self, kind):
        """Return the optional special-region files as a semicolon-separated list."""
        if self.root is not None:
            prefix = {"penalty": "penalties", "freespeed": "freespeed"}[kind]
            return ";".join(str(p) for p in sorted(
                self.root.glob(f"network_calibration_files/{prefix}_special_region_*.yml")) if p.is_file())
        return self.context.stage(f"calibration.road_regions.{kind}_calibration")
