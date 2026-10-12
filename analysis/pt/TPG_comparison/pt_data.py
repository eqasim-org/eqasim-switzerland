from functools import cached_property

import gtfs_utils
import tpg_data


class PtData:
    """Everything the PT comparison stages read from the GTFS and the MATSim output, loaded lazily and
    only once: the GTFS (with the stops of the simulated schedule), the MATSim passenger counts and
    the routes of the transit schedule are shared by all the stages."""

    def __init__(self, cfg):
        self.cfg = cfg

    @cached_property
    def schedule_path(self):
        return gtfs_utils.find_schedule_file(self.cfg.matsim_output_folder)

    def require_schedule_path(self):
        if self.schedule_path is None:
            raise FileNotFoundError(
                f"output_transitSchedule.xml.gz not found in {self.cfg.matsim_output_folder}: it is needed "
                "to match the stops and to get the stop order of the routes"
            )

        return self.schedule_path

    @cached_property
    def _stops_and_alias(self):
        gtfs_stops = gtfs_utils.add_missing_base_stops(gtfs_utils.read_gtfs(self.cfg.gtfs_zip))
        stop_alias = {}

        if self.cfg.use_schedule_stops:
            if self.schedule_path is None:
                print("No output_transitSchedule.xml.gz in the MATSim output: only the GTFS stops are used")
            else:
                gtfs_stops, stop_alias = gtfs_utils.supplement_with_schedule_stops(gtfs_stops, self.schedule_path)

        return gtfs_stops, stop_alias

    @property
    def gtfs_stops(self):
        """GTFS stops (EPSG:2056) completed with the stops of the simulated schedule."""
        return self._stops_and_alias[0]

    @property
    def stop_alias(self):
        """Schedule stop id -> GTFS stop id."""
        return self._stops_and_alias[1]

    @cached_property
    def stops_in_perimeter(self):
        stops_wgs = self.gtfs_stops.to_crs("EPSG:4326")
        return gtfs_utils.filter_stops_in_shapefile(stops_wgs, self.cfg.perimeter_shapefile)["stop_id"].values

    @cached_property
    def matsim_counts(self):
        """MATSim passenger counts of all the stops, with the GTFS stop id in stop_id_gtfs. Not scaled: do
        not modify, use counts_in or copy."""
        return tpg_data.read_matsim_counts(self.cfg.matsim_output_folder, self.stop_alias, self.cfg.night_hours, self.cfg.counts_source)

    def counts_in(self, stop_ids):
        """Copy of the MATSim counts of the given GTFS stop ids."""
        return self.matsim_counts[self.matsim_counts["stop_id_gtfs"].isin(stop_ids)].copy()

    @cached_property
    def route_stops(self):
        """{(transit line id, transit route id): stop facility ids in the order of the route}"""
        return gtfs_utils.read_schedule_route_stops(self.require_schedule_path())

    @cached_property
    def tpg_stops(self):
        return tpg_data.load_tpg_stops(self.cfg.tpg_data_path, self.gtfs_stops)

    @cached_property
    def stop_coordinates(self):
        """{GTFS stop code without platform suffix: (lat, lon)}"""
        stops  = self.gtfs_stops.drop_duplicates("stop_id")
        coords = stops.assign(code = stops["stop_id"].astype(str).str.split(":").str[0])
        coords = coords.groupby("code")[["stop_lat", "stop_lon"]].mean()

        return {code: (row.stop_lat, row.stop_lon) for code, row in coords.iterrows()}
