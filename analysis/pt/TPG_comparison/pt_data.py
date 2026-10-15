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

    def counts_in(self, stop_ids, own_lines_only = False):
        """Copy of the MATSim counts of the given GTFS stop ids.

        The comparisons match MATSim and the observed counts by line name, and names are shared by the lines of different
        cities (13 and 17 are trams in Geneva and buses in Lausanne, N1 exists in both). own_lines_only drops, among the
        names that several transit lines (line_id) have, the lines with less than OWN_LINE_SHARE of their passengers at the
        given stops: they belong to another network and only touch these stops (e.g. long-distance trains)."""
        counts = self.matsim_counts[self.matsim_counts["stop_id_gtfs"].isin(stop_ids)].copy()

        if own_lines_only:
            ids_per_name = self.matsim_counts.groupby("line_name")["line_id"].nunique()
            shared = ids_per_name[ids_per_name > 1].index

            passengers = (self.matsim_counts["boardings"] + self.matsim_counts["alightings"]).groupby(self.matsim_counts["line_id"]).sum()
            inside     = (counts["boardings"] + counts["alightings"]).groupby(counts["line_id"]).sum()
            share      = (inside / passengers).reindex(passengers.index).fillna(0)

            foreign = counts["line_name"].isin(shared) & (counts["line_id"].map(share) < self.OWN_LINE_SHARE)
            if foreign.any():
                dropped = counts[foreign].groupby(["line_name", "line_id"])["boardings"].sum()
                print(f"{len(dropped)} transit line(s) sharing a name with another line, mostly outside this perimeter, left out: "
                      + ", ".join(f"{name} ({line_id}, {boardings:.0f} boardings)" for (name, line_id), boardings in dropped.items()))
                counts = counts[~foreign]

        self.warn_shared_line_names(counts)
        return counts

    OWN_LINE_SHARE = 0.5

    @staticmethod
    def warn_shared_line_names(counts):
        """The comparisons match MATSim and the observed counts by line name: warn when one name covers several
        transit lines (line_id) among the given counts, as they would be added together."""
        ids = counts.groupby("line_name")["line_id"].nunique()
        for name in ids[ids > 1].index:
            lines = counts[counts["line_name"] == name]
            detail = ", ".join(
                f"{line_id} ({group['stop_id_gtfs'].nunique()} stops, {group['boardings'].sum():.0f} boardings)"
                for line_id, group in lines.groupby("line_id")
            )
            print(f"WARNING line name {name} covers several transit lines, matched together: {detail}")

    @cached_property
    def line_modes(self):
        """{transit line id: transport mode of the transit schedule}; empty without a schedule"""
        if self.schedule_path is None:
            return {}

        return gtfs_utils.read_schedule_line_modes(self.schedule_path)

    @cached_property
    def route_stops(self):
        """{(transit line id, transit route id): stop facility ids in the order of the route}"""
        routes = gtfs_utils.read_schedule_route_stops(self.require_schedule_path())

        # The schedule of the output folder can be older than the simulation: the routes of the counts that it does
        # not have are read from the GTFS (cfg.route_gtfs_zip), where their route id is a trip id
        if self.cfg.route_gtfs_zip:
            counts = self.matsim_counts[["line_id", "route_id"]].drop_duplicates()
            missing = {route_id: line_id for line_id, route_id in zip(counts["line_id"], counts["route_id"])
                       if (line_id, route_id) not in routes}

            if missing:
                print(f"{len(missing)} routes of the counts are not in the transit schedule: reading their stops from {self.cfg.route_gtfs_zip}")
                found = gtfs_utils.read_gtfs_route_stops(self.cfg.route_gtfs_zip, missing)
                routes.update({(missing[route_id], route_id): stops for route_id, stops in found.items()})
                print(f"  -> {len(found)} found in the GTFS, {len(missing) - len(found)} still unknown")

        return routes

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
