"""Shared boundary indexing and geographic endpoint resolution for routed cutting."""

import numpy as np
import shapely
from pyproj import Transformer

from sumo.cutter.tools import inside_intervals, read_region
from sumo.scenario.network.tools import position, read_network


class BoundaryIndex:
    """Index polygon boundaries in metres for repeated road-shape intersections.

    The boundary-segment index avoids intersecting every crossing trip with the
    whole detailed national polygon. Prepared predicates handle unchanged trips.
    """

    def __init__(self, border):
        self.region = read_region(border)
        shapely.prepare(self.region)
        self.forward = Transformer.from_crs(4326, border["crs"], always_xy=True)
        self.reverse = Transformer.from_crs(border["crs"], 4326, always_xy=True)
        rings = shapely.get_rings(shapely.get_parts(self.region))
        segments = []
        for ring in rings:
            xy = shapely.get_coordinates(ring)
            segments.append(shapely.linestrings(np.stack([xy[:-1], xy[1:]], axis=1)))
        self.boundary = shapely.STRtree(np.concatenate(segments))

    def intervals(self, line):
        if shapely.covers(self.region, line):
            return [(0.0, 1.0)]
        if not shapely.intersects(self.region, line):
            return []
        if len(line.coords) != 2 or not line.is_simple:
            return inside_intervals(line, self.region)
        hits = self.boundary.query(line, predicate="intersects")
        intersections = shapely.intersection(line, self.boundary.geometries[hits])
        xy = shapely.get_coordinates(intersections)
        fractions = shapely.line_locate_point(line, shapely.points(xy), normalized=True)
        cuts = np.unique(np.clip(np.concatenate([[0., 1.], fractions]), 0, 1))
        mids = shapely.line_interpolate_point(line, (cuts[:-1] + cuts[1:]) / 2, normalized=True)
        inside = shapely.covers(self.region, mids)
        result = []
        for start, end in zip(cuts[:-1][inside], cuts[1:][inside]):
            if end - start < 1e-10:
                continue
            if result and abs(result[-1][1] - start) < 1e-10:
                result[-1] = (result[-1][0], float(end))
            else:
                result.append((float(start), float(end)))
        return result



class EndpointResolver:
    """Resolve SUMO XML endpoints without changing their producing network."""

    def __init__(self, network):
        self.network_source, self.network = network, None

    def xml_coordinates(self, attributes):
        """Read geographic inputs, or resolve explicit edge/XY endpoints lazily."""
        if "fromLonLat" in attributes and "toLonLat" in attributes:
            return attributes["fromLonLat"], attributes["toLonLat"]
        if self.network is None:
            if self.network_source is None:
                raise ValueError("Border clipping of edge/XY trips requires the SUMO network")
            self.network = read_network(self.network_source())
        location = self.network["location"]
        projection = location.get("projParameter")
        if projection in {None, "!", "-"}:
            raise ValueError("Border clipping of edge/XY trips needs a georeferenced network; "
                             "use geographic trips or set clip_trips_to_border_in_sumo: false")
        reverse = Transformer.from_crs(projection, 4326, always_xy=True)
        offset = np.asarray([float(x) for x in location.get("netOffset", "0,0").split(",")])
        result = []
        for key, pos in (("from", "departPos"), ("to", "arrivalPos")):
            if key + "LonLat" in attributes:
                result.append(attributes[key + "LonLat"])
                continue
            if key + "XY" in attributes:
                xy = np.asarray([float(x) for x in attributes[key + "XY"].split(",")])
            elif key in attributes:
                edge = self.network["edges"][attributes[key]]
                value = attributes.get(pos, "0" if key == "from" else str(edge["length"]))
                try:
                    distance = position(value, edge["length"])
                except ValueError as error:
                    raise ValueError("Border clipping requires deterministic departPos/arrivalPos") from error
                xy = np.asarray(edge["shape"].interpolate(distance / edge["length"], normalized=True).coords[0][:2])
            else:
                raise ValueError("Border clipping requires from/to, fromXY/toXY or fromLonLat/toLonLat")
            lon, lat = reverse.transform(*(xy - offset), errcheck=True)
            result.append(f"{lon:.10f},{lat:.10f}")
        return tuple(result)

