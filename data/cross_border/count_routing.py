"""Fast Swiss-only free-flow costs for count-guided crossing assignment.

These directed shortest paths approximate geography; final MATSim routing still
respects turns and validates routes. Parallel links use the fastest road.
"""

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from matsim.readers import read_network


class SwissRoadCosts:
    def __init__(self, context):
        network = read_network(context.stage("matsim.scenario.network.pt")["network"],
                               skip_attributes=True, only_car_links=True, context=context)
        self.nodes = nodes = network.nodes.reset_index(drop=True)
        self.index = index = pd.Series(nodes.index, index=nodes.node_id)
        self.links = network.links.set_index("link_id")
        border = context.stage("data.spatial.swiss_border").geometry.union_all()
        inside = shapely.intersects_xy(border, nodes.x.to_numpy(), nodes.y.to_numpy())
        u = network.links.from_node.map(index).to_numpy()
        v = network.links.to_node.map(index).to_numpy()
        seconds = network.links.length.to_numpy(float) / network.links.freespeed.to_numpy(float)
        keep = inside[u] & inside[v] & np.isfinite(seconds) & (seconds > 0)
        edges = pd.DataFrame(dict(u=u[keep], v=v[keep], cost=seconds[keep]))
        edges = edges.sort_values("cost").drop_duplicates(["u", "v"])
        self.graph = csr_matrix((edges.cost, (edges.u, edges.v)), shape=(len(nodes), len(nodes)))
        self.reverse = self.graph.T.tocsr()
        _, components = connected_components(self.graph, directed=True, connection="strong")
        self.eligible = inside & (components == np.bincount(components).argmax())
        self.rows = np.flatnonzero(self.eligible)
        self.tree = cKDTree(nodes.loc[self.eligible, ["x", "y"]])

    def costs(self, rows, sites, exits):
        swiss = rows.category.eq("swiss").to_numpy()
        transit = rows.category.eq("transit").to_numpy()
        domestic = np.where(swiss[:, None], rows[["origin_x", "origin_y"]],
                            rows[["destination_x", "destination_y"]]).astype(float)
        known = transit & rows.reported_exit_point.isin(exits.index).to_numpy()
        for i in np.flatnonzero(known):
            domestic[i] = exits.loc[rows.iloc[i].reported_exit_point].coords[0]
        finite = np.isfinite(domestic).all(axis=1)
        offset, end = self.tree.query(np.where(finite[:, None], domestic, self.tree.data[0]))
        end = self.rows[end]
        supported = finite & (offset <= 5000) & (~transit | known)
        inward, outward, anchors = [], [], []
        for site in sites.itertuples():
            for direction, link, node_column, graph, output in (
                ("entry", site.entry_link, "to_node", self.graph, inward),
                ("exit", site.exit_link, "from_node", self.reverse, outward),
            ):
                node = int(self.index[self.links.loc[link, node_column]])
                distance = 0.
                if not self.eligible[node]:
                    distance, replacement = self.tree.query(self.nodes.loc[node, ["x", "y"]].to_numpy(float))
                    node = int(self.rows[replacement])
                if distance > 1000:
                    raise ValueError("Crossing %s has no connected Swiss road within 1 km" % site.fid)
                anchors.append(dict(fid=int(site.fid), direction=direction, offset_m=float(distance)))
                output.append(dijkstra(graph, indices=node, directed=True)[end] / 60.)
        inward, outward = np.asarray(inward).T, np.asarray(outward).T
        road = (inward + outward) / 2.
        entry = transit | (swiss & rows.direction.eq("entry").to_numpy())
        exit = swiss & rows.direction.eq("exit").to_numpy()
        road[entry], road[exit] = inward[entry], outward[exit]
        # The reported exit is not a valid entry alternative for transit. Exclude
        # it before finding the fastest entry, otherwise its near-zero route
        # would wrongly reject every genuine through-journey as a detour.
        candidate = ~(transit[:, None] & (rows.reported_exit_point.to_numpy()[:, None]
                                         == sites.point_id.to_numpy()))
        extra = road - np.where(candidate, road, np.inf).min(axis=1)[:, None]
        visitor = rows.category.eq("foreign_visitor").to_numpy()
        extra[visitor] = np.maximum(inward-inward.min(axis=1)[:, None],
                                    outward-outward.min(axis=1)[:, None])[visitor]
        foreign = np.where(swiss[:, None], rows[["destination_x", "destination_y"]],
                           rows[["origin_x", "origin_y"]]).astype(float)
        approach = np.linalg.norm(foreign[:, None] - sites[["x", "y"]].to_numpy(), axis=2) / 1000.
        # One minute per kilometre represents 60 km/h outside Switzerland.
        supported &= np.isfinite(road).all(axis=1) & np.isfinite(approach).all(axis=1)
        # Expose detours so count assignment can expand support without rerouting.
        # An impossible or prohibited choice stays infinite at every allowance.
        return road + approach, np.where(candidate, extra, np.inf), supported, road, anchors
