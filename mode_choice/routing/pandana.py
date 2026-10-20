"""Directed car routing using free-flow or departure-hour linkstats times."""
import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from mode_choice.routing.congestion import input_path, departure_hours, read_linkstats, hourly_links


class CarRouter:
    """Build one contraction hierarchy and reuse it across trip batches.

    Pandana requires pandas at its API boundary; network preparation and route
    outputs use Polars. One travel-time impedance ensures distance and time are
    measured along the same selected path, including parallel-link handling.
    """

    def __init__(self, nodes, links):
        import pandana

        links = links.filter(
            pl.col("modes").str.split(",").list.contains("car")
            & pl.col("length").is_finite() & (pl.col("length") > 0)
            & pl.col("freespeed").is_finite() & (pl.col("freespeed") > 0)
        )
        if "travel_time" not in links.columns:
            links = links.with_columns((pl.col("length") / pl.col("freespeed")).alias("travel_time"))
        if links["travel_time"].null_count() or not links["travel_time"].is_finite().all() or (links["travel_time"] <= 0).any():
            raise ValueError("Car link travel times must be finite and positive")
        if links.is_empty():
            raise ValueError("Pandana network contains no usable car links")
        used = pl.concat([links["from_node"], links["to_node"]]).unique()
        nodes = nodes.filter(pl.col("node_id").is_in(used.implode())).with_row_index("index")
        if nodes["node_id"].n_unique() != nodes.height:
            raise ValueError("Duplicate MATSim node IDs")
        coordinates = nodes.select("x", "y").to_numpy()
        if not np.isfinite(coordinates).all():
            raise ValueError("Non-finite network node coordinates")
        links = (links.join(nodes.select(pl.col("node_id").alias("from_node"), pl.col("index").alias("source")),
                            on="from_node", how="left", validate="m:1")
                 .join(nodes.select(pl.col("node_id").alias("to_node"), pl.col("index").alias("target")),
                       on="to_node", how="left", validate="m:1"))
        if links["source"].null_count() or links["target"].null_count():
            raise ValueError("Car link references a missing MATSim node")
        # Paths contain nodes, not link IDs. Pick the fastest parallel link
        # deterministically before graph construction; break ties by length/ID.
        links = (links.sort("travel_time", "length", "link_id")
                 .unique(["source", "target"], keep="first", maintain_order=True)
                 .sort("source", "target"))
        self.node_count = nodes.height
        self.edge_keys = (links["source"].cast(pl.Int64) * self.node_count + links["target"]).to_numpy()
        self.lengths = links["length"].to_numpy()
        self.times = links["travel_time"].to_numpy()
        self.tree = cKDTree(coordinates)
        self.graph = pandana.Network(
            nodes["x"].to_pandas(), nodes["y"].to_pandas(),
            links["source"].to_pandas(), links["target"].to_pandas(),
            links.select("travel_time").to_pandas(), twoway=False,
        )

    def route(self, trips, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("mode_choice_routing_batch_size must be a positive integer")
        outputs = []
        for batch in trips.iter_slices(batch_size):
            origins = batch.select("origin_x", "origin_y").to_numpy()
            destinations = batch.select("destination_x", "destination_y").to_numpy()
            if not np.isfinite(origins).all() or not np.isfinite(destinations).all():
                raise ValueError("Non-finite car trip coordinates")
            sources = self.tree.query(origins)[1]
            targets = self.tree.query(destinations)[1]
            # Repeated snapped OD pairs share a single routing query per batch.
            pairs, inverse = np.unique(np.column_stack([sources, targets]), axis=0, return_inverse=True)
            paths = self.graph.shortest_paths(pairs[:, 0], pairs[:, 1], imp_name="travel_time")
            valid = np.array([len(path) > 0 and path[0] == start and path[-1] == end
                              for path, (start, end) in zip(paths, pairs)])
            sizes = np.array([max(0, len(path) - 1) if ok else 0 for path, ok in zip(paths, valid)])
            times, distances = np.zeros(len(pairs)), np.zeros(len(pairs))
            if sizes.sum():
                keys = np.concatenate([path[:-1].astype(np.int64) * self.node_count + path[1:]
                                       for path, ok in zip(paths, valid) if ok and len(path) > 1])
                indices = np.searchsorted(self.edge_keys, keys)
                if (indices >= len(self.edge_keys)).any() or not np.array_equal(self.edge_keys[indices], keys):
                    raise ValueError("Pandana returned a path containing a nonexistent directed edge")
                owners = np.repeat(np.arange(len(pairs)), sizes)
                times = np.bincount(owners, weights=self.times[indices], minlength=len(pairs))
                distances = np.bincount(owners, weights=self.lengths[indices], minlength=len(pairs))
            times[~valid], distances[~valid] = np.nan, np.nan
            outputs.append(batch.select("identifier").with_columns(
                pl.Series("car_travel_time_min", times[inverse] / 60).fill_nan(None),
                pl.Series("car_distance_km", distances[inverse] / 1000).fill_nan(None),
            ))
        if not outputs:
            return trips.select("identifier").with_columns(
                pl.lit(None, dtype=pl.Float64).alias("car_travel_time_min"),
                pl.lit(None, dtype=pl.Float64).alias("car_distance_km"))
        return pl.concat(outputs)


def route(context, trips, batch_size):
    from matsim.readers import read_network

    network = read_network(context.stage("matsim.scenario.network.pt")["network"],
                           skip_attributes=True, only_car_links=True, context=context)
    nodes, links = pl.from_pandas(network.nodes), pl.from_pandas(network.links)
    del network
    path = input_path(context, "pandana")
    if path is None or trips.is_empty():
        return CarRouter(nodes, links).route(trips, batch_size)
    hours = departure_hours(trips)
    stats = read_linkstats(path, hours.unique().to_list())
    return route_hourly(nodes, links, trips, batch_size, stats, hours)


def route_hourly(nodes, links, trips, batch_size, stats, hours=None):
    """Route each departure-hour group once, holding only one hourly graph.

    The hour is fixed at departure for the whole path, as in the old branch;
    it is not advanced as the vehicle traverses links. Daily hours wrap at 24.
    """
    if hours is None:
        hours = departure_hours(trips)
    outputs = []
    requests = trips.select("identifier", "origin_x", "origin_y", "destination_x", "destination_y").with_columns(hours)
    for (hour,), group in requests.partition_by("departure_hour", as_dict=True).items():
        router = CarRouter(nodes, hourly_links(links, stats, hour))
        outputs.append(router.route(group, batch_size))
        del router  # Release this hierarchy before constructing the next hour.
    routes = pl.concat(outputs)
    return trips.select("identifier").join(routes, on="identifier", how="left", validate="1:1", maintain_order="left")
