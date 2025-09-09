import itertools
import pandas as pd
import networkx as nx

def configure(context):
    context.stage("data.pt_pricing.t603.prepare_t603")
    context.config("data_path")
    context.config("gtfs_name")


def create_network(triangles):
    G = nx.Graph()

    for _, row in triangles.iterrows():
        origin = row["origin_name"]
        destination = row["destination_name"]
        distance = row["distance"]

        if origin != destination and not G.has_edge(origin, destination):
            if pd.notnull(distance) and isinstance(distance, (int, float)):
                G.add_edge(origin, destination, weight=distance)

    return G


def best_path_by_hops_then_weight(G, source, target, weight="weight"):
    all_shortest_paths = list(nx.all_shortest_paths(G, source=source, target=target))

    def compute_path_weight(path):
        return sum(G[u][v][weight] for u, v in zip(path, path[1:]))

    best_path = min(all_shortest_paths, key=compute_path_weight)
    best_weight = compute_path_weight(best_path)

    return best_path, best_weight


def execute(context):
    distances = context.stage("data.pt_pricing.t603.prepare_t603")
    G         = create_network(distances)
    data_path = context.config("data_path")
    gtfs_name = context.config("gtfs_name")
    gtfs_stops_path = f"{data_path}/gtfs/{gtfs_name}/stops.txt"

    all_pair_paths = {}

    cpt = 0
    N   = len(list(itertools.combinations(G.nodes, 2)))

    for origin, destination in itertools.combinations(G.nodes, 2):
        if cpt % 1000 == 0:
            progress = cpt / N * 100
            print(f"{progress} % already done")
        if origin != destination:
            try:
                _, dist = best_path_by_hops_then_weight(G, origin, destination)
                all_pair_paths[(origin, destination)] = {"distance": dist}
            except nx.NetworkXNoPath:
                print(origin, destination)
                all_pair_paths[(origin, destination)] = {"distance": None}
        cpt += 1

    df = pd.DataFrame([
        {"origin_name": o, "destination_name": d, "distance": info["distance"]}
        for (o, d), info in all_pair_paths.items()
    ])

    gtfs_stops = pd.read_csv(gtfs_stops_path)[["stop_id", "stop_name", "location_type"]]
    gtfs_stops = gtfs_stops[gtfs_stops["location_type"].notna()]
    gtfs_stops["stop_id"] = gtfs_stops["stop_id"].str.split("Parent").str[-1]
    gtfs_stops["stop_id"] = gtfs_stops["stop_id"].astype(int)

    df = df.merge(gtfs_stops.copy().rename(columns={"stop_id" : "origin_id"}), how = "left", left_on = "origin_name", right_on = "stop_name")[
        ["origin_id", "origin_name", "destination_name", "distance"]
        ]

    df = df.merge(gtfs_stops.copy().rename(columns={"stop_id" : "destination_id"}), how = "left", left_on = "destination_name", right_on = "stop_name")[
        ["origin_id", "destination_id", "origin_name", "destination_name", "distance"]
        ]

    print(df.head())
    df = df[(df["origin_id"].notna()) & (df["destination_id"].notna())]

    df["origin_id"]      = df["origin_id"].astype(int)
    df["destination_id"] = df["destination_id"].astype(int)

    return df
