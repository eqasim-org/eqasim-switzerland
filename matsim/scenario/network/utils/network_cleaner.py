import warnings
import pandas as pd
import networkx as nx
import numpy as np

from matsim.scenario.network.utils.connectivity import (
    clean_restriction_references, largest_connected_links, restricted_link_ids,
)

import logging
logger = logging.getLogger(__name__)

class networkCleaner():
    def __init__(self, network):
        self.links = network.links.copy()
        self.nodes = network.nodes.copy()        
        self.network = network

    def run(self, remove_network_loops= True, remove_replicate_links= True,
                  remove_nodes_with_no_intersection= True, correct_speeds= True,
                  ensure_network_connectivity= True):
        stats = self.clean_network(remove_network_loops, remove_replicate_links, remove_nodes_with_no_intersection, 
                                   correct_speeds, ensure_network_connectivity)
        self.network.nodes = self.nodes
        self.network.links = self.links
        return self.network, stats

    def clean_network(self, remove_network_loops= True, remove_replicate_links= True,
                            remove_nodes_with_no_intersection= True, correct_speeds= True,
                            ensure_network_connectivity= True):
        stats = dict()

        """All network should be considered, because intersections could be with (pt) and (pt,car)"""
        df = self.links

        protected = restricted_link_ids(zip(df.link_id, df.attributes)) if (
            remove_network_loops or remove_replicate_links or remove_nodes_with_no_intersection
        ) else set()

        # Removing loops (restriction identities must remain valid).
        if remove_network_loops:
            sel = (df['from_node'] == df['to_node']) & ~df.link_id.isin(protected)
            stats["removed_loops"] = int(sel.sum())
            logger.info("There are %d loops in the network that are removed." % stats["removed_loops"] )
            df = df[~sel].reset_index(drop=True)

        # Remove replicated links
        if remove_replicate_links:            
            len_df = len(df)
            duplicates = df.duplicated(subset=['length', 'modes', 'from_node', 'to_node', 'capacity'])
            df = df.loc[~(duplicates & ~df.link_id.isin(protected))].reset_index(drop=True)
            stats["removed_link_duplicates"] = len_df-len(df)
            logger.info("There are %d link duplicates in the network that are removed." % stats["removed_link_duplicates"] )                
            
        # Removing nodes with no intersection
        if remove_nodes_with_no_intersection:
            logger.info("Removing nodes that do not represent an intersection.")
            df, stats2 = self.merge_link_chains(df, protected=protected)
            stats.update(stats2)
        
        # Removed links that are not connected to the whole graph
        if ensure_network_connectivity:
            logger.info("Remove unconnected links")
            func = self.remove_unconnected_links_with_turns_restrictions
            df, num_removed = func(df)        
            stats["removed_unconnected_links"] = num_removed
            logger.info("Remove truck access from disconnected truck components")
            df, num_removed = func(df, mode="truck", remove_mode_only=True)
            stats["removed_truck_mode_from_unconnected_links"] = num_removed
            logger.info("Removed truck access from %d disconnected links; links retained.", num_removed)
            df = clean_restriction_references(df)

        self.links = df        
        
        if correct_speeds:
            # Limit the speed limit in the network
            logger.info("Change the infinit freespeed to 85.")
            # This is also done in : https://github.com/eqasim-org/eqasim-java/blob/develop/core/src/main/java/org/eqasim/core/scenario/preparation/AdjustLinkLength.java#L9
            self.links.loc[self.links.freespeed.apply(np.isinf), "freespeed"] = 85
            self.links.loc[self.links.freespeed<20/3.6,"freespeed"] = 20/3.6
            self.links.loc[self.links.capacity<300,"capacity"] = 300

        # Update nodes dataframe to keep only the nodes present in the links
        unique_nodes = set(df.from_node.tolist() + df.to_node.tolist())
        self.nodes = self.nodes[self.nodes.node_id.isin(unique_nodes)].reset_index(drop=True)
            
        if "attributes" not in self.links:
            warnings.warn("Attributes are not present in the links dataframe.", category=UserWarning)
        
        return stats
        
    @staticmethod
    def _apply_connectivity_filter(df, selected, connected, mode, remove_mode_only):
        """Either remove disconnected links or only revoke the selected mode."""
        disconnected = selected & ~connected
        count = int(disconnected.sum())
        if remove_mode_only:
            df = df.copy()
            df.loc[disconnected, "modes"] = df.loc[disconnected, "modes"].map(
                lambda modes: ",".join(m for m in modes.split(",") if m != mode)
            )
            return df, count
        return df.loc[~disconnected].reset_index(drop=True), count

    def remove_unconnected_links(self, df, mode="car", remove_mode_only=False):
        """Use directed, mode-specific connectivity even without restrictions."""
        return self.remove_unconnected_links_with_turns_restrictions(df, mode, remove_mode_only)

    def remove_unconnected_links_with_turns_restrictions(self, df, mode="car", remove_mode_only=False):
        """Keep the largest strong component respecting complete restrictions."""
        selected = df["modes"].str.split(",").map(lambda modes: mode in modes)
        mode_links = df.loc[selected]
        if mode_links.empty:
            return df.copy(), 0
        logger.info("    Building sequence-aware connectivity graph for %s ...", mode)
        connected_ids = largest_connected_links(mode_links, mode)
        connected = df["link_id"].isin(connected_ids)
        logger.info("    %s: keeping %d of %d links in the largest component.",
                    mode, len(connected_ids), len(mode_links))
        return self._apply_connectivity_filter(df, selected, connected, mode, remove_mode_only)

    def merge_link_chains(self, df, protected=None):
        """Contract road chains between intersections or attribute boundaries.

        Two-way roads are checked in both directions. A node is retained if
        either direction changes speed, capacity or modes. Lanes are deliberately
        not a merge condition. Parallel links, loops and restriction identities
        are preserved rather than choosing an arbitrary path through them.
        """
        if protected is None:
            protected = restricted_link_ids(zip(df.link_id, df.attributes))
        stats = dict(number_of_nodes=0, number_of_links=len(df), attributes_change=0,
                     skiped_loop=0, successful_merge=0, degree_is_2=0,
                     one_in_one_out=0, ignored_no_car=0, break_no_car=0,
                     already_visited=0, ambiguous_nodes=0)
        if df.empty:
            return df.copy(), stats

        from collections import defaultdict
        incoming, outgoing = defaultdict(list), defaultdict(list)
        sources = df.from_node.to_numpy()
        targets = df.to_node.to_numpy()
        ids = df.link_id.to_numpy()
        speeds = df.freespeed.to_numpy()
        capacities = df.capacity.to_numpy()
        # Compare modes as sets: ordering in a comma-separated string is immaterial.
        mode_sets = {value: frozenset(value.split(",")) for value in df.modes.unique()}
        modes = [mode_sets[value] for value in df.modes]
        for i, (source, target) in enumerate(zip(sources, targets)):
            outgoing[source].append(i)
            incoming[target].append(i)
        nodes = set(incoming) | set(outgoing)
        stats["number_of_nodes"] = len(nodes)
        following, preceding = {}, {}
        road_modes = {"car", "car_passenger", "truck", "taxi", "bus"}

        for node in nodes:
            ins, outs = incoming.get(node, []), outgoing.get(node, [])
            stats["one_in_one_out"] += int(len(ins) == len(outs) == 1)
            # At most two directions, with exactly one link per direction.
            if not ins or not outs or len(ins) + len(outs) > 4:
                continue
            neighbours = {sources[i] for i in ins} | {targets[i] for i in outs}
            if node in neighbours or len(neighbours) != 2:
                continue
            stats["degree_is_2"] += 1
            incident = ins + outs
            if any(ids[i] in protected for i in incident):
                continue
            if any(not modes[i].intersection(road_modes) for i in incident):
                stats["ignored_no_car"] += 1
                continue
            # Straight-through continuation goes to the other physical neighbour,
            # never immediately back to the neighbour from which the link arrived.
            pairs = [(i, [j for j in outs if targets[j] != sources[i]]) for i in ins]
            if (any(len(candidates) != 1 for _, candidates in pairs)
                    or len({candidates[0] for _, candidates in pairs}) != len(outs)
                    or len(ins) != len(outs)):
                stats["ambiguous_nodes"] += 1
                continue
            pairs = [(i, candidates[0]) for i, candidates in pairs]
            if any(speeds[i] != speeds[j] or capacities[i] != capacities[j]
                   or modes[i] != modes[j] for i, j in pairs):
                stats["attributes_change"] += 1
                continue
            for i, j in pairs:
                following[i] = j
                preceding[j] = i

        removed, replacements = set(), []
        # A chain can start at an intersection, dead end or attribute boundary.
        # Closed rings have no start and are deliberately left intact.
        for first in following:
            if first in preceding:
                continue
            chain = [first]
            while chain[-1] in following:
                chain.append(following[chain[-1]])
            last = chain[-1]
            if sources[first] == targets[last]:
                stats["skiped_loop"] += 1
                continue
            row = df.iloc[first].to_dict()
            row["to_node"] = targets[last]
            row["length"] = df.iloc[chain].length.sum()
            # Preserve destination attributes without mutating the input dict.
            row["attributes"] = dict(df.iloc[last].attributes)
            row["attributes"]["old_link_id"] = "_".join(
                str(df.iloc[i].attributes.get("old_link_id", ids[i])) for i in chain
            )
            replacements.append(row)
            removed.update(chain)
            stats["successful_merge"] += 1

        if not replacements:
            return df.copy().reset_index(drop=True), stats
        retained = df.iloc[[i for i in range(len(df)) if i not in removed]]
        result = pd.concat([retained, pd.DataFrame(replacements, columns=df.columns)], ignore_index=True)
        return result, stats

    def add_bike_to_network(self):
        car_links = self.links.modes.str.split(',').map(lambda x: "car" in x)
        taxi_links = self.links.modes.str.split(',').map(lambda x: "taxi" in x)
        bus_links = self.links.modes.str.split(',').map(lambda x: "bus" in x)
        truck_links = self.links.modes.str.split(',').map(lambda x: "truck" in x)
        candidate_links = car_links | taxi_links | bus_links | truck_links
        
        # find largest connected component of this network and add bike to these links
        G = nx.DiGraph()
        G.add_edges_from(zip(self.links.loc[candidate_links,'from_node'], 
                             self.links.loc[candidate_links,'to_node']))
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        bike_links = candidate_links & (self.links.from_node.isin(largest_cc) & self.links.to_node.isin(largest_cc))

        # Add bike to the modes of the selected links
        self.links.loc[bike_links, "modes"] += ",bike"
        return self.links
