"""Directed connectivity with MATSim's complete disallowed link sequences."""
from array import array
from collections import defaultdict
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def turn_restrictions(attributes):
    value = attributes.get("disallowedNextLinks", {})
    return json.loads(value) if isinstance(value, str) else value


def restricted_link_ids(attributes):
    """Links whose identities must survive chain simplification."""
    protected = set()
    for link_id, attrs in attributes:
        restrictions = turn_restrictions(attrs)
        if any(restrictions.values()):
            protected.add(link_id)
            protected.update(link for sequences in restrictions.values()
                             for sequence in sequences for link in sequence)
    return protected


def largest_connected_links(links, mode):
    """Return physical link IDs represented in the largest strong component.

    A state is a link plus the longest relevant suffix of the route so far.
    Ordinary links need one state; extra states remember partial prohibitions.
    Only completing a prohibited sequence blocks a transition. Suffix matching
    also handles overlapping restrictions and paths that revisit the same link.
    Sparse arrays keep the graph practical for national networks.
    """
    ids = links.link_id.tolist()
    if not ids:
        return set()
    index = {link_id: i for i, link_id in enumerate(ids)}
    sources = links.from_node.to_numpy()
    targets = links.to_node.to_numpy()
    patterns, prefixes, restricted_sources = set(), {}, set()
    for link_id, attrs in zip(ids, links.attributes):
        for sequence in turn_restrictions(attrs).get(mode, []):
            pattern = (link_id, *sequence)
            # MATSim discards restrictions referencing unavailable mode links.
            if not sequence or any(link not in index for link in sequence):
                continue
            if any(targets[index[a]] != sources[index[b]] for a, b in zip(pattern, pattern[1:])):
                continue
            patterns.add(pattern)
            restricted_sources.add(link_id)
            for length in range(2, len(pattern)):
                prefix = pattern[:length]
                if prefix not in prefixes:
                    prefixes[prefix] = len(ids) + len(prefixes)

    histories = list(prefixes)
    state_link = np.concatenate((np.arange(len(ids)),
                                 np.array([index[p[-1]] for p in histories], dtype=np.int64)))
    successors = defaultdict(list)
    for i, node in enumerate(sources):
        successors[node].append(i)
    left, right = array("q"), array("q")
    for state, physical in enumerate(state_link):
        link_id = ids[physical]
        history = (link_id,) if state < len(ids) else histories[state - len(ids)]
        for next_link in successors.get(targets[physical], ()):
            destination = next_link
            if len(history) > 1 or link_id in restricted_sources:
                extended = (*history, ids[next_link])
                if any(extended[-length:] in patterns for length in range(2, len(extended) + 1)):
                    continue
                for length in range(len(extended), 1, -1):
                    prefix_state = prefixes.get(extended[-length:])
                    if prefix_state is not None:
                        destination = prefix_state
                        break
            left.append(state)
            right.append(destination)

    graph = csr_matrix((np.ones(len(left), dtype=np.int8),
                        (np.frombuffer(left, dtype=np.int64), np.frombuffer(right, dtype=np.int64))),
                       shape=(len(state_link), len(state_link)))
    _, labels = connected_components(graph, directed=True, connection="strong")
    # Count physical links, not additional restriction-history states.
    unique_memberships = np.unique(labels.astype(np.int64) * len(ids) + state_link)
    sizes = np.bincount(unique_memberships // len(ids))
    largest = sizes.argmax()
    return {ids[i] for i in state_link[labels == largest]}


def clean_restriction_references(df):
    """Drop sequences that use deleted links or modes no longer allowed.

    Other attributes and input dictionaries are not mutated.
    """
    modes = dict(zip(df.link_id, df.modes.map(lambda value: set(value.split(",")))))
    result = df.copy()
    for row in df.itertuples():
        if "disallowedNextLinks" not in row.attributes:
            continue
        original = turn_restrictions(row.attributes)
        cleaned = {}
        for mode, sequences in original.items():
            if mode not in modes[row.link_id]:
                continue
            kept = [sequence for sequence in sequences
                    if sequence and all(mode in modes.get(link, ()) for link in sequence)]
            if kept or not sequences:
                cleaned[mode] = kept
        if cleaned != original:
            attrs = dict(row.attributes)
            if cleaned:
                attrs["disallowedNextLinks"] = json.dumps(cleaned)
            else:
                attrs.pop("disallowedNextLinks", None)
            result.at[row.Index, "attributes"] = attrs
    return result
