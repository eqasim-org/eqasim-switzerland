"""Remaining plan features available before choosing the next secondary place."""

import numpy as np
from .hierarchical_utils import SECONDARY_ACTIVITIES, ORIGIN_PURPOSE_REMAP, PRIMARY_ACTIVITIES

CHAIN_STATE_FEATURES = [
    'remaining_distance_budget_m', 'remaining_trip_count', 'budget_per_remaining_trip_m',
    'consumed_budget_fraction',
] + ['remaining_' + purpose for purpose in SECONDARY_ACTIVITIES] + [
    'next_anchor_distance_m', 'trips_to_next_anchor', 'has_next_anchor',
    'next_anchor_home', 'next_anchor_work', 'next_anchor_education', 'origin_ovgk',
]


def chain_state_vector(daily_total, consumed_before, remaining_purposes, origin_x, origin_y,
                       primary_locations=None, next_anchor=None, origin_ovgk=4.):
    """Use plan purposes and known primary coordinates; never future secondary coordinates."""
    purposes = list(remaining_purposes)
    budget = max(0., float(daily_total) - float(consumed_before))
    counts = np.zeros(len(SECONDARY_ACTIVITIES))
    for purpose in purposes:
        purpose = ORIGIN_PURPOSE_REMAP.get(str(purpose), str(purpose))
        if purpose in SECONDARY_ACTIVITIES:
            counts[SECONDARY_ACTIVITIES.index(purpose)] += 1
    if next_anchor is None and primary_locations:
        for offset, purpose in enumerate(purposes):
            point = primary_locations.get(purpose)
            if purpose in PRIMARY_ACTIVITIES and point is not None and np.isfinite(point).all():
                next_anchor = (purpose, *point, offset + 1)
                break
    anchor = np.zeros(6)
    if next_anchor is not None:
        purpose, x, y, steps = next_anchor
        if purpose not in PRIMARY_ACTIVITIES or not np.isfinite([x, y]).all():
            raise ValueError('Next anchor must be a known finite primary location')
        anchor[:3] = [np.hypot(x - origin_x, y - origin_y), steps, 1]
        anchor[3 + PRIMARY_ACTIVITIES.index(purpose)] = 1
    result = np.concatenate([[budget, len(purposes), budget / max(len(purposes), 1),
                              max(0., float(consumed_before)) / max(float(daily_total), 1.)], counts, anchor, [origin_ovgk]])
    if not np.isfinite(result).all():
        raise ValueError('Remaining-chain features must be finite')
    return result.astype(np.float32)


def survey_chain_states(trips):
    """Read only the next primary anchor from each observed, ordered diary."""
    result = np.zeros((len(trips), len(CHAIN_STATE_FEATURES)), dtype=np.float32)
    for indices in trips.groupby('person_id', sort=False).indices.values():
        group = trips.iloc[indices]
        purposes = group.purpose.to_numpy()
        anchor = None
        for i in range(len(group) - 1, -1, -1):
            row = group.iloc[i]
            if purposes[i] in PRIMARY_ACTIVITIES and np.isfinite([row.destination_x, row.destination_y]).all():
                anchor = (purposes[i], row.destination_x, row.destination_y, i)
            next_anchor = None if anchor is None or not np.isfinite([row.origin_x, row.origin_y]).all() else (*anchor[:3], anchor[3] - i + 1)
            result[indices[i]] = chain_state_vector(
                row.daily_crowfly_total, row.crowfly_consumed_before_trip, purposes[i:],
                row.origin_x, row.origin_y, next_anchor=next_anchor, origin_ovgk=row.origin_ovgk if "origin_ovgk" in group else 4.)
    return list(result)
