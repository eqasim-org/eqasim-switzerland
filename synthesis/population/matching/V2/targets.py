"""Observed whole-day sequence targets, masks and behavioural scaling."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_sequence_targets(source_ids, trips, max_sequence_len=12, trip_count_scale=5.0,
                           trip_distance_scale=30000.0, sample_weights=None):
    """Learn full counts and masked times; matching still transfers the complete diary."""
    if max_sequence_len < 1 or trip_count_scale <= 0 or trip_distance_scale <= 0:
        raise ValueError("Sequence length and target scales must be positive")
    if trips.duplicated(["person_id", "trip_id"]).any():
        raise ValueError("Microcensus trips contain duplicate person/trip keys")
    trips = trips.sort_values(["person_id", "trip_id"])
    groups = {pid: group for pid, group in trips.groupby("person_id", sort=False)}
    activities = ["<pad>", "<bos>", "home"] + sorted(set(trips["purpose"].dropna().astype(str)) - {"home"})
    modes = ["<pad>", "<bos>", "no trip"] + sorted(set(trips["mode"].dropna().astype(str)) - {"no trip"})
    act_map = {value: i for i, value in enumerate(activities)}
    mode_map = {value: i for i, value in enumerate(modes)}
    n = len(source_ids)
    length = max(1, min(int(max_sequence_len), max((len(g) for g in groups.values()), default=1)))
    targets = {"act_tgt": np.full((n, length), -100, dtype=np.int64),
               "mode_tgt": np.full((n, length), -100, dtype=np.int64),
               "trip_count_tgt": np.zeros(n, dtype=np.float32)}
    for name in ("dep", "dist", "duration", "activity_duration"):
        targets[name + "_tgt"] = np.zeros((n, length), dtype=np.float32)
        targets[name + "_mask"] = np.zeros((n, length), dtype=np.float32)
    truncated = 0
    for i, pid in enumerate(source_ids):
        group = groups.get(pid)
        if group is None:
            targets["act_tgt"][i, 0] = act_map["home"]
            targets["mode_tgt"][i, 0] = mode_map["no trip"]
            continue
        count = len(group)
        truncated += count > length
        targets["trip_count_tgt"][i] = np.log1p(count) / np.log1p(trip_count_scale)
        take = min(count, length)
        targets["act_tgt"][i, :take] = [act_map.get(v, -100) for v in group["purpose"].iloc[:take]]
        targets["mode_tgt"][i, :take] = [mode_map.get(v, -100) for v in group["mode"].iloc[:take]]
        departure = pd.to_numeric(group["departure_time"], errors="coerce").to_numpy(dtype=float)
        arrival = pd.to_numeric(group["arrival_time"], errors="coerce").to_numpy(dtype=float)
        distance = pd.to_numeric(group["crowfly_distance"], errors="coerce").to_numpy(dtype=float)
        duration = arrival - departure
        dwell = np.r_[departure[1:] - arrival[:-1], np.nan]
        for name, values, scale, logarithmic in (
            ("dep", departure, 30 * 3600.0, False),
            ("dist", distance, trip_distance_scale, True),
            ("duration", duration, 3600.0, True),
            ("activity_duration", dwell, 3600.0, True),
        ):
            values = values[:take]
            valid = np.isfinite(values) & (values >= 0)
            if name == "dep":
                valid &= values <= scale
            normalized = np.zeros(take)
            normalized[valid] = (np.log1p(values[valid]) / np.log1p(scale) if logarithmic
                                 else np.clip(values[valid] / scale, 0, 1))
            targets[name + "_tgt"][i, :take] = normalized
            targets[name + "_mask"][i, :take] = valid
    weights = np.ones(n) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("Training survey weights must be finite and positive")
    targets.update(sample_weight=(weights / weights.mean()).astype(np.float32),
                   num_activity_classes=len(activities), num_mode_classes=len(modes),
                   activity_tokens=activities, mode_tokens=modes, truncated_diaries=int(truncated))
    return targets


@dataclass(frozen=True)
class SequenceTargetBuilder:
    """Configure sequence length and scales once for all training targets."""

    max_sequence_len: int
    trip_count_scale: float
    trip_distance_scale: float

    def build(self, donors, trips):
        return build_sequence_targets(donors["mz_id"].to_numpy(), trips,
            max_sequence_len=self.max_sequence_len, trip_count_scale=self.trip_count_scale,
            trip_distance_scale=self.trip_distance_scale,
            sample_weights=donors["person_weight"].to_numpy())
