import synthesis.population.spatial.secondary.rda as rda
import sklearn.neighbors
import numpy as np
import logging
from synthesis.population.spatial.secondary.opening_hours import eligible_candidates

logger = logging.getLogger("synpp")


class DestinationAcceptance:
    """Soft proposal acceptance, with one factor per distinct activity purpose."""

    def __init__(self, destinations, random):
        self.random = random
        self.probabilities = {}
        for purpose, data in destinations.items():
            values = np.asarray(data.get("proposal_acceptance", np.ones(len(data["identifiers"]))), dtype=float)
            if values.shape != (len(data["identifiers"]),) or not np.all(
                    np.isfinite(values) & (values >= 0) & (values <= 1)):
                raise ValueError("Proposal acceptance must be aligned probabilities in [0, 1]")
            selected = values < 1
            self.probabilities[purpose] = dict(zip(data["identifiers"][selected], values[selected]))

    def probability(self, problem, discretization):
        by_purpose = {}
        for purpose, identifier in zip(problem["purposes"], discretization["identifiers"], strict=True):
            probability = self.probabilities.get(purpose, {}).get(identifier, 1.0)
            by_purpose[purpose] = min(by_purpose.get(purpose, 1.0), probability)
        # Stable ordering avoids process-dependent set iteration in products.
        return float(np.prod([by_purpose[purpose] for purpose in sorted(by_purpose)]))

    def accept(self, probability):
        # Deterministic cases do not advance this separate random stream.
        return probability >= 1 or (probability > 0 and self.random.random_sample() < probability)


class CustomDistanceSampler(rda.FeasibleDistanceSampler):
    def __init__(self, random, distributions, maximum_iterations = 1000):
        rda.FeasibleDistanceSampler.__init__(self, random=random, maximum_iterations=maximum_iterations)

        self.random = random
        self.distributions = distributions
        self._cached_problem = None
        self._selected_distributions = None
        self._sorted_cdfs = {
            id(item): bool(np.isfinite(item["cdf"]).all() and np.all(np.diff(item["cdf"]) >= 0))
            for mode in distributions.values() for item in mode["distributions"]
        }

    def sample_distances(self, problem):
        # Modes and travel times are fixed throughout one assignment problem.
        if self._cached_problem is not problem:
            selected = []
            for mode, travel_time in zip(problem["modes"], problem["travel_times"]):
                distribution = self.distributions["walk" if mode == "remote_walk" else mode]
                item = distribution["distributions"][np.count_nonzero(travel_time > distribution["bounds"])]
                selected.append((item["values"], item["cdf"], self._sorted_cdfs[id(item)]))
            self._cached_problem, self._selected_distributions = problem, selected
        distances = np.zeros(len(problem["modes"]))
        uniforms = self.random.random_sample(len(self._selected_distributions))
        for index, ((values, cdf, sorted_cdf), uniform) in enumerate(zip(self._selected_distributions, uniforms)):
            # side='left' exactly matches count(uniform > cdf), including ties.
            position = np.searchsorted(cdf, uniform, side="left") if sorted_cdf else np.count_nonzero(uniform > cdf)
            distances[index] = values[position]
        return distances


class CandidateIndex:
    def __init__(self, data, number_candidates = 10, alpha_probabilities = 1.0, random = None):
        self.data = data
        self.number_candidates = number_candidates
        self.alpha_probabilities = alpha_probabilities
        self.indices = {}
        self.weights = {}
        self.global_probabilities = {}
        self.random = random if random is not None else np.random.RandomState()

        for purpose, data in self.data.items():
            logger.info("Constructing spatial index for %s ...", purpose)
            self.indices[purpose] = sklearn.neighbors.KDTree(data["locations"])

            weights = data["number_employees"] ** self.alpha_probabilities
            self.weights[purpose] = weights

            total_weight = np.sum(weights)
            if total_weight > 0.0:
                self.global_probabilities[purpose] = weights / total_weight
            else:
                self.global_probabilities[purpose] = np.full_like(weights, 1.0 / len(weights), dtype=float)

    def query(self, purpose, location):
        location = np.asarray(location).reshape(1, -1)
        index = self.indices[purpose].query(location, k = self.number_candidates, return_distance = False)[0]
        w = self.weights[purpose][index]
        p = w / w.sum()
        chosen = self.random.choice(index, p=p)

        identifier = self.data[purpose]["identifiers"][chosen]
        location = self.data[purpose]["locations"][chosen]
        return identifier, location

    def query_batch(self, purpose, locations):
        locations = np.asarray(locations)
        indices = self.indices[purpose].query(locations, k = min(self.number_candidates, len(self.weights[purpose])), return_distance = False)

        candidate_weights = self.weights[purpose][indices]
        weight_sums = np.sum(candidate_weights, axis=1, keepdims=True)

        probabilities = np.divide(
            candidate_weights,
            weight_sums,
            out=np.full_like(candidate_weights, 1.0 / candidate_weights.shape[1], dtype=float),
            where=weight_sums > 0.0
        )

        cumulative = np.cumsum(probabilities, axis=1)
        random_values = self.random.random_sample(len(locations))
        chosen_columns = np.sum(random_values[:, np.newaxis] > cumulative, axis=1)
        chosen_columns = np.minimum(chosen_columns, indices.shape[1] - 1)

        chosen_indices = indices[np.arange(len(locations)), chosen_columns]

        identifiers = self.data[purpose]["identifiers"][chosen_indices]
        sampled_locations = self.data[purpose]["locations"][chosen_indices]

        return identifiers, sampled_locations

    def query_open_batch(self, purpose, locations, starts, ends):
        schedules = self.data[purpose].get("opening_schedule")
        if schedules is None:
            return self.query_batch(purpose, locations)
        identifiers, selected_locations = [], []
        total = len(self.weights[purpose])
        for location, start, end in zip(locations, starts, ends):
            k = min(self.number_candidates, total)
            while True:
                candidates = self.indices[purpose].query(np.asarray(location).reshape(1, -1),
                                                        k=k, return_distance=False)[0]
                candidates = candidates[eligible_candidates(schedules, candidates, start, end,
                    self.data[purpose].get("opening_tolerance_seconds", 600.))]
                if len(candidates):
                    # Retain the usual candidate count after filtering by hours.
                    if len(candidates) >= self.number_candidates or k == total:
                        candidates = candidates[:self.number_candidates]
                        break
                if k == total:
                    raise ValueError(f"No eligible {purpose} destination for activity {start:.0f}–{end:.0f}s")
                k = min(total, k * 2)
            weights = self.weights[purpose][candidates]
            probabilities = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
            chosen = self.random.choice(candidates, p=probabilities)
            identifiers.append(self.data[purpose]["identifiers"][chosen])
            selected_locations.append(self.data[purpose]["locations"][chosen])
        return np.asarray(identifiers), np.asarray(selected_locations)

    def sample(self, purpose, random):
        p = self.global_probabilities[purpose]
        chosen = random.choice(len(p), p=p)

        identifier = self.data[purpose]["identifiers"][chosen]
        location = self.data[purpose]["locations"][chosen]
        return identifier, location

class CustomDiscretizationSolver(rda.DiscretizationSolver):
    def __init__(self, index):
        self.index = index

    def solve(self, problem, locations):
        discretized_locations = np.empty_like(locations)
        discretized_identifiers = [None] * problem["size"]

        purposes = np.array(problem["purposes"])
        unique_purposes = np.unique(purposes)

        for purpose in unique_purposes:
            purpose_indices = np.where(purposes == purpose)[0]
            if "activity_start_times" in problem:
                identifiers, purpose_locations = self.index.query_open_batch(
                    purpose, locations[purpose_indices], problem["activity_start_times"][purpose_indices],
                    problem["activity_end_times"][purpose_indices])
            else:
                identifiers, purpose_locations = self.index.query_batch(purpose, locations[purpose_indices])

            discretized_locations[purpose_indices] = purpose_locations

            for local_index, identifier in zip(purpose_indices, identifiers):
                discretized_identifiers[local_index] = identifier

        assert len(discretized_locations) == problem["size"]

        return dict(
            valid = True, locations = discretized_locations, identifiers = discretized_identifiers
        )


class CustomFreeChainSolver(rda.RelaxationSolver):
    def __init__(self, random, index):
        self.random = random
        self.index = index

    def solve(self, problem, distances):
        identifier, anchor = self.index.sample(problem["purposes"][0], self.random)
        locations = rda.sample_tail(self.random, anchor, distances)
        locations = np.vstack((anchor, locations))

        assert len(locations) == len(distances) + 1
        return dict(valid = True, locations = locations)
