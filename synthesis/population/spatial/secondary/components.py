import synthesis.population.spatial.secondary.rda as rda
import sklearn.neighbors
import numpy as np
import logging

logger = logging.getLogger("synpp")

class CustomDistanceSampler(rda.FeasibleDistanceSampler):
    def __init__(self, random, distributions, maximum_iterations = 1000):
        rda.FeasibleDistanceSampler.__init__(self, random=random, maximum_iterations=maximum_iterations)

        self.random = random
        self.distributions = distributions

    def sample_distances(self, problem):
        distances = np.zeros((len(problem["modes"])))

        for index, (mode, travel_time) in enumerate(zip(problem["modes"], problem["travel_times"])):
            if mode == "remote_walk":
                mode = "walk"
            mode_distribution = self.distributions[mode]

            bound_index = np.count_nonzero(travel_time > mode_distribution["bounds"])
            mode_distribution = mode_distribution["distributions"][bound_index]

            distances[index] = mode_distribution["values"][
                np.count_nonzero(self.random.random_sample() > mode_distribution["cdf"])
            ]

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
        indices = self.indices[purpose].query(locations, k = self.number_candidates, return_distance = False)

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

    def sample(self, purpose, random):
        p = self.global_probabilities[purpose]
        chosen = random.choice(np.arange(len(p)), p=p)

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
