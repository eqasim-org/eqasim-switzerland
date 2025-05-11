import synthesis.population.spatial.secondary.rda as rda
import sklearn.neighbors
import numpy as np

class CustomDistanceSampler(rda.FeasibleDistanceSampler):
    def __init__(self, random, distributions, maximum_iterations = 1000):
        rda.FeasibleDistanceSampler.__init__(self, random=random, maximum_iterations=maximum_iterations)
        self.random = random
        self.distributions = distributions

    def sample_distances(self, problem):
        distances = np.zeros((problem["size"] + 1))
        for index, (mode, travel_time) in enumerate(zip(problem["modes"], problem["travel_times"])):
            mode_distribution = self.distributions[mode]
            bound_index = np.count_nonzero(travel_time > mode_distribution["bounds"])
            mode_distribution = mode_distribution["distributions"][bound_index]
            distances[index] = mode_distribution["values"][
                np.count_nonzero(self.random.random_sample() > mode_distribution["cdf"])
            ]
        return distances

class CustomDiscretizationSolver(rda.DiscretizationSolver):
    def __init__(self, data):
        self.data = data
        self.indices = {
            purpose: sklearn.neighbors.KDTree(data[purpose]["locations"])
            for purpose in data
        }

    def solve(self, problem, locations):
        purposes = np.array(problem["purposes"])
        locations = np.array(locations)
        discretized_identifiers = [None] * len(locations)
        discretized_locations = [None] * len(locations)
        for purpose in np.unique(purposes):
            purpose_mask = (purposes == purpose)
            purpose_locs = locations[purpose_mask]
            tree = self.indices[purpose]
            indices = tree.query(purpose_locs, return_distance=False)[:, 0]
            ids = self.data[purpose]["identifiers"]
            locs = self.data[purpose]["locations"]
            for idx_in_batch, idx_in_data in enumerate(indices):
                original_idx = np.where(purpose_mask)[0][idx_in_batch]
                discretized_identifiers[original_idx] = ids[idx_in_data]
                discretized_locations[original_idx] = locs[idx_in_data]
        return dict(
            valid=True,
            locations=np.vstack(discretized_locations),
            identifiers=discretized_identifiers
        )
