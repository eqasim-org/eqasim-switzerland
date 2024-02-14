import synthesis.population.spatial.secondary.rda as rda
import sklearn.neighbors
import numpy as np

def compute_weight(purpose, duration, capacity):
    if purpose == "culture":
        if duration / 60 < 39.40000000000003:
            x = 0
        elif duration / 60 >= 39.40000000000003 and duration / 60 < 85.0: 
            x = 0.00548245614035088 * duration / 60 + -0.21600877192982482
        elif duration / 60 >= 85.0 and duration / 60 < 140.0: 
            x = 0.004545454545454545 * duration / 60 + -0.13636363636363635
        elif duration / 60 >= 140.0 and duration / 60 < 210.0: 
            x = 0.0035714285714285713 * duration / 60 + 0.0
        elif duration / 60 >= 210.0 and duration / 60 < 319.59999999999985: 
            x = 0.0022810218978102223 * duration / 60 + 0.2709854014598533
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 1.0: 
            y = np.inf * capacity + -np.inf
        elif capacity > 1.0 and capacity <= 4.0: 
            y = 0.08333333333333333 * capacity + 0.16666666666666669
        elif capacity > 4.0 and capacity <= 10.0: 
            y = 0.041666666666666664 * capacity + 0.33333333333333337
        elif capacity > 10.0 and capacity <= 19.0: 
            y = 0.027777777777777776 * capacity + 0.4722222222222222
        else:
            y = 1

    if purpose == "religion":
        if duration / 60 < 15.0:
            x = 0
        elif duration / 60 >= 15.0 and duration / 60 < 42.5: 
            x = 0.00909090909090909 * duration / 60 + -0.13636363636363635
        elif duration / 60 >= 42.5 and duration / 60 < 75.0: 
            x = 0.007692307692307693 * duration / 60 + -0.07692307692307698
        elif duration / 60 >= 75.0 and duration / 60 < 123.5: 
            x = 0.005154639175257732 * duration / 60 + 0.11340206185567014
        elif duration / 60 >= 123.5 and duration / 60 < 180.0: 
            x = 0.004424778761061947 * duration / 60 + 0.20353982300884954
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 7.0: 
            y = 0.041666666666666664 * capacity + -0.04166666666666663
        elif capacity > 7.0 and capacity <= 17.0: 
            y = 0.025 * capacity + 0.07499999999999996
        elif capacity > 17.0 and capacity <= 28.0: 
            y = 0.022727272727272728 * capacity + 0.11363636363636365
        elif capacity > 28.0 and capacity <= 39.0: 
            y = 0.022727272727272728 * capacity + 0.11363636363636365
        else:
            y = 1

    if purpose == "grocery":
        if duration / 60 < 5.0:
            x = 0
        elif duration / 60 >= 5.0 and duration / 60 < 10.0: 
            x = 0.05 * duration / 60 + -0.25
        elif duration / 60 >= 10.0 and duration / 60 < 20.0: 
            x = 0.025 * duration / 60 + 0.0
        elif duration / 60 >= 20.0 and duration / 60 < 41.0: 
            x = 0.011904761904761904 * duration / 60 + 0.2619047619047619
        elif duration / 60 >= 41.0 and duration / 60 < 70.0: 
            x = 0.008620689655172414 * duration / 60 + 0.39655172413793105
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 9.0: 
            y = 0.03125 * capacity + -0.03125
        elif capacity > 9.0 and capacity <= 17.0: 
            y = 0.03125 * capacity + -0.03125
        elif capacity > 17.0 and capacity <= 32.0: 
            y = 0.016666666666666666 * capacity + 0.21666666666666667
        elif capacity > 32.0 and capacity <= 48.0: 
            y = 0.015625 * capacity + 0.25
        else:
            y = 1

    if purpose == "other(S)":
        if duration / 60 < 5.0:
            x = 0
        elif duration / 60 >= 5.0 and duration / 60 < 10.0: 
            x = 0.05 * duration / 60 + -0.25
        elif duration / 60 >= 10.0 and duration / 60 < 30.0: 
            x = 0.0125 * duration / 60 + 0.125
        elif duration / 60 >= 30.0 and duration / 60 < 65.0: 
            x = 0.007142857142857143 * duration / 60 + 0.28571428571428575
        elif duration / 60 >= 65.0 and duration / 60 < 120.0: 
            x = 0.004545454545454545 * duration / 60 + 0.4545454545454546
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 5.0: 
            y = 0.0625 * capacity + -0.0625
        elif capacity > 5.0 and capacity <= 10.0: 
            y = 0.05 * capacity + 0.0
        elif capacity > 10.0 and capacity <= 15.0: 
            y = 0.05 * capacity + 0.0
        elif capacity > 15.0 and capacity <= 20.0: 
            y = 0.05 * capacity + 0.0
        else:
            y = 1

    if purpose == "gastronomy":
        if duration / 60 < 25.0:
            x = 0
        elif duration / 60 >= 25.0 and duration / 60 < 40.0: 
            x = 0.016666666666666666 * duration / 60 + -0.41666666666666663
        elif duration / 60 >= 40.0 and duration / 60 < 65.0: 
            x = 0.01 * duration / 60 + -0.15000000000000002
        elif duration / 60 >= 65.0 and duration / 60 < 120.0: 
            x = 0.004545454545454545 * duration / 60 + 0.20454545454545459
        elif duration / 60 >= 120.0 and duration / 60 < 194.0: 
            x = 0.0033783783783783786 * duration / 60 + 0.3445945945945945
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 8.0: 
            y = 0.03571428571428571 * capacity + -0.0357142857142857
        elif capacity > 8.0 and capacity <= 16.0: 
            y = 0.03125 * capacity + 0.0
        elif capacity > 16.0 and capacity <= 25.0: 
            y = 0.027777777777777776 * capacity + 0.05555555555555558
        elif capacity > 25.0 and capacity <= 33.57999999999811: 
            y = 0.02913752913753556 * capacity + 0.021561771561610987
        else:
            y = 1

    if purpose == "other(L)":
        if duration / 60 < 1.0:
            x = 0
        elif duration / 60 >= 1.0 and duration / 60 < 10.0: 
            x = 0.027777777777777776 * duration / 60 + -0.02777777777777779
        elif duration / 60 >= 10.0 and duration / 60 < 55.0: 
            x = 0.005555555555555556 * duration / 60 + 0.19444444444444442
        elif duration / 60 >= 55.0 and duration / 60 < 132.0: 
            x = 0.003246753246753247 * duration / 60 + 0.3214285714285714
        elif duration / 60 >= 132.0 and duration / 60 < 225.0: 
            x = 0.002688172043010753 * duration / 60 + 0.3951612903225806
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 1.0: 
            y = inf * capacity + -inf
        elif capacity > 1.0 and capacity <= 2.0: 
            y = 0.25 * capacity + 0.0
        elif capacity > 2.0 and capacity <= 6.0: 
            y = 0.0625 * capacity + 0.375
        elif capacity > 6.0 and capacity <= 10.0: 
            y = 0.0625 * capacity + 0.375
        else:
            y = 1

    if purpose == "services":
        if duration / 60 < 2.0:
            x = 0
        elif duration / 60 >= 2.0 and duration / 60 < 5.0: 
            x = 0.08333333333333333 * duration / 60 + -0.16666666666666663
        elif duration / 60 >= 5.0 and duration / 60 < 15.0: 
            x = 0.025 * duration / 60 + 0.125
        elif duration / 60 >= 15.0 and duration / 60 < 50.0: 
            x = 0.007142857142857143 * duration / 60 + 0.39285714285714285
        elif duration / 60 >= 50.0 and duration / 60 < 85.0: 
            x = 0.007142857142857143 * duration / 60 + 0.3928571428571429
        else:
            x = 1

        if capacity <= 1.0:
            y = 0
        elif capacity > 1.0 and capacity <= 1.0: 
            y = np.inf * capacity + -np.inf
        elif capacity > 1.0 and capacity <= 3.0: 
            y = 0.125 * capacity + 0.125
        elif capacity > 3.0 and capacity <= 6.0: 
            y = 0.08333333333333333 * capacity + 0.25
        elif capacity > 6.0 and capacity <= 10.0: 
            y = 0.0625 * capacity + 0.375
        else:
            y = 1

    return max(2**5*0.98*(x-0.5)**3*(y-0.5)**3+0.5,0)

def compute_all_weights(purpose, duration, capacities, openings):
    L = [compute_weight(purpose, duration, c) for c in capacities]
    if np.all(openings) == 0:
        return L
    L = [o*w for (o,w) in list(zip(openings, L))]
    return L


class CustomDistanceSampler(rda.FeasibleDistanceSampler):
    def __init__(self, random, distributions, maximum_iterations = 1000):
        rda.FeasibleDistanceSampler.__init__(self, random = random, maximum_iterations = maximum_iterations)

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
    

class CandidateIndex:
    def __init__(self, data):
        self.data = data
        self.indices = {}

        for purpose, data in self.data.items():
            print("Constructing spatial index for %s ..." % purpose)
            print(len(data["locations"]))
            self.indices[purpose] = sklearn.neighbors.KDTree(data["locations"])
            
    def query(self, purpose, location, start_time, act_dur, detailed, capacity_based_secondary_location_assignment = False):
        index = self.indices[purpose].query(location.reshape(1, -1), return_distance = False)[0][0]
        
        
        # For some activity chains (eg "start not from home -> outdoor" and that's it), not all of the activity durations and 
        # activity starting times are registered in the probelm characteristics so let's deactivate the location assignment
        # based on opening schedule and activity duration and number of employees for them
        if purpose in ["grocery", "other(S)", "culture", "religion", "other(L)", "services"] and detailed:
            # For these purposes, we try to restrict the opening times
            self.query_size = 3
            _, indices = self.indices[purpose].query(location.reshape(1, -1), self.query_size, return_distance=True)
            start = (start_time // 3600) % 24
            if capacity_based_secondary_location_assignment:
                candidates_nboemployees = self.data[purpose]["number_employees"][indices][0]
                if start < 3:
                    openings = self.data[purpose]["open03"][indices][0]
                elif start >= 3 and start < 6:
                    openings = self.data[purpose]["open36"][indices][0]
                elif start >= 6 and start < 9:
                    openings = self.data[purpose]["open69"][indices][0]
                elif start >= 9 and start < 12:
                    openings = self.data[purpose]["open912"][indices][0]
                elif start >= 12 and start < 15:
                    openings = self.data[purpose]["open1215"][indices][0]
                elif start >= 15 and start < 18:
                    openings = self.data[purpose]["open1518"][indices][0]
                elif start >= 18 and start < 21:
                    openings = self.data[purpose]["open1821"][indices][0]
                elif start >= 21 and start < 24:
                    openings = self.data[purpose]["open2124"][indices][0]
                weights = compute_all_weights(purpose, act_dur, candidates_nboemployees, openings)
                weights = weights / np.sum(weights)
                selector = np.random.choice(self.query_size, p=weights) 
            else:
                selector = np.random.choice(self.query_size)
            index = np.choose(selector, indices.T)

        else:
            index = self.indices[purpose].query(location.reshape(1, -1), return_distance = False)[0][0]


        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["locations"][index]
        return identifier, location

    def sample(self, purpose, random):
        index = random.randint(0, len(self.data[purpose]["locations"]))
        identifier = self.data[purpose]["identifiers"][index]
        location = self.data[purpose]["locations"][index]
        return identifier, location


class CustomDiscretizationSolver(rda.DiscretizationSolver):
    def __init__(self, index):
        self.index = index

        #for purpose, data in self.data.items():
        #    print("Constructing spatial index for %s ..." % purpose)
        #    self.indices[purpose] = sklearn.neighbors.KDTree(data["locations"])

    def solve(self, problem, locations):
        discretized_locations = []
        discretized_identifiers = []
        
        detailed = len(problem["purposes"]) == len(problem["activity_duration"])
        if detailed:
            for location, purpose, act_dur, start_time in zip(locations, problem["purposes"], problem["activity_duration"], problem["activity_start_time"]):
                identifier, location = self.index.query(purpose, location.reshape(1, -1), start_time, act_dur, detailed = detailed, capacity_based_secondary_location_assignment=False)
                discretized_identifiers.append(identifier)
                discretized_locations.append(location)
                
        else:
            for location, purpose in zip(locations, problem["purposes"]):
                identifier, location = self.index.query(purpose, location.reshape(1, -1), None, None, detailed = detailed, capacity_based_secondary_location_assignment=False)
                discretized_identifiers.append(identifier)
                discretized_locations.append(location)

        assert len(discretized_locations) == problem["size"]

        return dict(
            valid = True, locations = np.vstack(discretized_locations), identifiers = discretized_identifiers
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
