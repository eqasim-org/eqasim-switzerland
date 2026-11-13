"""Optional local Swiss-facility repair of exhausted country-preference chains."""
import copy

import numpy as np
from sklearn.neighbors import KDTree
from synthesis.population.spatial.secondary.opening_hours import eligible_candidates


class SwissFallbackRepair:
    """Keep sampled distances and anchors; replace restricted French facilities only.

    Failed repairs return the original result. Separate random streams leave the
    normal solver's subsequent proposals unchanged, whether or not repair succeeds.
    """

    def __init__(self, destinations, objective, random_seed, maximum_attempts,
                 number_candidates=20, alpha_probabilities=0.8):
        self.destinations = destinations
        self.maximum_attempts = maximum_attempts
        self.number_candidates = number_candidates
        self.alpha_probabilities = alpha_probabilities
        self.random = np.random.RandomState((int(random_seed) + 2000003) % (2**32))
        self.objective = copy.copy(objective)
        self.objective.random = np.random.RandomState((int(random_seed) + 3000001) % (2**32))
        self.indices = {}
        self.restricted = {
            purpose: set(data["identifiers"][data["proposal_acceptance"] < 1])
            for purpose, data in destinations.items()
        }

    def candidates(self, purpose, location):
        if purpose not in self.indices:
            data = self.destinations[purpose]
            indices = np.flatnonzero(data["repair_swiss"])
            self.indices[purpose] = (
                indices, KDTree(data["locations"][indices]) if len(indices) else None)
        indices, tree = self.indices[purpose]
        if tree is None:
            return np.array([], dtype=int)
        return indices[tree.query(np.asarray(location).reshape(1, -1),
                                  k=min(self.number_candidates, len(indices)),
                                  return_distance=False)[0]]

    def repair(self, problem, original):
        diagnostics = dict(repair_attempted=True, repair_trials=0,
                           repair_success=False, repair_status="exhausted")
        # Discretization alone cannot fix failed sampling or continuous relaxation.
        if not original["distance"]["valid"] or not original["relaxation"]["valid"]:
            diagnostics["repair_status"] = "upstream_invalid"
            return original, diagnostics

        discretization = original["discretization"]
        replacements = []
        for index, (purpose, identifier) in enumerate(zip(
                problem["purposes"], discretization["identifiers"], strict=True)):
            if identifier not in self.restricted.get(purpose, set()):
                continue
            candidates = self.candidates(purpose, discretization["locations"][index])
            data = self.destinations[purpose]
            if "opening_schedule" in data and "activity_start_times" in problem:
                candidates = candidates[eligible_candidates(
                    data["opening_schedule"], candidates,
                    problem["activity_start_times"][index], problem["activity_end_times"][index],
                    data.get("opening_tolerance_seconds", 600.))]
            if not len(candidates):
                diagnostics["repair_status"] = "no_swiss_candidates"
                return original, diagnostics
            weights = self.destinations[purpose]["number_employees"][candidates] ** self.alpha_probabilities
            total = weights.sum()
            probabilities = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
            replacements.append((index, purpose, candidates, probabilities))
        if not replacements:
            diagnostics["repair_status"] = "not_applicable"
            return original, diagnostics

        # Common detour random numbers across candidates prevent repeated tests of
        # the same geometry from passing just because of a different random draw.
        self.objective.random.random_sample()
        objective_state = self.objective.random.get_state()
        tested = set()
        for attempt in range(self.maximum_attempts):
            locations = discretization["locations"].copy()
            identifiers = list(discretization["identifiers"])
            for index, purpose, candidates, probabilities in replacements:
                # Include the closest Swiss combination, then retain attractiveness.
                chosen = candidates[0] if attempt == 0 else self.random.choice(candidates, p=probabilities)
                data = self.destinations[purpose]
                locations[index] = data["locations"][chosen]
                identifiers[index] = data["identifiers"][chosen]
            signature = tuple(identifiers)
            if signature in tested:
                continue
            tested.add(signature)
            diagnostics["repair_trials"] += 1
            candidate = dict(valid=True, locations=locations, identifiers=identifiers)
            self.objective.random.set_state(objective_state)
            score = self.objective.evaluate(problem, original["distance"], original["relaxation"], candidate)
            if not score["valid"]:
                continue
            if original["valid"] and score["objective"] > original["objective"]:
                continue
            result = dict(original, **score)
            result["discretization"] = candidate
            diagnostics.update(repair_success=True, repair_status="repaired")
            return result, diagnostics
        return original, diagnostics
