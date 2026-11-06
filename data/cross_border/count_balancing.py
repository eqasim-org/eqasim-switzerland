"""Small dual optimisation for geographically constrained crossing shares."""

from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, vstack
from scipy.special import logsumexp


def allocate(mass, cost, target, allowed, penalty=10., temperature=5.):
    """Smooth geographical allocation with soft observed-count margins.

    Every OD row retains its demand. The bounded crossing-variable dual penalises
    column deviations instead of moving geographically implausible trips to
    force an exact count fit. All time parameters are minutes per crossing.
    """
    if ((mass <= 0).any() or not np.isfinite(target).all()
            or (penalty is None and (target < 0).any())
            or not np.isclose(mass.sum(), target.sum())):
        raise ValueError('Positive row mass and equal row/column totals required')
    if not allowed.any(axis=1).all():
        raise ValueError('An OD has no feasible crossing')
    extra = cost - np.min(np.where(allowed, cost, np.inf), axis=1)[:, None]
    kernel = np.where(allowed, -extra / temperature, -np.inf)

    def objective(bias):
        score = kernel + bias[None, :] / temperature
        normaliser = logsumexp(score, axis=1)
        flows = mass[:, None] * np.exp(score - normaliser[:, None])
        value = temperature * np.dot(mass, normaliser) - np.dot(target, bias)
        return value / mass.sum(), (flows.sum(axis=0) - target) / mass.sum()

    bounds = None if penalty is None else [(-penalty, penalty)] * cost.shape[1]
    started = perf_counter()
    fit = minimize(objective, np.zeros(cost.shape[1]), jac=True, method='L-BFGS-B',
                   bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 2000})
    if not fit.success:
        raise RuntimeError(fit.message)
    score = kernel + fit.x[None, :] / temperature
    flows = mass[:, None] * np.exp(score - logsumexp(score, axis=1)[:, None])
    assert np.allclose(flows.sum(axis=1), mass)
    assert np.all(flows[~allowed] == 0)
    return flows, dict(seconds=perf_counter()-started, iterations=int(fit.nit),
                       bias_minutes=fit.x.tolist())


def feasible(mass, target, allowed):
    """Check all joint supply constraints, not only each crossing separately."""
    # Feasibility depends only on a row's allowed crossings. Combine identical
    # choice sets before the LP: at nine crossings there are at most 511 useful
    # patterns, even for millions of people. This avoids a large degenerate LP.
    allowed, pattern = np.unique(allowed, axis=0, return_inverse=True)
    mass = np.bincount(pattern, weights=mass, minlength=len(allowed))
    i, j = np.where(allowed)
    n = len(i)
    rows = coo_matrix((np.ones(n), (i, np.arange(n))), shape=(len(mass), n))
    columns = coo_matrix((np.ones(n), (j, np.arange(n))), shape=(len(target), n))
    fit = linprog(np.zeros(n), A_eq=vstack([rows, columns]), b_eq=np.r_[mass, target],
                  bounds=(0, None), method='highs')
    return bool(fit.success)



class CountAssignmentConflict(RuntimeError):
    """Count margins cannot be honoured within the configured geographic limit."""


def exact_support(mass, target, extra_minutes, maximum=30.):
    """Find the smallest feasible minute allowance from 20 up to the ceiling.

    Count shares remain hard constraints. Binary search uses only the small
    support-pattern LP; road costs are computed once. Whole-minute resolution
    avoids implying precision beyond this free-flow routing approximation.
    """
    if not np.isfinite(maximum) or maximum < 20:
        raise ValueError("cross_border_count_max_extra_minutes must be finite and at least 20")
    if not np.isfinite(target).all() or (target < 0).any():
        raise CountAssignmentConflict("Fixed crossings exceed a site's target; geographic expansion cannot resolve this")
    if not np.isclose(mass.sum(), target.sum()):
        raise CountAssignmentConflict("Remaining crossing demand and count targets have different totals")
    # A fractional user ceiling is tried as the last step, never rounded upward.
    caps = np.unique(np.r_[np.arange(20., maximum, 1.), maximum])
    checks = {}

    def supported(index):
        if index not in checks:
            allowed = np.isfinite(extra_minutes) & (extra_minutes <= caps[index])
            checks[index] = bool(allowed.any(axis=1).all() and feasible(mass, target, allowed))
        return checks[index]

    if supported(0):
        selected = 0
    else:
        if not supported(len(caps)-1):
            raise CountAssignmentConflict(
                "Exact crossing shares remain infeasible at %.1f extra Swiss minutes; "
                "review count coverage, fixed demand and geographic support. "
                "No soft-count fallback was applied." % maximum)
        low, high = 0, len(caps)-1
        while high-low > 1:
            mid = (low+high)//2
            if supported(mid):
                high = mid
            else:
                low = mid
        selected = high
    cap = float(caps[selected])
    return np.isfinite(extra_minutes) & (extra_minutes <= cap), cap, len(checks)
