"""Proper distribution scores with fixed physical scales and bounded memory."""

import numpy as np
import torch


def discrete_crps(probabilities, values, observation):
    """Exact CRPS for a discrete scalar distribution, differentiable in its masses."""
    values, order = torch.sort(values, dim=-1)
    probabilities = torch.gather(probabilities, -1, order)
    cumulative = probabilities.cumsum(dim=-1)
    first = (probabilities * (values - observation[..., None]).abs()).sum(dim=-1)
    half_pairwise = (probabilities * values * (2 * cumulative - probabilities - 1)).sum(dim=-1)
    return (first - half_pairwise).clamp_min(0)


def projected_energy_score(probabilities, candidate_xy, observed_xy, projections=8):
    """Angular quadrature of the 2-D energy score using projected exact CRPS.

    Each projection is proper. Fixed directions approximate the Euclidean energy
    score without allocating a batch-by-candidate-by-candidate distance tensor.
    """
    if projections < 2:
        raise ValueError("At least two spatial projections are required")
    centered = candidate_xy - observed_xy[:, None, :]
    total = probabilities.new_zeros(probabilities.shape[0])
    for angle in np.arange(projections) * np.pi / projections:
        direction = probabilities.new_tensor([np.cos(angle), np.sin(angle)])
        positions = centered @ direction
        total = total + discrete_crps(probabilities, positions, torch.zeros_like(total))
    return total * (np.pi / (2 * projections))


def ensemble_score_components(short, hierarchy, origin, target, scale_m=1000.0):
    """Cache sufficient statistics for exact CRPS/energy of a two-expert mixture."""
    short, hierarchy = np.asarray(short, dtype=float), np.asarray(hierarchy, dtype=float)
    origin, target = np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
    if scale_m <= 0 or short.shape != hierarchy.shape or short.ndim != 2 or short.shape[1] != 2:
        raise ValueError("Expert draws must be matching nonempty coordinate arrays")
    if not len(short) or not np.isfinite(np.concatenate([short.ravel(), hierarchy.ravel(), origin, target])).all():
        raise ValueError("Expert scores require finite coordinates")
    result = {}
    for name, a, b, y in [
        ('energy', short / scale_m, hierarchy / scale_m, target / scale_m),
        ('crps', np.linalg.norm(short - origin, axis=1)[:, None] / scale_m,
         np.linalg.norm(hierarchy - origin, axis=1)[:, None] / scale_m,
         np.array([np.linalg.norm(target - origin) / scale_m])),
    ]:
        result[name + '_short_target'] = float(np.linalg.norm(a - y, axis=1).mean())
        result[name + '_hierarchy_target'] = float(np.linalg.norm(b - y, axis=1).mean())
        # Exclude self-pairs for unbiased estimates of the underlying expert score.
        n = len(a)
        if n < 2:
            raise ValueError("At least two draws per expert are required")
        result[name + '_short_pair'] = float(np.linalg.norm(a[:, None] - a[None, :], axis=-1).sum() / (n * (n - 1)))
        result[name + '_hierarchy_pair'] = float(np.linalg.norm(b[:, None] - b[None, :], axis=-1).sum() / (n * (n - 1)))
        result[name + '_cross_pair'] = float(np.linalg.norm(a[:, None] - b[None, :], axis=-1).mean())
    return result


def mixture_score(probability, statistics, name):
    """CRPS or energy score of g P_short + (1-g) P_hierarchy."""
    g = probability
    return (g * statistics[name + '_short_target'] + (1 - g) * statistics[name + '_hierarchy_target']
            - .5 * (g.square() * statistics[name + '_short_pair']
                     + (1 - g).square() * statistics[name + '_hierarchy_pair']
                     + 2 * g * (1 - g) * statistics[name + '_cross_pair']))
