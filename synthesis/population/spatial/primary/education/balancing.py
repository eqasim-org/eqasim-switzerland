"""Balance supported domestic OD flows against provisional enrolment shares."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from data.statent.education_institutions import pooled_compatibility
from data.statent.education_campus_weights import DEFAULT_ELIGIBILITY_RULES
from synthesis.population.spatial.primary.education.tertiary import domestic_cohort, zone_candidates


def configure(context):
    context.config("data_path")
    for stage in ["synthesis.population.sampled",
                  "synthesis.population.spatial.primary.education.targets",
                  "data.statent.education_institutions", "data.statent.statent",
                  "data.spatial.zones", "data.od.tertiary_education_matrix"]:
        context.stage(stage)
    context.config("education_site_eligibility", str(Path(context.config("data_path")) / DEFAULT_ELIGIBILITY_RULES))
    context.config("education_tertiary_balance", True)


def validate(context):
    import hashlib
    from pathlib import Path
    import data.statent.education_institutions as institutions
    files = [Path(__file__).with_name("tertiary.py"), Path(institutions.__file__),
             Path(context.config("education_site_eligibility"))]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def fit_flows(seed, row_totals, targets, tolerance=1e-7, max_iterations=2000):
    """IPF preserves structural zeros; final rows always conserve student counts.

    Infeasible margins return explicit residuals rather than inventing OD support.
    Convergence applies to fractional flows; controlled rounding follows below.
    """
    seed = np.asarray(seed, dtype=float)
    rows, columns = np.asarray(row_totals, dtype=float), np.asarray(targets, dtype=float)
    if seed.shape != (len(rows), len(columns)) or not all(
            np.isfinite(a).all() and (a >= 0).all() for a in [seed, rows, columns]):
        raise ValueError("Balancing requires finite nonnegative flows and margins")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Balancing requires positive iteration and tolerance settings")
    if not np.isclose(rows.sum(), columns.sum()):
        raise ValueError("Balancing row and target totals must agree")
    if ((seed.sum(axis=1) == 0) & (rows > 0)).any():
        raise ValueError("Positive origin counts require destination support")
    flows = seed.copy()
    reachable = (seed[rows > 0] > 0).any(axis=0)
    converged = False
    for iteration in range(max_iterations):
        sums = flows.sum(axis=0)
        flows *= np.divide(columns, sums, out=np.zeros_like(columns), where=sums > 0)
        sums = flows.sum(axis=1)
        # An origin supporting only zero-target groups still needs a destination.
        stranded = (sums == 0) & (rows > 0)
        flows[stranded] = seed[stranded]
        sums = flows.sum(axis=1)
        flows *= np.divide(rows, sums, out=np.zeros_like(rows), where=sums > 0)[:, None]
        if np.max(np.abs(flows.sum(axis=0) - columns), initial=0) <= tolerance * max(1., rows.sum()):
            converged = True
            break
    return flows, {"converged": converged, "iterations": iteration + 1,
                   "reachable": reachable, "residual": flows.sum(axis=0) - columns}


def integerise_flows(flows, row_totals):
    """Round fitted flows while preserving origins, support and campus margins.

    Each cell and column is rounded to its floor or ceiling. This is a bipartite
    network-flow problem: its LP has integral vertices, so a sparse HiGHS solve
    avoids a general integer programme. Independent destination draws would
    otherwise undo calibrated campus totals, especially in small populations.
    """
    flows = np.asarray(flows, dtype=float)
    rows = np.asarray(row_totals, dtype=int)
    if (flows.ndim != 2 or len(rows) != len(flows) or not np.isfinite(flows).all()
            or not np.array_equal(rows, np.asarray(row_totals))
            or (flows < 0).any() or (rows < 0).any()
            or not np.allclose(flows.sum(axis=1), rows, atol=1e-6, rtol=0)):
        raise ValueError("Integer rounding requires nonnegative flows with integer origin totals")
    floors = np.floor(flows + 1e-9).astype(int)
    remaining = rows - floors.sum(axis=1)
    fractional = np.maximum(flows - floors, 0.)
    origin, destination = np.nonzero(fractional > 1e-9)
    if not len(origin):
        if remaining.any():
            raise ValueError("No fractional support for remaining students")
        return floors
    variables = np.arange(len(origin))
    row_constraints = coo_matrix((np.ones(len(origin)), (origin, variables)),
                                 shape=(len(rows), len(origin))).tocsr()
    column_constraints = coo_matrix((np.ones(len(origin)), (destination, variables)),
                                    shape=(flows.shape[1], len(origin))).tocsr()
    totals = flows.sum(axis=0)
    # Small numerical IPF residuals near exact integers must not add fake slots.
    lower = np.floor(totals + 1e-8).astype(int) - floors.sum(axis=0)
    upper = np.ceil(totals - 1e-8).astype(int) - floors.sum(axis=0)
    cost = 1. - 2. * fractional[origin, destination]
    result = linprog(cost, A_eq=row_constraints, b_eq=remaining,
                     A_ub=vstack([column_constraints, -column_constraints]),
                     b_ub=np.concatenate([upper, -lower]), bounds=(0., 1.), method="highs")
    if not result.success or not np.allclose(result.x, np.rint(result.x), atol=1e-6, rtol=0):
        raise ValueError("Could not round supported education flows integrally: %s" % result.message)
    integer = floors.copy()
    integer[origin, destination] += np.rint(result.x).astype(int)
    if (not np.array_equal(integer.sum(axis=1), rows)
            or np.any((flows == 0) & (integer > 0))
            or np.any(np.abs(integer.sum(axis=0) - totals) > 1. + 1e-6)):
        raise ValueError("Rounded education flows violate origin totals, support or margins")
    return integer


def build_balancing(persons, active_ids, statent, registry, compatibility, zones, pdf, targets):
    cohort = domestic_cohort(persons, active_ids)
    if cohort["person_id"].duplicated().any():
        raise ValueError("Calibration cohort must contain unique person IDs")
    positions = dict(zip(zones["zone_id"], range(len(zones))))
    valid_origins = set(zones.loc[zones["zone_level"].isin(["municipality", "quarter"]), "zone_id"])
    if zones["zone_id"].duplicated().any() or np.shape(pdf) != (len(zones), len(zones)):
        raise ValueError("Balancing requires the matrix's unique spatial-zone ordering")
    models, audits = {}, []
    for level, students in cohort.groupby("study_level", sort=True):
        type_targets = targets["institution_type_targets"].query("study_level == @level")
        if type_targets.empty:  # Higher vocational education has no BFS reference here.
            continue
        campus_targets = targets["campus_targets"].query("study_level == @level")
        # These are disjoint margins: named campuses get their own BFS shares,
        # while each residual UH/FH group receives the remainder of that type.
        # Counting both a whole type and its campuses as separate destinations
        # would duplicate students and overallocate the linked institutions.
        shares, campus_keys = {}, {}
        for row in campus_targets.itertuples():
            key = "campus:" + row.campus_id
            campus_keys[row.campus_id] = key
            shares[key] = row.institution_enrolled_students / row.bfs_level_enrolment * row.campus_share
        for row in type_targets.itertuples():
            named_share = sum(shares["campus:" + c.campus_id] for c in campus_targets.itertuples()
                              if c.institution_type == row.institution_type)
            residual = row.enrolment_share - named_share
            if residual < -1e-10:
                raise ValueError("Mapped campus shares exceed their institution-type total")
            shares["other:" + row.institution_type] = max(0., residual)
        if not np.isfinite(list(shares.values())).all() or not np.isclose(sum(shares.values()), 1.):
            raise ValueError("Unresolved enrolment shares cannot be used for balancing")
        sites, pools = zone_candidates(statent, compatibility, level)
        metadata = registry["establishments"].set_index("enterprise_id")
        sites["balance_group"] = [campus_keys.get(metadata.loc[i, "campus_id"],
                                  "other:" + metadata.loc[i, "institution_type"]) for i in sites.index]
        if not sites["balance_group"].isin(shares).all():
            raise ValueError("Eligible sites require a known enrolment calibration group")
        groups = list(shares)
        group_position = {g: i for i, g in enumerate(groups)}
        alternatives, site_pools, weights = [], [], []
        # A zone can contain several campus/type groups. Split its original OD
        # probability by eligible FTE to seed zone/group alternatives; later
        # establishment choice must stay inside the sampled alternative.
        for zone in sorted(z for z in pools if z in positions):
            candidates = sites.loc[pools[zone]]
            fte = candidates["number_employees_fte"].to_numpy(dtype=float)
            if not np.isfinite(fte).all() or (fte < 0).any():
                raise ValueError("Within-zone calibration weights require valid eligible-site FTE")
            fte = fte if fte.sum() > 0 else np.ones(len(fte))
            for group in sorted(candidates["balance_group"].unique()):
                selected = candidates["balance_group"].eq(group).to_numpy()
                if fte[selected].sum() == 0:
                    continue
                alternatives.append((zone, group))
                site_pools.append(candidates.index[selected].tolist())
                weights.append(fte[selected].sum() / fte.sum())
        alt_zones = np.array([positions[z] for z, g in alternatives], dtype=int)
        alt_groups = np.array([group_position[g] for z, g in alternatives], dtype=int)
        origins, counts, seeds = [], [], []
        for origin, group in students.groupby("home_zone_id", sort=True, dropna=False):
            if pd.isna(origin) or origin not in valid_origins:
                continue
            row = np.asarray(pdf[positions[origin]], dtype=float)
            if np.isnan(row).all():
                continue
            if not np.isfinite(row).all() or (row < 0).any():
                raise ValueError("Invalid OD probabilities in calibration")
            seed = row[alt_zones] * weights
            if seed.sum() <= 0:
                continue
            origins.append(origin)
            counts.append(len(group))
            seeds.append(seed / seed.sum())
        if not origins:
            for group, share in shares.items():
                audits.append(dict(study_level=level, calibration_group=group, requested_students=0.,
                                   expected_students=0., residual=0., reachable=False, converged=False,
                                   iterations=0, supported_persons=0, excluded_persons=len(students),
                                   integer_target_students=0,
                                   enrolment_share=share, status="no_supported_origins"))
            continue
        seeds = np.array(seeds).reshape(len(origins), len(alternatives))
        # Fit origin-to-group totals, rather than enforcing a second set of zone
        # totals. Within a group the relative OD zone probabilities remain fixed.
        group_seed = np.zeros((len(origins), len(groups)))
        for column, g in enumerate(alt_groups):
            group_seed[:, g] += seeds[:, column]
        # Unsupported origins retain their old compatible location. Only the
        # supported cohort contributes students to these calibration margins.
        # All residents contribute through the same municipality/quarter OD rows.
        # Accommodation flags neither reserve campus places nor alter OD support.
        requested = np.array(list(shares.values())) * sum(counts)
        flows, diagnostics = fit_flows(group_seed, counts, requested)
        # Round feasible fitted margins, retaining residuals for infeasible targets.
        integer_flows = integerise_flows(flows, counts)
        multipliers = np.divide(flows, group_seed,
                                out=np.zeros_like(group_seed), where=group_seed > 0)
        # Expand fitted group flows back to zone/group probabilities without
        # adding support where the original OD seed had a structural zero.
        probabilities = seeds * multipliers[:, alt_groups]
        probabilities /= np.asarray(counts)[:, None]
        models[level] = {"origins": dict(zip(origins, range(len(origins)))), "probabilities": probabilities,
                         "alternatives": alternatives, "site_pools": site_pools,
                         "groups": groups, "integer_flows": integer_flows}
        for j, group in enumerate(groups):
            audits.append(dict(study_level=level, calibration_group=group, requested_students=requested[j],
                               expected_students=flows[:, j].sum(), residual=diagnostics["residual"][j],
                               reachable=diagnostics["reachable"][j], converged=diagnostics["converged"],
                               iterations=diagnostics["iterations"], supported_persons=sum(counts),
                               excluded_persons=len(students) - sum(counts), enrolment_share=shares[group],
                               integer_target_students=int(integer_flows[:, j].sum()),
                               status="balanced" if diagnostics["converged"] else "unresolved_target_residual"))
    diagnostics = pd.DataFrame(audits)
    diagnostics["target_method"] = "provisional_bfs_share_and_eligible_fte_calibration"
    diagnostics["coverage_assumption"] = "Equal Swiss-residence shares across pooled tertiary institutions; higher-vocational coverage incomplete"
    diagnostics["reference_assumptions_verified"] = False
    return {"models": models, "diagnostics": diagnostics}


def execute(context):
    if not context.config("education_tertiary_balance"):
        return {"models": {}, "diagnostics": pd.DataFrame(columns=["study_level", "status", "converged"])}
    registry = context.stage("data.statent.education_institutions")
    rules = pd.read_csv(context.config("education_site_eligibility"), dtype={"statent_local_id": str})
    return build_balancing(context.stage("synthesis.population.sampled"), None,
                           context.stage("data.statent.statent"), registry, pooled_compatibility(registry, rules),
                           context.stage("data.spatial.zones"), context.stage("data.od.tertiary_education_matrix")[0],
                           context.stage("synthesis.population.spatial.primary.education.targets"))
