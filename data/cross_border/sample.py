import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("synpp")
SAMPLING_COLUMNS = ["interview_place", "label", "trip_mode"]


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("random_seed")

    context.stage("data.cross_border.generate_od")


def execute(context):

    # Load the expanded population; survey weights have already been applied.

    df = context.stage("data.cross_border.generate_od")
    probability = context.config("input_downsampling")

    # Preserve local driver/person demand through downsampling. Keep every row
    # belonging to a selected person, without changing its identifiers or order.

    person_ids, audit = sample_person_ids(df, probability, context.config("random_seed"))
    result = df[df["cross_border_person_id"].isin(person_ids)]
    logger.info("Cross-border sample: %d of %d persons at fraction %s",
                len(person_ids), audit["full_persons"].sum(), probability)

    # Expose integer rounding separately from survey calibration and routing.

    audit.to_csv(Path(context.path()) / "sampling_counts.csv", index=False)

    # Return the existing population table contract.

    return result


def sample_person_ids(df, probability, random_seed):
    """Allocate a fixed-size stratified sample, then draw people without replacement."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Cross-border input_downsampling must be between zero and one")

    # A person must belong to one stratum even if a producer supplies several
    # rows for that person. Sorting IDs makes the draw independent of row order.
    persons = df[["cross_border_person_id"] + SAMPLING_COLUMNS].drop_duplicates()
    if persons["cross_border_person_id"].isna().any() or persons["cross_border_person_id"].duplicated().any():
        raise ValueError("Cross-border persons require unique IDs and consistent sampling categories")
    persons = persons.sort_values("cross_border_person_id")
    groups = persons.groupby(SAMPLING_COLUMNS, sort=True, dropna=False, observed=True)
    audit = groups.size().rename("full_persons").reset_index()
    expected = audit["full_persons"].to_numpy() * probability
    allocated = np.floor(expected).astype(int)
    random = np.random.RandomState(random_seed)

    # Largest remainders reconcile fractional strata with the nearest integer
    # national sample size. Seeded tie-breaking avoids favouring group names.
    # Each stratum receives floor(p*N) or ceil(p*N), never an arbitrary minimum.
    remaining = int(np.rint(len(persons) * probability)) - int(allocated.sum())
    order = np.lexsort((random.random_sample(len(audit)), -(expected - allocated)))
    allocated[order[:remaining]] += 1

    # Separate modes retain vehicle counts for drivers and person counts for
    # passengers/PT. Visitor and transit exits are still assigned downstream.
    selected = []
    for (_, group), count in zip(groups, allocated):
        ids = group["cross_border_person_id"].to_numpy()
        if count == len(ids):
            selected.extend(ids)
        elif count:
            selected.extend(random.choice(ids, size=count, replace=False))

    audit["expected_sample_persons"] = expected
    audit["sampled_persons"] = allocated
    audit["expanded_sample_persons"] = allocated / probability if probability > 0 else np.nan
    return np.asarray(selected, dtype=persons["cross_border_person_id"].to_numpy().dtype), audit
