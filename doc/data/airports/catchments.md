# Airport municipality catchments with distance-ratio overlap

`data.airports.catchments` builds the municipality pools for Geneva (`GVA`) and
Zurich (`ZRH`). It always permits the nearest airport and permits the farther
one when its Euclidean distance is at most **1.3 times** the nearer distance.

```text
farther distance / nearer distance <= 1.3  → both airports eligible
farther distance / nearer distance > 1.3   → nearest airport only
```

The boundary is inclusive. Exact distance ties permit both airports. If a
municipal reference point is exactly at one airport, only that airport is
eligible, unless both airport coordinates coincide there.

Configure the threshold with:

```yaml
config:
  airport_catchment_distance_ratio: 1.3 # Default; must be finite and at least 1.
```

This replaces the earlier exclusive-nearest rule. There is no additional
100 km rule. The overlap allows places with similar distances to serve both
airports while excluding the much farther airport near either airport's core.
Eligibility for both does **not** imply a 50/50 split or generate extra demand.
Each airport still distributes its existing passenger total using gravity weights.

## Returned municipality table

The stage returns one row per gravity-model municipality:

| Column | Meaning |
| --- | --- |
| `municipality_id`, `municipality_name`, `canton_id` | Municipality and canton identifiers |
| `x`, `y` | Municipal reference point in EPSG:2056 |
| `distance_gva_km`, `distance_zrh_km` | Euclidean distance to each airport |
| `nearest_airport` | Nearest airport; reporting only, GVA breaks exact ties |
| `distance_ratio` | Farther/nearer distance; infinite when only the nearer distance is zero |
| `eligible_gva`, `eligible_zrh` | Candidate-pool membership used by the corresponding OD stage |
| `is_shared` | True when both airports are eligible |

`data.airports.geneva.model` supplies the municipal reference points and airport
coordinates so catchments and gravity use exactly the same distances. Municipal
reference points use the mean census home location when it lies inside the
polygon, otherwise an interior polygon point. Municipalities with neither
residents nor accommodation are excluded from the gravity support.

## Allocation and totals

1. Load the saved gravity coefficients and current municipal features.
2. Compute the distances and each airport's eligibility mask.
3. Intersect each observed Swiss region/profile/direction row with its airport's
   eligible municipality pool.
4. Normalize the remaining gravity weights to distribute that row's existing
   passenger total; municipalities outside the pool have zero weight.
5. Scale the combined ODs once and generate trips using the existing stages.

Geneva's observed regional constraints and Zurich's Swiss total remain fixed.
Fractional and integer regional counts are preserved. Foreign regional rows
remain unchanged and excluded from Swiss trip synthesis. A required region
without candidate support causes an explicit allocation error rather than
silently dropping or moving passengers.

The rule applies before resident/visitor assignment and affects both profiles.
It does not change mode evidence, physical airport access points or saved
gravity coefficients. Existing resident/accommodation capacity checks remain.

## Caching and assumptions

This is a normal Synpp stage. Its signature covers coordinates, distances,
eligibility, the ratio setting and catchment code. Both airport OD signatures
include it, invalidating downstream saved demand when the rule changes. OD
summaries record the threshold, candidate counts and shared-municipality count.
No gravity coefficient re-estimation is required.

The threshold is a modelling assumption, not an estimated airport-choice
parameter. Eligibility uses one point per municipality, not each home or hotel.
Only GVA and ZRH compete; other airports are outside this model.
