# Prepared scenario tables and MATSim XML writers

The four preparation stages under `synthesis.prepared` assemble cached,
inspectable dataframes without writing XML. Existing stage names remain the
serialization entry points and keep returning their original `.xml.gz` paths:

| Preparation stage | Writer stage | Result |
| --- | --- | --- |
| `synthesis.prepared.facilities` | `matsim.scenario.facilities` | `facilities.xml.gz` |
| `synthesis.prepared.households` | `matsim.scenario.households` | `households.xml.gz` |
| `synthesis.prepared.population` | `matsim.scenario.population` | `population.xml.gz` |
| `synthesis.prepared.vehicles` | `matsim.scenario.vehicles` | `vehicles.xml.gz` |

The preparation stages now live under synthesis and contain no imports from
MATSim. The unused empty `ready` placeholders are replaced by `synthesis.prepared`. Writers read
only their prepared stage plus writer configuration (population compression).
All raw-data dependencies and scenario-population switches belong to preparation.
No full scenario build or simulation is required to test or inspect these stages.

## Return contracts

Frozen dataclasses in `synthesis/prepared/tables.py` name the returned
frames. The frames themselves are mutable pandas objects: consumers must treat
cached results as read-only. Producers copy frames before changing columns.
Shared population-field lists and the home-destination sentinel live in
`synthesis/prepared/schema.py`. MATSim attribute spellings stay in
`matsim/scenario/schema.py`; existing `population.HOME_DESTINATION_ID` imports
remain supported, and upstream stages import the generic sentinel directly.

- **Facilities:** `facilities`, with `facility_id`, integer `x`/`y`, and
  `activity_options` (tuple of purposes). Includes business sites, resident homes,
  optional cross-border homes/through endpoints and imported external facilities.
  Identical ID definitions are deduplicated; conflicting coordinates or activity
  definitions fail during preparation. IDs are compared in their serialized
  string representation.
- **Households:** `households` contains one globally sorted attribute row per
  household, including resolved income and the original car-count category; `members` contains
  sorted `household_id`/`person_id` pairs. Income-per-capita calculation and
  external/cross-border attribute conversion happen here, not in the writer. The
  bundle also records the upper car-count class; its plus notation is applied
  only by the MATSim household writer.
- **Population:** `persons`, `activities`, `vehicles`, and optional
  `freight_trips`/`freight_vehicles` and `lcv_trips`/`lcv_vehicles`. Optional frames
  are `None` when disabled. Residents have aligned departure/activity times,
  locations, following modes, subscription zones and residence flags. Existing
  removal of people with incomplete activity geometries is retained, including
  filtering their vehicles. Person/activity/owner order is resolved in preparation.
  Freight and LCV use a common named schema: `agent_id`, origin/destination
  coordinates, departure/arrival times, mode and day-end time. The existing
  one-hour freight trip assumption and 30-hour plan end are preserved.
- **Vehicles:** `types` holds physical type fields, optional velocity/flow
  efficiency and emission descriptor columns; `vehicles` holds vehicle IDs,
  type IDs, age and Euro emission class. XML attribute dictionaries are created
  by the MATSim writer, not stored in the prepared frames. Imported loop-mode aliases resolve to
  their base physical types. Undefined type references and duplicate IDs fail
  before XML writing.

Preparation implementation tokens include shared schema/contracts. Population
preparation also fingerprints the imported activity-time alignment helper.
Compression defaults remain on the population writer; preparation needs no
compression executable. Population export retains Python gzip and optional pigz.

## Serialization and intentional corrections

Household memberships and population activities/owned vehicles are streamed in
sorted order, rather than constructing complete XML trees in memory. Writers
perform XML formatting and escaping, not income computation, geometry joins or
population selection. Existing `PersonWriter` and preparation-helper imports are
kept where used by other modules.

Two existing writer issues are corrected during the split:

1. Facility purposes now follow flag **names**. The old positional mapping could
   turn leisure into other, shopping into leisure and other into shopping.
2. People with no owned vehicles receive an empty vehicle mapping. The old
   fallback passed a dataframe into a loop expecting vehicle records.

Global household ordering and ID checks also prevent duplicate XML definitions
from being silently emitted. These are validations of prepared data, not changes
to demand generation or matching behaviour.

## Validation

Local fixtures compare parsed XML against the original writers for domestic,
external and cross-border demand and freight/LCV. They check that preparation
writes no XML and that neither preparation nor serialization mutates upstream
frames. Separate tests cover named facility purposes, ID conflicts, undefined
vehicle types, and non-travellers without owned vehicles. Tests and before-change
snapshots remain in the ignored local `tests/` directory.

Inspect conditional dependency graphs with the working configuration before
running these stages. Do not replace `config_dib_10.yml` or launch a full MATSim
simulation to validate this export-only split.
