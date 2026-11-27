# How we create airport demand

This is the short explanation of the Geneva and Zurich airport pipeline.
The [airport reference guide](README.md) lists source files, settings and refresh
procedures. Airport demand is enabled by default with `use_airports: true`.
Set it to false to disable scenario integration.

## 1. Read the airport evidence

We use reviewed local copies of the source figures; normal pipeline runs do
not download new reports or refresh flight boards.

| Source | What we use it for |
| --- | --- |
| Geneva airport's supplied 2025 accessibility survey | Swiss catchment groups, resident/visitor mixture and access-mode shares |
| Geneva annual report and monthly statistics | Passenger totals, ground-access accounting, arrival/departure split and monthly seasonality |
| Zurich published 2025 traffic statistics | Local passenger totals and arrival/departure split |
| Zurich historical catchment benchmark | Assumed Swiss share of ground endpoints: 88% |
| Historical Geneva/OFAC–Intraplan report | Additional canton observations for the geographic model |
| Archived flight-board snapshots | Time-of-day distributions for each airport and direction |
| STATPOP, STATENT and municipality polygons | Municipal population, hotels/dormitories and geography |
| Microcensus 2015 and 2021 airport observations | Coarse age, sex and employment preferences when choosing residents |
| OVGK accessibility polygons | Whether a sampled home or accommodation supports PT |

Evidence files and their source links live under
`<data_path>/airports/geneva/` and `airports/zurich/`, including
`data_from_websites/`. See [Geneva sources](geneva.md) and
[Zurich sources](zurich.md) for the report pages, accounting assumptions and URLs.
The PDFs are input data, not files bundled in the Python package.

## 2. Calculate ground-access passenger movements

Airport passenger statistics include travellers who connect between flights
without leaving the airport. We remove the applicable airside/transit counts
before calculating ground demand. Geneva's treatment of connecting passengers
has an explicit configurable counting assumption, documented in its source guide.

We then obtain movements **to** and **from** the airport using the published
arrival/departure counts. These are passenger movements, not vehicle counts.
We do not multiply them by two or assume that every passenger returns that day.

By default we use an annual-average day. Optional monthly inputs change the
daily volume; choosing a weekday flight-time distribution alone does not
convert that volume into measured weekday demand.

## 3. Restrict demand to Swiss ground endpoints

We model only travel between the airport and a location inside Switzerland.
French ground access to Geneva and German ground access to Zurich are excluded.
Foreign counts remain available in the source accounting.

For Geneva, the supplied survey determines the Swiss share and observed groups:
Genève, Vaud, Valais and the rest of Switzerland. For Zurich, we apply the
historical 88% Swiss benchmark to its current local passenger traffic. This
Zurich share is an assumption transferred across years.

## 4. Assign catchments and allocate Swiss totals to municipalities

`data.airports.geneva.model` uses municipal population, accommodation establishment
counts and Euclidean distance to the airport. Nearby municipalities receive
more weight through distance decay; larger population/accommodation counts
increase their weight.

The historical Geneva observations support an initial four-parameter fit.
Three parameters are then updated against the latest Geneva regional shares.
Saved parameters live in `<data_path>/airports/models/parameters.json`; they are
re-estimated only when `airport_model_overwrite: true` is explicitly requested.

`data.airports.catchments` always permits the nearest airport and also permits
the farther airport when its distance is at most 1.3 times the nearer distance.
Each airport distributes passengers only within its eligible pool. Municipalities
in the overlap can serve both, with gravity weights rather than a fixed 50/50 split.
See [ratio-based catchments](catchments.md).

`data.airports.geneva.ods` preserves the latest observed Geneva regional totals
and distributes them among municipalities within each group.
`data.airports.zurich.ods` transfers the Geneva model to Zurich and distributes
its Swiss total over its assigned municipalities.

**The model distributes the published passenger demand; it does not predict a
new airport total.** Zurich's municipal distribution is modelled, rather than
observed. See [model details](model.md).

## 5. Scale once to the simulated population fraction

`data.airports.ods` combines both airports' municipal ODs, retains Swiss
endpoints and applies `input_downsampling` once. Deterministic integer rounding
preserves the allocated totals. Trip generation does not apply another sample.

For example, the current annual-average inputs give approximately 29,975 Swiss
movements/day at Geneva and 55,997 at Zurich before sampling. At a 10% sample,
the rounded totals are 2,998 and 5,600: **8,598 movements altogether**.
These figures describe the current inputs, not hard-coded future targets.

## 6. Decide which movements belong to residents and visitors

The reconciled Geneva Swiss-endpoint survey mixture is approximately
**57.56% residents and 42.44% visitors**. We transfer this mixture to Zurich
because an equivalent Zurich breakdown is unavailable.

Municipal movement counts are already fixed. Within those counts, visitors are
allocated using the number of hotels and dormitories as weights, including
student residences. Accommodation-rich municipalities receive more visitors,
subject to their movement capacity. Municipalities without usable accommodation
contain residents only.

`is_visitor` describes the resident/accommodation profile, **not nationality**.
Foreign nationals living in Switzerland can be residents in this model.

## 7. Select people and accommodation

`synthesis.airports.locations` prepares the location pools.
`synthesis.airports.trips` selects residents from `synthesis.population.enriched`,
after matching, using actual plans from `synthesis.population.activities`. The
preferred pool has a home activity and at most one non-home activity (for example
`home`, `home-other-home` or `home-leisure-home`). Count visits, not distinct
purpose types. `airport_resident_max_nonhome_activities` changes this limit.
Selection stays inside the assigned municipality and uses
smoothed pooled microcensus age, sex and employment preferences. Each resident
is selected at most once across both airports and directions. Their real person
ID, household ID and home coordinates are retained.

If selected homes cannot support the PT target, first try unused PT-accessible
people from the preferred pool in that municipality, A/B/C before D. Only a
remaining PT shortage permits selecting a more active resident. Those extensions
must use PT. Road-demand shortages do not trigger a general fallback to active
people. `resident_nonhome_activities` and `resident_selection_pool` audit this.

Visitors receive a sampled hotel or dormitory location in the assigned
municipality. Establishments have equal weight; room counts are unavailable.
Missing resident or accommodation capacity is reported as unresolved demand
rather than silently moving trips to another municipality.

## 8. Assign modes while respecting PT availability

Both airports use the Geneva survey's Swiss-endpoint mode targets:

| Detailed mode | Share |
| --- | ---: |
| Drop-off/pick-up | 20% |
| Private car | 10% |
| Rail | 28% |
| Bus/tram/Aérobus | 23% |
| Taxi | 6% |
| Ride-hail | 6% |
| Other | 7% |

Simple linear distance preferences fitted to Geneva's three Swiss regional
mode observations guide where these modes are allocated. They distribute the
fixed airport-wide targets; they do not change municipal movement counts.

PT first uses OVGK A/B/C locations. If these cannot meet its target, it uses D.
Any remaining PT shortfall transfers to car/car-passenger categories and is
reported. For residents, eligibility concerns the already selected person's
home; visitor PT locations are sampled from eligible accommodation.

Detailed modes remain in the output. For simulation, rail and bus become `pt`;
private car, drop-off, taxi and ride-hail become `car`; `other` becomes
`car_passenger` as an explicit assumption. No unsupported walk/bike share is added.

## 9. Choose coordinates and departure times

PT trips use the airport station point; road trips use its parking/drop-off
point. The other endpoint is the selected resident's home or visitor's
accommodation.

Flight snapshots provide scheduled flight times by airport, direction and day.
Named days and averaged `weekday`/`weekend` groups are supported; incomplete
snapshot coverage is reported.

For airport departures, we subtract a sampled check-in lead time and estimated
ground travel time from the flight time. For arrivals, we add a sampled exit
clearance time before starting the ground trip.

Ground travel time uses Euclidean distance multiplied by the default detour
factor 1.35 and divided by mode speed: all road categories use 60 km/h, rail
70 km/h and bus 30 km/h. These are configurable approximations, not routed times.
Times before midnight or after the modelled day retain their offsets.

## 10. Inject the plans into the scenario

`synthesis.prepared.population` calls
`inject_airport_trips_in_population` from `synthesis.airports.trips`.
Selected residents' ordinary plans are removed and replaced with airport plans
under the **same IDs**. Visitors become additional agents with collision-checked
IDs. Each plan contains one ground leg:

- Residents: `home → airport` or `airport → home`.
- Visitors: `home_secondary → airport` or `airport → home_secondary`.

Prepared vehicle and facility stages supply the corresponding resources.
MATSim writes the airport plans alongside freight and LCV. Airport agents use
a separate subpopulation with fixed modes and departure times; route replanning
is allowed. Ordinary matching and location-assignment stages remain unchanged.
See [integration details](population_integration.md).

## 11. Reuse results and inspect assumptions

Synpp caches stage results. Signed OD/trip exports also check their inputs,
settings and artifact contents before reusing saved results. Changing inputs
regenerates affected demand, while frozen model parameters remain unchanged
unless overwrite is requested.

Inspect the summary files for unresolved demand, source provenance, PT
fallbacks and timing coverage. The [interactive trip map](../../analysis/airports/trips_html.md)
shows both airports and lets you filter directions, modes and traveller profiles.

The main limitations are the transferred Zurich shares/model, coarse regional
mode evidence, incomplete flight snapshots and approximate ground travel times.
We do not generate airport employee commuting, connecting flights, escorts'
return legs or empty taxi movements. Those would require additional demand
assumptions.
