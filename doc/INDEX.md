# Documentation

Commands in these guides run from the repository root unless stated otherwise.
Module documentation mirrors the source-package paths below this directory.
The root [AGENTS.md](../AGENTS.md) contains guidance for coding agents.

## Pipeline and inputs

- [Overview, setup and input datasets](../README.md)
- [Beginner configuration template](../config.yml)
- [Public-transport pricing inputs](PT_pricing_inputs_and_explanations.md)

- [Education reference maintenance](analysis/education_reference_update.md)
- [Education model and configuration (LaTeX)](synthesis/population/education_model.tex)

## Implementation guides

- [Cross-border activity timing](data/cross_border/README.md)
- [Shared MATSim configuration](matsim/config/README.md)
- [SUMO overview and stage flowcharts](sumo/WORKFLOW.md)
- [SUMO runtime and road network](sumo/README.md)
- [SUMO population and plans](sumo/population/README.md)
- [SUMO Polars caches and native XML](sumo/native/README.md)
- [SUMO simulation presets and loaded assignment](sumo/config/README.md)
- [SUMO network loading and regional cutter](sumo/cutter/README.md)
- [Native network reader and binary cache](matsim/native/README.md)
- [Population mode choice](mode_choice/README.md)
- [Microcensus diary matching: learned V2 inputs and masks](synthesis/population/matching/README.md)
- [Matching predictive validation and embedding analysis](analysis/matching/README.md)
- [Secondary-location neural models](synthesis/population/spatial/secondary_nn/README.md)
- [Traffic-count matching](analysis/counts/matching/README.md)
- [Road-user map and activity-purpose shares](analysis/counts/road_users/README.md)
- [Cross-border survey entries and through-trip exits](data/cross_border/README.md)
- [Secondary-location behavior comparison](<analysis/secondary locations/README.md>)

## Changelogs

- [Main changelog](CHANGELOG.md)
- [Aurore's changelog](CHANGELOG_aurore.md)
- [Dib's changelog](CHANGELOG_dib.md)

Tests remain in the local `tests/` directory and are intentionally excluded from Git.

## Education model and diagnostics

- [Education model and configuration (LaTeX)](synthesis/population/education_model.tex)
- [Education audit](analysis/education_audit.md)
- [Education campus weights](analysis/education_campus_weights.md)
- [Education classification](analysis/education_classification.md)
- [Education distances](analysis/education_distances.md)
- [Education doctoral scope](analysis/education_doctoral_scope.md)
- [Education enrolment](analysis/education_enrolment.md)
- [Education institution matching](analysis/education_institution_matching.md)
- [Education institutions](analysis/education_institutions.md)
- [Education location eligibility](analysis/education_location_eligibility.md)
- [Education student stock](analysis/education_student_stock.md)
- [Non private residents](analysis/non_private_residents.md)
- [Secondary education assignment](analysis/secondary_education_assignment.md)
- [Tertiary education age distance](analysis/tertiary_education_age_distance.md)
- [Tertiary education assignment](analysis/tertiary_education_assignment.md)
- [Tertiary education distance regularisation](analysis/tertiary_education_distance_regularisation.md)
- [Tertiary education od](analysis/tertiary_education_od.md)
- [Tertiary education od smoothing](analysis/tertiary_education_od_smoothing.md)
- [GWR residence reference](data/gwr/README.md)
- [Commute](data/microcensus/commute.md)
- [Secondary trips](data/microcensus/secondary_trips.md)
- [Households](data/statpop/households.md)
- [Secondary](synthesis/population/spatial/primary/education/secondary.md)
- [Secondary distances](synthesis/population/spatial/primary/education/secondary_distances.md)
