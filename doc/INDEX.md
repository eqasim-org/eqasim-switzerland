# Documentation

Commands in these guides run from the repository root unless stated otherwise.
Module documentation mirrors the source-package paths below this directory.
The root [AGENTS.md](../AGENTS.md) contains guidance for coding agents.

## Pipeline and inputs

- [Overview, setup and input datasets](../README.md)
- [Beginner configuration template](../config.yml)
- [Public-transport pricing inputs](PT_pricing_inputs_and_explanations.md)

## Implementation guides

- [Cross-border activity timing](data/cross_border/README.md)
- [Shared MATSim configuration](matsim/config/README.md)
- [SUMO overview and stage flowcharts](sumo/WORKFLOW.md)
- [SUMO runtime and road network](sumo/README.md)
- [SUMO population and plans](sumo/population/README.md)
- [SUMO demand upsampling and portable export](sumo/scaling.md)
- [SUMO Polars caches and native XML](sumo/native/README.md)
- [SUMO simulation presets and loaded assignment](sumo/config/README.md)
- [SUMO network loading and regional cutter](sumo/cutter/README.md)
- [Native network reader and binary cache](matsim/native/README.md)
- [Population mode choice](mode_choice/README.md)
- [Experimental FLASH-TB routing and benchmarks](mode_choice/routing/flash/README.md)
- [Java / FLASH population PT router comparison](analysis/pt_routers/README.md)
- [Microcensus diary matching: learned V2 inputs and masks](synthesis/population/matching/README.md)
- [Matching predictive validation and embedding analysis](analysis/matching/README.md)
- [Secondary-location neural models](synthesis/population/spatial/secondary_nn/README.md)
- [Traffic-count matching](analysis/counts/matching/README.md)
- [Road-user map and activity-purpose shares](analysis/counts/road_users/README.md)
- [Cross-border survey entries and through-trip exits](data/cross_border/README.md)
- [French population and trip explorer](analysis/french_activities.md)
- [Secondary-location behavior comparison](<analysis/secondary locations/README.md>)

- [SUMO performance and stage caches](sumo/performance.md)

## Changelogs

- [Main changelog](CHANGELOG.md)
- [Aurore's changelog](CHANGELOG_aurore.md)
- [Dib's changelog](CHANGELOG_dib.md)

Tests remain in the local `tests/` directory and are intentionally excluded from Git.

