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

- [Shared MATSim configuration](matsim/config/README.md)
- [SUMO overview and stage flowcharts](sumo/WORKFLOW.md)
- [SUMO runtime and road network](sumo/README.md)
- [SUMO population and plans](sumo/population/README.md)
- [SUMO simulation presets and loaded assignment](sumo/config/README.md)
- [SUMO network loading and regional cutter](sumo/cutter/README.md)
- [Native network reader and binary cache](matsim/native/README.md)
- [Population mode choice](mode_choice/README.md)
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
