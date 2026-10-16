# Neural secondary-location assignment

This stage assigns secondary activities in the existing synthetic population.
It reads `synthesis.population.trips` and `synthesis.population.enriched`, attaches
households and primary locations from the same sampled population and spatial
stages as `secondary`, and selects facilities from
`synthesis.population.destinations`. This includes French secondary destinations
when the configured outbound-flow stages include them. Rows without any secondary
offer (such as border facilities and remote-work-only locations) are excluded.
Microcensus persons and trips supply training examples and matched trip features.
The output remains `(locations, convergence)`, with the secondary activity index
in `locations.trip_index`, as required by the downstream spatial-location stage.

Configure the pipeline in `config_dib_10.yml`:

```yaml
config:
    locations_algorithm: nn
    secondary_nn_use_target_distance: true
```

`secondary_nn_use_target_distance` defaults to `true`. Set it to `false` to remove
`target_distance` from the input vector of the regional, subregional, local, and
short-range models. The feature layout is saved with the person scaler and used
at inference, including standalone wrapper predictions. Changing the setting
requires retraining all models. With `true`, target distance selects between the
short-range model and the regional → subregional → local hierarchy. With `false`,
the separate `synthesis.population.spatial.secondary_nn.short_trip_classifier`
stage is scheduled and its binary prediction selects the path. This classifier
never takes target distance as an input. Its features use the same demographic,
daily-chain, timing, progression, and purpose vocabulary as the choice models.
It has its own saved scalers, fitted only on its training persons.

The classifier uses L2-regularized logistic regression, a fixed split by person,
and no hyperparameter search on the held-out validation set. Validation persons
are excluded from both scaler fitting and coefficient fitting, and the saved
model is not refitted on them. `validation.json` in the classifier stage directory
reports training/validation log loss, constant-probability baseline log loss,
validation balanced accuracy, and ROC AUC. These metrics allow checking for poor
generalization; they do not guarantee the absence of overfitting on real data.
The optional settings `secondary_nn_short_classifier_C` (default `0.1`, smaller
means stronger regularization) and
`secondary_nn_short_classifier_validation_fraction` (default `0.2`) control its
regularization and held-out person fraction. `random_seed` controls the split
and preprocessing. Classifier training requires both short and long trips in
each split; it fails explicitly if that is impossible.

Target distances still supply the classifier labels and distance-loss training
targets, identify stationary trips, and weight the final facility draw. The other
distance features remain available to the models.

`secondary_nn_short_trip_threshold_m` (default 1500 metres) controls both the
short-range training cutoff, classifier labels, and distance-based selection when
`secondary_nn_use_target_distance` is `true`.

The H3 stage indexes coordinates independently at resolutions 5, 7, and 10.
A root ID is a resolution-5 cell; middle and leaf IDs are slash-separated paths
of two and three cells. A finer H3 cell crossing a coarse boundary has a separate
node in each occupied branch. Node polygons are intersected in EPSG:2056 with all
ancestors so child geometries stay inside parent geometries. Representative
points stay inside those polygons. Feature aggregation, training labels,
candidate lists, destination pools, sibling fallbacks, and short-range parent
lookup all use these paths and the shared tree; no H3 logical ancestry is used.
Existing models trained with the old hierarchy must be retrained. The default
`overwrite_*_model_if_exists` settings already enable this; incompatible saved
candidate trees are rejected before assignment.

Run focused regression tests from the repository root:

```sh
uv run -m pytest tests/test_secondary_nn.py tests/test_secondary_nn_short_trip_classifier.py -q
```

Run the configured pipeline with `uv run -m synpp config_dib_10.yml`.
