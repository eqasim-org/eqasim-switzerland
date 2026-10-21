"""Out-of-fold expert likelihoods and ensembles for probabilistic gate fitting."""

from pathlib import Path
import gc
import json
import pandas as pd
import torch

MODELS = ('regional_model', 'subregional_model', 'local_model', 'short_range_model')
PREFIX = 'synthesis.population.spatial.secondary_nn.'


def configure(context):
    context.config('secondary_nn_use_target_distance', True)
    if context.config('secondary_nn_use_target_distance'):
        return
    context.config('secondary_nn_crossfit_folds', 3)
    context.config('secondary_nn_selector_draws', 10)
    context.config('secondary_nn_distribution_scale_m', 1000.0)
    context.config('secondary_nn_short_trip_threshold_m', 1500.0)
    context.config('secondary_nn_short_query_radius_m', 200.0)
    context.config('secondary_nn_short_min_candidates', 6)
    context.config('random_seed')
    context.config('threads', volatile=True)
    for stage in ['data.microcensus.trips', 'data.microcensus.persons', 'data.constants',
                  'data.spatial.ovgk', 'synthesis.population.destinations',
                  'synthesis.population.spatial.secondary.detour_factors.factors',
                  PREFIX + 'mz_chains', PREFIX + 'h3', PREFIX + 'training_split']:
        context.stage(stage)
    for fold in range(int(context.config('secondary_nn_crossfit_folds'))):
        for model in MODELS:
            context.stage(PREFIX + model, config={'secondary_nn_crossfit_fold': fold},
                          alias=f'fold_{fold}_{model}')


def execute(context):

    # Predict each development person using experts that excluded their entire fold.

    if context.config('secondary_nn_use_target_distance'):
        return None
    from .short_trip_classifier import prepare_training_data, compare_experts
    from .model_wrappers import HierarchicalLocationChoiceModel
    torch.set_num_threads(max(1, int(context.config('threads'))))
    frame = prepare_training_data(context)
    pieces, audit = [], []
    split = context.stage(PREFIX + 'training_split')
    for fold in range(int(context.config('secondary_nn_crossfit_folds'))):
        held_out = frame.loc[frame.fold.eq(fold)]
        if held_out.empty:
            raise ValueError(f'Cross-fitting fold {fold} has no eligible trips')
        training_ids = set(split.loc[split.partition.ne('validation') & split.fold.ne(fold), 'person_id'])
        if not training_ids.isdisjoint(held_out.person_id):
            raise ValueError('Cross-fitting leaked held-out people into an expert')
        wrapper = HierarchicalLocationChoiceModel.build(FoldContext(context, fold), optimize=False, load_selector=False)
        with context.progress(label=f'Cross-fitted expert predictions {fold + 1}', total=len(held_out)) as progress:
            result = compare_experts(held_out, wrapper, draws=int(context.config('secondary_nn_selector_draws')),
                                     seed=int(context.config('random_seed')),
                                     scale_m=float(context.config('secondary_nn_distribution_scale_m')), progress=progress)
        result['prediction_fold'] = fold
        pieces.append(result)
        audit.append(dict(fold=fold, training_persons=len(training_ids),
                          held_out_persons=int(held_out.person_id.nunique()), trips=len(held_out),
                          person_overlap=0, excludes_outer_validation=True))
        del wrapper
        gc.collect()

    # Save immutable out-of-fold observations for selector fitting and auditing.

    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(['person_id', 'trip_id']).any():
        raise ValueError('Cross-fitted predictions contain duplicate trips')
    path = Path(context.path()) / 'expert_predictions.parquet'
    result.to_parquet(path, index=False)
    (Path(context.path()) / 'fold_audit.json').write_text(json.dumps(audit, indent=2))
    return (str(path),)


class FoldContext:
    def __init__(self, context, fold):
        self.context, self.fold = context, fold

    def stage(self, name):
        model = name.removeprefix(PREFIX)
        if name.startswith(PREFIX) and model in MODELS:
            return self.context.stage(f'fold_{self.fold}_{model}')
        return self.context.stage(name)

    def config(self, name):
        return self.context.config(name)
