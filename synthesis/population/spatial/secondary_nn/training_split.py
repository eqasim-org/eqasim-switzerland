"""Person-separated cross-fitting with an untouched outer validation set."""

import numpy as np
import pandas as pd


def configure(context):
    context.stage('data.microcensus.persons')
    context.config('random_seed')
    context.config('secondary_nn_crossfit_folds', 3)
    context.config('secondary_nn_selector_validation_fraction', .2)


def execute(context):

    # Reserve outer validation people before assigning cross-fitting folds.

    persons = context.stage('data.microcensus.persons')
    ids = np.sort(persons.loc[~persons.weekend, 'person_id'].unique())
    folds = int(context.config('secondary_nn_crossfit_folds'))
    fraction = float(context.config('secondary_nn_selector_validation_fraction'))
    if folds < 2 or not 0 < fraction < .5 or len(ids) < max(20, 4 * folds):
        raise ValueError('Cross-fitting requires >=2 folds, enough people and a validation fraction in (0, .5)')
    rng = np.random.default_rng(int(context.config('random_seed')))
    ids = rng.permutation(ids)
    count = max(2, int(np.ceil(len(ids) * fraction)))
    partition = np.full(len(ids), 'train', dtype=object)
    partition[:count] = 'validation'
    stopping = max(2, int(np.ceil((len(ids) - count) * .2)))
    partition[count:count + stopping] = 'stopping'
    fold = np.full(len(ids), -1, dtype=int)
    fold[count:] = np.arange(len(ids) - count) % folds
    return pd.DataFrame(dict(person_id=ids, partition=partition, fold=fold))


def expert_persons(context, frame):
    if context.config('secondary_nn_use_target_distance'):
        return frame.copy()
    split = context.stage('synthesis.population.spatial.secondary_nn.training_split')
    fold = int(context.config('secondary_nn_crossfit_fold'))
    eligible = split.partition.ne('validation')
    if fold >= 0:
        if fold not in set(split.fold):
            raise ValueError('Unknown cross-fitting fold')
        eligible &= split.fold.ne(fold)
    return frame.loc[frame.person_id.isin(split.loc[eligible, 'person_id'])].copy()
