import pandas as pd
import numpy as np
from tqdm import tqdm
import functools, itertools
import multiprocessing as mp

class HotDeckMatcher:
    def __init__(self, df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id):
        self.source_ids = df_source[source_id].values
        self.default_id = default_id

        if len(mandatory_fields) > len(np.unique(mandatory_fields)):
            raise RuntimeError("Duplicate mandatory fields")

        if len(preference_fields) > len(np.unique(preference_fields)):
            raise RuntimeError("Duplicate preference fields")

        self.all_fields = mandatory_fields + preference_fields

        self.categories = {}

        for field in self.all_fields:
            self.categories[field] = set(np.unique(df_source[field]))
            df_source["__hdm_%s" % field] = pd.Categorical(df_source[field], categories = self.categories[field])
            print("Found categories for %s:" % field, ", ".join([str(c) for c in self.categories[field]]))

        self.source_indices = {}

        for field in self.all_fields:
            self.source_indices[field] = {}

            for value in self.categories[field]:
                #self.source_indices[field][value] = set(np.where(df_source["__hdm_%s" % field] == value)[0])
                self.source_indices[field][value] = df_source["__hdm_%s" % field] == value

        self.source_weights = df_source[source_weight].values

        self.dimension_selectors = [[True]] * len(mandatory_fields)
        self.dimension_selectors += [[False, True]] * len(preference_fields)

        self.progress_total = 0

        for dimensions in itertools.product(*self.dimension_selectors):
            selected_fields = [self.all_fields[i] for i, select in enumerate(dimensions) if select]
            self.progress_total += np.product([len(self.categories[field]) for field in selected_fields])

        del df_source

    def __call__(self, df_target, chunk_index = 0):
        for field in self.all_fields:
            df_target["__hdm_%s" % field] = pd.Categorical(df_target[field], categories = self.categories[field])

        target_indices = {}

        for field in self.all_fields:
            target_indices[field] = {}

            for value in self.categories[field]:
                #target_indices[field][value] = set(np.where(df_target["__hdm_%s" % field] == value)[0])
                target_indices[field][value] = df_target["__hdm_%s" % field] == value

        matched_indices = (np.ones((len(df_target))) * -1).astype(np.int)

        with tqdm(total = self.progress_total, position = chunk_index) as progress:
            for dimensions in itertools.product(*self.dimension_selectors):
                selected_fields = [self.all_fields[i] for i, select in enumerate(dimensions) if select]

                for values in itertools.product(*[self.categories[field] for field in selected_fields]):
                    source_selected_indices = [self.source_indices[field][value] for value, field in zip(values, selected_fields)]
                    source_selected_indices = functools.reduce(np.logical_and, source_selected_indices)

                    target_selected_indices = [target_indices[field][value] for value, field in zip(values, selected_fields)]
                    target_selected_indices = functools.reduce(np.logical_and, target_selected_indices)

                    if np.any(source_selected_indices) and np.any(target_selected_indices):
                        weights = self.source_weights[source_selected_indices]
                        weights /= np.sum(weights)

                        target_count = np.count_nonzero(target_selected_indices)
                        selection_count = np.random.multinomial(target_count, weights)

                        indices = []
                        for index, count in zip(np.where(source_selected_indices)[0], selection_count):
                            indices += [index] * count
                        np.random.shuffle(indices)

                        matched_indices[target_selected_indices] = indices

                    progress.update()

        matched_ids = self.source_ids[matched_indices]
        matched_ids[matched_indices == -1] = self.default_id

        return matched_ids

def run(df_target, target_id, df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id = int(-1), runners = -1):
    matcher = HotDeckMatcher(df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id)

    if runners == -1:
        runners = mp.cpu_count()

    pool = mp.Pool(processes = runners, initializer = initializer, initargs = (matcher,))

    chunks = np.array_split(df_target, runners)
    df_target.loc[:, "hdm_source_id"] = np.hstack(pool.map(runner, enumerate(chunks)))

matcher = None
def initializer(_matcher):
    global matcher
    matcher = _matcher

def runner(args):
    index, df_chunk = args
    return matcher(df_chunk, index)

















#
