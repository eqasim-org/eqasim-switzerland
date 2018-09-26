import pandas as pd
import numpy as np
from tqdm import tqdm
import functools, itertools
import multiprocessing as mp

class HotDeckMatcher:
    def __init__(self, df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id):
        self.df_source = df_source
        self.default_id = default_id
        self.mandatory_fields = mandatory_fields
        self.preference_fields = preference_fields
        self.all_fields = mandatory_fields + preference_fields
        self.source_weight = source_weight
        self.source_id = source_id

        self.categories = {}

        for field in self.all_fields:
            self.categories[field] = set(np.unique(df_source[field]))
            self.df_source["__hdm_%s" % field] = pd.Categorical(self.df_source[field], categories = self.categories[field])
            print("Found categories for %s:" % field, ", ".join([str(c) for c in self.categories[field]]))

        self.selectors = {}

        for field in self.all_fields:
            self.selectors[field] = {}

            for value in self.categories[field]:
                self.selectors[field][value] = df_source["__hdm_%s" % field] == value

    def __call__(self, df_target, chunk_index = 0):
        for field in self.all_fields:
            df_target["__hdm_%s" % field] = pd.Categorical(df_target[field], categories = self.categories[field])

        matched_ids = np.ones((len(df_target), )) * self.default_id

        cache = {}

        for i, (_, row) in tqdm(enumerate(df_target.iterrows()), total = len(df_target), position = chunk_index):
            selector = np.ones((len(self.df_source),)).astype(np.bool)

            mandatory_values = [row["__hdm_%s" % field] for field in self.mandatory_fields]
            preference_values = [row["__hdm_%s" % field] for field in self.preference_fields]

            cache_key = tuple(mandatory_values + preference_values)
            selector = cache[cache_key] if cache_key in cache else None

            if selector is None:
                selector = np.ones((len(self.df_source), )).astype(np.bool)

                for value, field in zip(mandatory_values, self.mandatory_fields):
                    selector &= self.selectors[field][value]

                for value, field in zip(preference_values, self.preference_fields):
                    local_selector = self.selectors[field][value]

                    if np.sum(local_selector) > 0:
                        selector &= local_selector

                cache[cache_key] = selector

            if sum(selector) > 0:
                df_selection = self.df_source[selector]
                weights = df_selection[self.source_weight].values

                cdf = np.cumsum(weights)
                cdf /= cdf[-1]

                observation_selector = np.random.random()
                selected_index = np.sum(observation_selector > cdf)

                matched_ids[i] = df_selection[self.source_id].values[selected_index]

        return matched_ids

def run(df_target, target_id, df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id = int(-1)):
    matcher = HotDeckMatcher(df_source, source_id, source_weight, mandatory_fields, preference_fields, default_id)
    pool = mp.Pool(processes = 4, initializer = initializer, initargs = (matcher,))

    chunks = np.array_split(df_target, 4)
    df_target.loc[:, "hdm_source_id"] = np.hstack(pool.map(runner, enumerate(chunks)))

matcher = None
def initializer(_matcher):
    global matcher
    matcher = _matcher

def runner(args):
    index, df_chunk = args
    return matcher(df_chunk, index)

















#
