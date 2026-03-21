import numpy as np
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.census.selected")
    
    context.config("input_downsampling")
    context.config("random_seed")


def execute(context):
    df = context.stage("data.census.selected")
    probability = context.config("input_downsampling")
    aggregation_col = "home_zone_id" if "home_zone_id" in df.columns else "home_municipality_id"
    sampling_col = "household_id" if "household_id" in df.columns else "person_id"

    if probability < 1.0:
        logger.info("Downsampling (%f)", probability)
        random = np.random.RandomState(context.config("random_seed"))

        df_hh = df.drop_duplicates(sampling_col)[[sampling_col, aggregation_col]]
        logger.info("  Initial unique %s: %d, persons: %d", sampling_col, df_hh.shape[0], len(df["person_id"].unique()))

        def sample_stratum(g):
            idx = random.choice(len(g), size=max(1, round(len(g) * probability)), replace=False)
            return g.iloc[idx]

        kept_ids = df_hh.groupby(aggregation_col, group_keys=False).apply(sample_stratum)[sampling_col].values
        df = df[df[sampling_col].isin(kept_ids)]
        logger.info("  Sampled %s: %d, persons: %d", sampling_col, len(kept_ids), len(df["person_id"].unique()))
        logger.info("Proportion of original population: %f", len(df) / len(context.stage("data.census.selected")))


    return df
