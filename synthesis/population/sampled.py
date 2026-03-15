import numpy as np
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.census.selected")
    
    context.config("input_downsampling")
    context.config("random_seed")


def execute(context):
    df = context.stage("data.census.selected")

    # If we do not want to downsample, set the value to 1.0 in config
    probability = context.config("input_downsampling")

    if probability < 1.0:
        logger.info("Downsampling (%f)", probability)

        if "household_id" in df.columns:

            household_ids = np.unique(df["household_id"])
            logger.info("  Initial number of households: %d", len(household_ids))
            logger.info("  Initial number of persons: %d", len(np.unique(df["person_id"])))

            # Set up RNG
            random = np.random.RandomState(context.config("random_seed"))
            
            # Perform sampling
            f = random.random_sample(size=(len(household_ids),)) < probability
            remaining_household_ids = household_ids[f]
            logger.info("  Sampled number of households: %d", len(remaining_household_ids))

            df = df[df["household_id"].isin(remaining_household_ids)]
            logger.info("  Sampled number of persons: %d", len(np.unique(df["person_id"])))

        else:

            person_ids = np.unique(df["person_id"])
            logger.info("  Initial number of persons: %d", len(person_ids))

            # Set up RNG
            random = np.random.RandomState(context.config("random_seed"))
            
            # Perform sampling
            f = random.random_sample(size=(len(person_ids),)) < probability
            remaining_person_ids = person_ids[f]
            logger.info("  Sampled number of persons: %d", len(remaining_person_ids))

            df = df[df["person_id"].isin(remaining_person_ids)]

            # Create household id to avoid issues later in the pipeline
            df["household_id"] = df["person_id"]

    return df
