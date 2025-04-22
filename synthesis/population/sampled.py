import numpy as np


def configure(context):
    context.stage("data.census.selected")
    
    context.config("input_downsampling")
    context.config("random_seed")


def execute(context):
    df = context.stage("data.census.selected")

    # If we do not want to downsample, set the value to 1.0 in config
    probability = context.config("input_downsampling")

    if probability < 1.0:
        print("Downsampling (%f)" % probability)

        if "household_id" in df.columns:

            household_ids = np.unique(df["household_id"])
            print("  Initial number of households:", len(household_ids))
            print("  Initial number of persons:", len(np.unique(df["person_id"])))

            # Set up RNG
            random = np.random.RandomState(context.config("random_seed"))
            
            # Perform sampling
            f = random.random_sample(size=(len(household_ids),)) < probability
            remaining_household_ids = household_ids[f]
            print("  Sampled number of households:", len(remaining_household_ids))

            df = df[df["household_id"].isin(remaining_household_ids)]
            print("  Sampled number of persons:", len(np.unique(df["person_id"])))

        else:

            person_ids = np.unique(df["person_id"])
            print("  Initial number of persons:", len(person_ids))

            # Set up RNG
            random = np.random.RandomState(context.config("random_seed"))
            
            # Perform sampling
            f = random.random_sample(size=(len(person_ids),)) < probability
            remaining_person_ids = person_ids[f]
            print(f"  Sampled number of persons: {len(remaining_person_ids)}")

            df = df[df["person_id"].isin(remaining_person_ids)]

            # Create household id to avoid issues later in the pipeline
            df["household_id"] = df["person_id"]


    return df
