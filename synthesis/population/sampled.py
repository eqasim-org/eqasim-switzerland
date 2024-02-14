import numpy as np


def configure(context):
    context.config("input_downsampling")
    context.stage("data.statpop.scaled")
    context.config("output_path")
    context.config("random_seed")


def execute(context):
    df = context.stage("data.statpop.scaled")

    # If we do not want to downsample, set the value to 1.0 in config
    probability = context.config("input_downsampling")

    if probability < 1.0:
        print("Downsampling (%f)" % probability)

        household_ids = np.unique(df["household_id"])
        print("  Initial number of households:", len(household_ids))
        print("  Initial number of persons:", len(np.unique(df["person_id"])))

        # TODO: specify random seed
        # during downsampling, households are selected randomly without specifying a seed,
        # which means that running the pipeline twice will produce different populations
        # resulting in potentially different simulation results
        # Set up RNG
        random = np.random.RandomState(context.config("random_seed"))
        
        # Perform sampling
        f = random.random_sample(size=(len(household_ids),)) < probability

        remaining_household_ids = household_ids[f]
        print("  Sampled number of households:", len(remaining_household_ids))
        print(remaining_household_ids[1])

        df = df[df["household_id"].isin(remaining_household_ids)]
        print("  Sampled number of persons:", len(np.unique(df["person_id"])))
        
        output_path = context.config("output_path")
        #df.to_csv(output_path + "/statpop_sampled.csv", index = False)

    return df
