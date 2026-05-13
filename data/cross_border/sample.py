import numpy as np

def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("random_seed")

    context.stage("data.cross_border.generate_od")


def execute(context):
    df = context.stage("data.cross_border.generate_od")

    # If we do not want to downsample, set the value to 1.0 in config
    probability = context.config("input_downsampling")

    if probability < 1.0:
        print("Downsampling - cross-border population (%f)" % probability)

        person_ids = np.unique(df["cross_border_person_id"])
        print("  Initial number of persons:", len(person_ids))

        # Set up RNG
        random = np.random.RandomState(context.config("random_seed"))
        
        # Perform sampling
        f = random.random_sample(size=(len(person_ids),)) < probability
        remaining_person_ids = person_ids[f]
        print(f"  Sampled number of persons: {len(remaining_person_ids)}")

        df = df[df["cross_border_person_id"].isin(remaining_person_ids)]

    print(df.columns)

    return df