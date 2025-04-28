import pandas as pd
import numpy as np

def configure(context):
    context.stage("data.are_synpop.persons")
    context.stage("data.are_synpop.projections.population")

    context.config("enable_scaling", default=False)
    context.config("scaling_year", default=2015)
    
    context.config("random_seed")
    context.config("threads")  

    context.stage("data.constants")  


def compute_individual_filters(df, individual_controls):
    # create filters for individual level controls
    individual_filters = []
    for control in individual_controls:
        for _, row in control.iterrows():
            individual_filter = [row["weight"]]

            # build a filter to select all individuals that match current control values
            f_individual = np.ones(df.shape[0], dtype=bool)
            for c in list(row.drop("weight").index):
                f_individual &= (df[c] == row[c])

            individual_filter.append(row)
            individual_filter.append(f_individual)
            individual_filters.append(individual_filter)

    return individual_filters


def process_canton(canton_id, projection_df, df_population):

    print(f"Starting to weight and scale the population of canton {canton_id}.")

    # Isolate agents living in the canton
    df         = df_population[df_population["canton_id"]==canton_id]
    df_reduced = df[["synpop_person_id", "age_class", "sex", "nationality", "canton_id"]]

    # Accessing aggregated projection counts
    individual_controls = [projection_df[projection_df["canton_id"] == canton_id]]

    # Compute filters identifying the number of people for each group in the population
    individual_filters = compute_individual_filters(df_reduced, individual_controls)

    # Weight the individuals based on the difference between the controls and what is in synpop
    new_df_reduced = []

    for control_number, row, thefilter in individual_filters:
        scaling_factor = control_number / len(df_reduced[thefilter.values])

        df_group = df_reduced[thefilter.values]

        count     = np.floor(scaling_factor).astype(int)
        remainder = scaling_factor - count

        indices = np.repeat(list(df_group.index), count)
        df_replicate = df_group.loc[indices]

        choices = list(df_group.index)
        size    = control_number - len(df_replicate)

        if remainder > 0:

            indices = np.random.choice(a=choices, size = size, replace = False, p = [remainder / (remainder * len(choices)) for _ in choices])
            df_sample = df_group.loc[indices]

            new_df_group = pd.concat([df_replicate, df_sample])

        else:
             new_df_group = df_replicate

        new_df_reduced.append(new_df_group)

    new_df_reduced_all = pd.concat(new_df_reduced)

    return new_df_reduced_all


def execute(context):
    df_synpop = context.stage("data.are_synpop.persons")
    c         = context.stage("data.constants")

    if context.config("enable_scaling"):

        scaling_year = context.config("scaling_year")

        print("Scaling Synpop to year ", scaling_year, " using IPU.")

        df_population_controls, pop_year = context.stage("data.are_synpop.projections.population")

        assert pop_year == scaling_year

        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of persons before scaling :", len(df_synpop["person_id"].unique()))

        # Weighting by canton
        canton_ids        = list(df_synpop.sort_values("canton_id")["canton_id"].unique())
        processed_cantons = []

        for canton_id in context.progress(canton_ids, label="Constructing separate scaling problems by canton..."):
            df_synpop_canton = process_canton(canton_id, df_population_controls, df_synpop)
            processed_cantons.append(df_synpop_canton)

        # Concat to get the entire population
        df_synpop_new = pd.concat(processed_cantons)

        # Arrange IDs
        df_synpop_new              = df_synpop_new.sort_values(by="synpop_person_id", ascending=True)
        df_synpop_new["person_id"] = np.arange(len(df_synpop_new))

        # Merge with all attributes
        df_synpop_new = pd.merge(df_synpop_new[["person_id", "synpop_person_id"]], df_synpop[df_synpop.columns[1:]], on="synpop_person_id", how="left")

        print("Number of persons in population controls :", df_population_controls["weight"].sum())
        print("Number of persons after scaling :", len(df_synpop_new["person_id"].unique()))

    return df_synpop_new