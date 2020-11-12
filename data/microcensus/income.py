import sklearn.tree

def impute(df_mz):
    # Train the tree
    no_income_selector = df_mz["income_class"] == -1

    # TODO: We don't use any weights here. Shouldn't we?
    training_data = df_mz[~no_income_selector][[
        "age", "sex", "marital_status", "household_size", "number_of_cars", "number_of_bikes"
    ]].values

    training_labels = df_mz[~no_income_selector]["income_class"].values
    training_weights = df_mz[~no_income_selector]["person_weight"].values

    # TODO: Maybe adjusted later!
    classifier = sklearn.tree.DecisionTreeClassifier(min_samples_leaf = 30, max_depth = 5)

    classifier.fit(X=training_data, y=training_labels, sample_weight=None)

    # Predict the incomes
    prediction_data = df_mz[no_income_selector][[
        "age", "sex", "marital_status", "household_size", "number_of_cars", "number_of_bikes"
    ]].values

    df_mz.loc[no_income_selector, "income_class"] = classifier.predict(prediction_data)

    df_mz["income_imputed"] = False
    df_mz.loc[no_income_selector, "income_imputed"] = True

    return df_mz

    # TODO: Also, we can visualize such a tree nicely if it doesn't get too large.
