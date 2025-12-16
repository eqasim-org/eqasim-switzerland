import sklearn.tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


######################################################################
################### Model similar to the old model ###################
######################################################################

def impute_basic(df_mz):
    # Train the tree
    no_income_selector = df_mz["income_class"] == -1

    training_data = df_mz[~no_income_selector][[
        "age", "sex", "marital_status", "household_size", "number_of_cars", "number_of_bikes"
    ]].values

    training_labels = df_mz[~no_income_selector]["income_class"].values
    training_weights = df_mz[~no_income_selector]["person_weight"].values

    # Use sample weights for proper representation in training
    classifier = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=30, max_depth=5)

    classifier.fit(X=training_data, y=training_labels, sample_weight=training_weights)

    # Predict the incomes using probabilities and sampling for variability
    prediction_data = df_mz[no_income_selector][[
        "age", "sex", "marital_status", "household_size", "number_of_cars", "number_of_bikes"
    ]].values

    probabilities = classifier.predict_proba(prediction_data)
    classes = classifier.classes_
    sampled_classes = np.array([
        np.random.choice(classes, p=prob_row) for prob_row in probabilities
    ])
    df_mz.loc[no_income_selector, "income_class"] = sampled_classes

    df_mz["income_imputed"] = False
    df_mz.loc[no_income_selector, "income_imputed"] = True

    return df_mz


#########################################################################
################### Advanced imputation using sklearn ###################
#########################################################################

class ColumnClipper(BaseEstimator, TransformerMixin):
    def __init__(self, clip_dict):
        """
        clip_dict example:
        {
            "household_size": (1, 10),
            "number_of_cars": (0, 5),
            "number_of_bikes": (0, 10)
        }
        """
        self.clip_dict = clip_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.clip_dict.items():
            X[col] = X[col].clip(lower=low, upper=high)
        return X
    
def impute(df_mz):
    """
    Impute missing income_class values (-1) using a RandomForest Pipeline.
    It uses numerical and categorical features to predict the income class.
    """

    # Features
    num_cols = [
        "age", "sex", "marital_status", "household_size",
        "number_of_cars", "number_of_bikes"
    ]
    cat_cols = [
        "highest_education", "employment_status",
        "municipality_type", "canton_id"
    ]

    missing = df_mz["income_class"] == -1
    df_train = df_mz[~missing].copy()
    df_pred = df_mz[missing].copy()

    # Return early if no missing
    if df_pred.empty:
        df_mz["income_imputed"] = False
        return df_mz

    # -----------------------
    # 1. Build preprocessing
    # -----------------------
    clipper = ColumnClipper({
                        "household_size": (1, 10),
                        "number_of_cars": (0, 5),
                        "number_of_bikes": (0, 10)
                    })

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    # -----------------------
    # 2. Full pipeline
    # -----------------------
    model = Pipeline(steps=[
                        ("clip", clipper),     
                        ("preprocess", preprocessor),
                        ("clf", RandomForestClassifier(
                            n_estimators=100,
                            min_samples_leaf=20,
                            max_depth=10,                            
                        ))
                    ])

    # Train
    model.fit(
        df_train[num_cols + cat_cols],
        df_train["income_class"]
    )

    # Predict missing (probabilistic sampling)
    probas = model.predict_proba(df_pred[num_cols + cat_cols])
    classes = model.named_steps["clf"].classes_

    imputed = np.array([
        np.random.choice(classes, p=p) for p in probas
    ])

    # Insert results
    df_mz.loc[missing, "income_class"] = imputed
    df_mz["income_imputed"] = False
    df_mz.loc[missing, "income_imputed"] = True

    return df_mz
  