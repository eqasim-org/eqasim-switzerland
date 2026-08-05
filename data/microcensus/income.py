from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from catboost import CatBoostClassifier


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
    I modified the imputation approach in order to only use household features and not use age, sex, ... attributes because in the Mz, they as one person from that household, so this is biased
    """

    # Features
    num_cols = [
        "household_size", "number_of_cars", "number_of_bikes"
    ]
    cat_cols = [
        "N_adults", "N_children_under_18", "N_children_under_6", "is_swiss",
        "population_density", "ovgk", "municipality_type", "canton_id", "employment_status"
    ]
    feature_cols = num_cols + cat_cols

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
        "number_of_bikes": (0, 5)
    })

    # -----------------------
    # 2. Full pipeline
    # -----------------------
    model = Pipeline(steps=[
        ("clip", clipper),
        ("clf", CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42
        ))
    ])

    # Train
    # Since we keep DataFrame through the pipeline, we can pass categorical column names to CatBoost.
    model.fit(
        df_train[feature_cols],
        df_train["income_class"],
        clf__cat_features=cat_cols
    )

    # Predict missing (probabilistic sampling)
    probas = model.predict_proba(df_pred[feature_cols])
    classes = model.named_steps["clf"].classes_

    imputed = np.array([
        np.random.choice(classes, p=p) for p in probas
    ])

    # Insert results
    df_mz.loc[missing, "income_class"] = imputed
    df_mz["income_imputed"] = False
    df_mz.loc[missing, "income_imputed"] = True

    return df_mz
    