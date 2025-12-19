import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from joblib import Parallel, delayed  # kept in case you need later, but not used here

def configure(context):
    context.config("data_path")
    context.stage("data.statpop.employment_v2")
    context.stage("data.structural_survey.structural_survey")

def execute(context):

    # =========================================================
    # 0. PREP: CLEAN + ALIGN EMPLOYMENT INFO
    # =========================================================
    survey_df = context.stage("data.structural_survey.structural_survey")

    pop_df = context.stage("data.statpop.employment_v2")

    # Ensure survey targets are ints
    survey_df['employed'] = survey_df['employed'].astype('int64')
    survey_df['job_position'] = survey_df['job_position'].astype('int64')
    survey_df['is_student'] = survey_df['is_student'].astype('int64')

    # Drop rows with missing key vars in survey (safety)
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'is_student'
    ])

    # In the population, use the *drawn* employment as features
    pop_df['employed'] = pop_df['employed'].astype('int64')
    pop_df['job_position'] = pop_df['job_position'].astype('int64')

    # =========================================================
    # 1. AGE BINS + CATEGORICAL CLEANING
    # =========================================================

    age_bins = [0, 15, 20, 25, 31, 41, 51, 66, 71, 200]
    age_labels = [
        '0-14', '15-19', '20-24', '25-30',
        '31-40', '41-50', '51-65', '66-70', '71+'
    ]

    for df in (survey_df, pop_df):
        df['age_bin'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )

    # Harmonize categoricals as strings
    cat_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # =========================================================
    # 2. DESIGN MATRICES FOR STUDENT MODEL
    # =========================================================

    # Features: age, sex, activity, job, and spatial IDs
    student_feat_cols = [
        'age_bin',
        'sex',
        'employed',
        'job_position',
        'home_municipality_id',
        'district_id',
        'canton_id'
    ]

    X_student_survey = pd.get_dummies(survey_df[student_feat_cols], drop_first=False)
    X_student_pop    = pd.get_dummies(pop_df[student_feat_cols], drop_first=False)

    student_feature_cols = X_student_survey.columns
    X_student_pop = X_student_pop.reindex(columns=student_feature_cols, fill_value=0)

    y_student = survey_df['is_student'].astype('int64')
    w_student = survey_df['weight']

    # =========================================================
    # 3. FIT ONE GLOBAL STUDENT MODEL (GBM OR RF)
    # =========================================================

    # choose model type: "gbm" or "rf"
    STUDENT_MODEL = "gbm"   # or "rf"

    if STUDENT_MODEL == "gbm":
        student_model = HistGradientBoostingClassifier(
            loss='log_loss',   # binary logistic loss
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        student_model.fit(X_student_survey, y_student, sample_weight=w_student)

    elif STUDENT_MODEL == "rf":
        student_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        student_model.fit(X_student_survey, y_student, sample_weight=w_student)

    else:
        raise ValueError(f"Unknown STUDENT_MODEL={STUDENT_MODEL}, use 'gbm' or 'rf'.")

    print("Fitted global student model using:", STUDENT_MODEL)

    # =========================================================
    # 4. STOCHASTIC DRAW HELPER
    # =========================================================

    def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
        rng = np.random.default_rng(seed)
        cum_proba = np.cumsum(proba_matrix, axis=1)
        r = rng.random(proba_matrix.shape[0])[:, None]
        chosen_idx = (r < cum_proba).argmax(axis=1)
        return classes[chosen_idx]

    # =========================================================
    # 5. STOCHASTIC ASSIGNMENT OF STUDENT_draw TO POPULATION
    # =========================================================

    pop_df['STUDENT_draw'] = np.nan

    SEED_STUDENT = 789

    proba_stu   = student_model.predict_proba(X_student_pop)
    classes_stu = student_model.classes_.astype('int64')  # should be [0,1] or [1,0]

    stu_draw = draw_multinomial_from_proba(
        proba_stu,
        classes_stu,
        seed=SEED_STUDENT
    ).astype('int64')

    pop_df['STUDENT_draw'] = stu_draw

    # Final cast
    pop_df['STUDENT_draw'] = pop_df['STUDENT_draw'].astype('int64')
    pop_df = pop_df.rename(columns={"STUDENT_draw": "is_student"})
    pop_df.loc[pop_df['age'] < 15, 'is_student'] = 1
    print("Final STUDENT_draw distribution:")
    print(pop_df['is_student'].value_counts(normalize=True))
    print(pop_df.columns)
    return pop_df
