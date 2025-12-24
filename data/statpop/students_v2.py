import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from joblib import Parallel, delayed  # kept in case you need later, but not used here

def configure(context):
    context.config("data_path")
    context.stage("data.microcensus.income_predictor")
    context.stage("data.structural_survey.structural_survey")

def execute(context):

    # =========================================================
    # CONFIG FOR ANALYSIS
    # =========================================================
    # Set to e.g. "1" or "ZH" depending on your canton_id coding AFTER string cast
    # Set to None to use ALL cantons (global comparison)
    CANTON_FOR_ANALYSIS = None   # e.g. "1" or None

    # choose model type: "gbm" or "rf"
    STUDENT_MODEL = "gbm"   # or "rf"

    # =========================================================
    # 0. PREP: CLEAN + ALIGN EMPLOYMENT INFO
    # =========================================================
    survey_df = context.stage("data.structural_survey.structural_survey")
    pop_df    = context.stage("data.microcensus.income_predictor")

    # Ensure survey targets are ints
    survey_df['employed']    = survey_df['employed'].astype('int64')
    survey_df['job_position'] = survey_df['job_position'].astype('int64')
    survey_df['is_student']  = survey_df['is_student'].astype('int64')

    # Drop rows with missing key vars in survey (safety)
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'is_student'
    ])

    # In the population, use the *drawn* employment as features
    pop_df['employed']     = pop_df['employed'].astype('int64')
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
       # 'job_position',
       # 'home_municipality_id',
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

    # Final cast + overwrite name
    pop_df['STUDENT_draw'] = pop_df['STUDENT_draw'].astype('int64')
    pop_df = pop_df.rename(columns={"STUDENT_draw": "is_student"})
    # force <15 to be student
    pop_df.loc[pop_df['age'] < 15, 'is_student'] = 1

    print("Final STUDENT distribution in population:")
    print(pop_df['is_student'].value_counts(normalize=True))

    # =========================================================
    # 6. DIAGNOSTIC: COMPARE STUDENT RATES BY AGE_BIN (SURVEY vs POP)
    # =========================================================

    # Optionally filter by canton
    if CANTON_FOR_ANALYSIS is not None:
        canton_key = str(CANTON_FOR_ANALYSIS)
        survey_diag = survey_df[survey_df['canton_id'] == canton_key].copy()
        pop_diag    = pop_df[pop_df['canton_id'].astype(str) == canton_key].copy()
        print(f"\n[DIAGNOSTIC] Analysis restricted to canton_id = {canton_key}")
    else:
        survey_diag = survey_df.copy()
        pop_diag    = pop_df.copy()
        print("\n[DIAGNOSTIC] Analysis for ALL cantons (global)")

    # helper for weighted mean
    def weighted_mean(x, w):
        return np.average(x, weights=w) if len(x) > 0 else np.nan

    # --- Survey: weighted share of students by age_bin ---
    survey_age = (
        survey_diag
        .groupby('age_bin')
        .apply(lambda g: weighted_mean(g['is_student'], g['weight']))
        .reset_index(name='share_student_survey')
    )

    # --- Population: share of students by age_bin (unweighted or weighted if available) ---
    # detect optional weight in pop
    pop_weight_col = None
    for cand in ['weight', 'person_weight', 'household_weight']:
        if cand in pop_diag.columns:
            pop_weight_col = cand
            break

    if pop_weight_col is not None:
        pop_age = (
            pop_diag
            .groupby('age_bin')
            .apply(lambda g: weighted_mean(g['is_student'], g[pop_weight_col]))
            .reset_index(name='share_student_pop')
        )
    else:
        pop_age = (
            pop_diag
            .groupby('age_bin')['is_student']
            .mean()
            .reset_index(name='share_student_pop')
        )

    # --- Merge and compute differences ---
    age_compare = pd.merge(
        survey_age,
        pop_age,
        on='age_bin',
        how='outer'
    )

    age_compare['share_student_survey'] = age_compare['share_student_survey'].fillna(0.0)
    age_compare['share_student_pop']    = age_compare['share_student_pop'].fillna(0.0)
    age_compare['diff_pop_minus_survey'] = (
        age_compare['share_student_pop'] - age_compare['share_student_survey']
    )

    print("\nShare of students by age_bin (survey vs population):")
    print(
        age_compare
        .sort_values('age_bin')
        .to_string(index=False)
    )

    return pop_df
