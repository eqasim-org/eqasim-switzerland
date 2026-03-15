import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from joblib import Parallel, delayed
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger("synpp")
def configure(context):
    context.config("data_path")
    context.stage("synthesis.population.models.income")
    context.stage("data.structural_survey.structural_survey")

def execute(context):

    # =========================================================
    # CONFIG FOR ANALYSIS
    # =========================================================
    CANTON_FOR_ANALYSIS = 25   # e.g. "1" or None

    # choose model type: "gbm" or "rf" or "catboost"
    STUDENT_MODEL = "catboost"   # <-- allow "catboost" too

    CHUNK_SIZE = 50_000

    # =========================================================
    # 0. PREP: CLEAN + ALIGN EMPLOYMENT INFO
    # =========================================================
    survey_df = context.stage("data.structural_survey.structural_survey")
    pop_df    = context.stage("synthesis.population.models.income")

    survey_df['employed']     = survey_df['employed'].astype('int64')
    survey_df['job_position'] = survey_df['job_position'].astype('int64')
    survey_df['is_student']   = survey_df['is_student'].astype('int64')

    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'is_student', 'municipality_type'
    ])

    pop_df['employed']     = pop_df['employed'].astype('int64')
    pop_df['job_position'] = pop_df['job_position'].astype('int64')

    # =========================================================
    # 1. AGE BINS + CATEGORICAL CLEANING
    # =========================================================

    age_bins = [0, 15, 20, 25, 31, 41, 51, 200]
    age_labels = [
        '0-14', '15-19', '20-24', '25-30',
        '31-40', '41-50', '51+'
    ]

    for df in (survey_df, pop_df):
        df['age_bin'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )

    cat_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id', 'municipality_type']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # =========================================================
    # 2. DESIGN MATRICES FOR STUDENT MODEL
    # =========================================================

    student_feat_cols = [
        'age',          # <-- ADD
        'age_bin',
        'sex',
        'employed',
        'municipality_type',
        #'district_id',
        'canton_id'
    ]

    X_student_survey = pd.get_dummies(survey_df[student_feat_cols], drop_first=False)

    # ensure 'age' is numeric after get_dummies (robust)
    X_student_survey['age'] = survey_df['age'].astype(np.float32).to_numpy()

    # reduce memory
    X_student_survey = X_student_survey.astype(np.float32)

    student_feature_cols = X_student_survey.columns

    y_student = survey_df['is_student'].astype('int64')
    w_student = survey_df['weight']

    # =========================================================
    # 3. FIT ONE GLOBAL STUDENT MODEL (GBM / RF / CATBOOST)
    # =========================================================

    if STUDENT_MODEL == "gbm":
        student_model = HistGradientBoostingClassifier(
            loss='log_loss',
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
            n_jobs=1
        )
        student_model.fit(X_student_survey, y_student, sample_weight=w_student)

    elif STUDENT_MODEL in ("catboost", "cat"):
        student_model = CatBoostClassifier(
            loss_function="Logloss",   # binary
            iterations=4200,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=6.0,
            random_seed=42,
            verbose=100,
            bootstrap_type="Bernoulli",
            subsample=0.8
        )
        student_model.fit(X_student_survey, y_student, sample_weight=w_student)

    else:
        raise ValueError(f"Unknown STUDENT_MODEL={STUDENT_MODEL}, use 'gbm' or 'rf' or 'catboost'.")

    logger.info(f"Fitted global student model using: {STUDENT_MODEL}")

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
    # 5. STOCHASTIC ASSIGNMENT OF STUDENT_draw TO POPULATION (CHUNKED)
    # =========================================================

    pop_df['STUDENT_draw'] = np.nan
    SEED_STUDENT = 789

    classes_stu = student_model.classes_.astype('int64')

    n = len(pop_df)
    stu_out = np.empty(n, dtype=np.int64)

    logger.info(f"Predicting students in chunks: n={n:,}, CHUNK_SIZE={CHUNK_SIZE:,}")

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = pop_df.iloc[start:end]

        X_chunk = pd.get_dummies(chunk[student_feat_cols], drop_first=False)

        # ✅ ensure numeric 'age' exists in chunk too
        X_chunk['age'] = chunk['age'].astype(np.float32).to_numpy()

        X_chunk = X_chunk.reindex(columns=student_feature_cols, fill_value=0).astype(np.float32)

        proba_stu = student_model.predict_proba(X_chunk)

        stu_draw = draw_multinomial_from_proba(
            proba_stu,
            classes_stu,
            seed=SEED_STUDENT + start
        ).astype('int64')

        stu_out[start:end] = stu_draw

        del X_chunk, proba_stu, stu_draw
        gc.collect()

        if (start // CHUNK_SIZE) % 20 == 0:
            logger.info(f"  ... processed {end:,}/{n:,}")

    pop_df['STUDENT_draw'] = stu_out

    pop_df['STUDENT_draw'] = pop_df['STUDENT_draw'].astype('int64')
    pop_df = pop_df.rename(columns={"STUDENT_draw": "is_student"})
    pop_df.loc[pop_df['age'] < 15, 'is_student'] = 1

    logger.info("Final STUDENT distribution in population:")
    logger.info(pop_df['is_student'].value_counts(normalize=True))

    # =========================================================
    # 6. DIAGNOSTIC: COMPARE STUDENT RATES BY AGE_BIN (SURVEY vs POP)
    # =========================================================

    if CANTON_FOR_ANALYSIS is not None:
        canton_key = str(CANTON_FOR_ANALYSIS)
        survey_diag = survey_df[survey_df['canton_id'] == canton_key]
        pop_diag    = pop_df[pop_df['canton_id'].astype(str) == canton_key]
        logger.info(f"\n[DIAGNOSTIC] Analysis restricted to canton_id = {canton_key}")
    else:
        survey_diag = survey_df
        pop_diag    = pop_df
        logger.info("\n[DIAGNOSTIC] Analysis for ALL cantons (global)")

    def weighted_mean(x, w):
        return np.average(x, weights=w) if len(x) > 0 else np.nan

    survey_age = (
        survey_diag
        .groupby('age_bin')
        .apply(lambda g: weighted_mean(g['is_student'], g['weight']))
        .reset_index(name='share_student_survey')
    )

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

    logger.info("\nShare of students by age_bin (survey vs population):")
    logger.info(
        age_compare
        .sort_values('age_bin')
        .to_string(index=False)
    )

    pop_df['canton_id'] = pop_df['canton_id'].astype("int64")
    return pop_df
