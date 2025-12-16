import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# helper: stochastic draw
# ---------------------------------------------------------
def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
    rng = np.random.default_rng(seed)
    cum_proba = np.cumsum(proba_matrix, axis=1)
    r = rng.random(proba_matrix.shape[0])[:, None]
    chosen_idx = (r < cum_proba).argmax(axis=1)
    return classes[chosen_idx]

def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.statpop.statpop")

def execute(context):
    # -------------------------------------------------------------------
    # 0. BASIC PREP
    # -------------------------------------------------------------------

    survey_df = context.stage("data.structural_survey.structural_survey")
    survey_df['CURRACTIVITYSTATUSI'] = survey_df['CURRACTIVITYSTATUSI'].astype(int)
    survey_df['STATUSINEMPL_DETAIL'] = survey_df['STATUSINEMPL_DETAIL'].astype(int)
    survey_df = survey_df.rename(columns={"RES_DISTRICT": "district_id"})
    survey_df = survey_df.rename(columns={"RES_CANTON": "canton_id"})

    pop_df = context.stage("data.statpop.statpop")
    pop_df = pop_df.rename(columns={"age": "AGE"})
    pop_df = pop_df[pop_df["AGE"] >= 15]
    pop_df = pop_df.rename(columns={"sex": "SEX"})

    sum_by_status = survey_df.groupby("CURRACTIVITYSTATUSI")["weight"].sum()
    print("Survey CURRACTIVITYSTATUSI weighted totals:")
    print(sum_by_status)

    # =========================================================
    # 1. BASIC CLEANING + AGE BINS
    # =========================================================

    survey_df['CURRACTIVITYSTATUSI'] = survey_df['CURRACTIVITYSTATUSI'].astype('int64')
    survey_df['STATUSINEMPL_DETAIL'] = survey_df['STATUSINEMPL_DETAIL'].astype('int64')

    survey_df = survey_df.dropna(subset=[
        'AGE', 'SEX', 'home_municipality_id', 'district_id', 'canton_id',
        'CURRACTIVITYSTATUSI', 'STATUSINEMPL_DETAIL'
    ])

    # Age bins
    age_bins = [0, 15, 20, 25, 31, 41, 51, 66, 71, 200]
    age_labels = [
        '0-14', '15-19', '20-24', '25-30',
        '31-40', '41-50', '51-65', '66-70', '71+'
    ]

    for df in (survey_df, pop_df):
        df['AGE_BIN'] = pd.cut(
            df['AGE'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )

    # Harmonize categoricals as strings
    cat_cols = ['AGE_BIN', 'SEX', 'home_municipality_id', 'district_id', 'canton_id']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # =========================================================
    # 2. GLOBAL DESIGN MATRICES (AGE_BIN + SEX + GEO IDs)
    # =========================================================

    # Features we will use for both activity and job models
    feature_cols = ['AGE_BIN', 'SEX', 'home_municipality_id', 'district_id', 'canton_id']

    X_survey = pd.get_dummies(survey_df[feature_cols], drop_first=False)
    X_pop    = pd.get_dummies(pop_df[feature_cols], drop_first=False)

    # Align columns between survey and population
    global_feature_cols = X_survey.columns
    X_pop = X_pop.reindex(columns=global_feature_cols, fill_value=0)

    w = survey_df['weight']

    # =========================================================
    # 3. FIT GLOBAL ACTIVITY MODEL (CURRACTIVITYSTATUSI)
    # =========================================================

    # Choose model type here: "gbm" or "rf"
    ACTIVITY_MODEL = "gbm"   # or "rf"

    y_act = survey_df['CURRACTIVITYSTATUSI'].astype('int64')

    if ACTIVITY_MODEL == "gbm":
        act_model = HistGradientBoostingClassifier(
            loss='log_loss',
            max_depth=3,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        act_model.fit(X_survey, y_act, sample_weight=w)

    elif ACTIVITY_MODEL == "rf":
        act_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        act_model.fit(X_survey, y_act, sample_weight=w)

    else:
        raise ValueError(f"Unknown ACTIVITY_MODEL={ACTIVITY_MODEL}, use 'gbm' or 'rf'.")

    print("Fitted global activity model using:", ACTIVITY_MODEL)

    # =========================================================
    # 4. FIT GLOBAL JOB MODEL (STATUSINEMPL_DETAIL, ONLY EMPLOYED)
    # =========================================================

    # Choose job model: "gbm" or "rf"
    JOB_MODEL = "gbm"   # or "rf"

    df_emp = survey_df[survey_df['CURRACTIVITYSTATUSI'] == 1]
    X_job_survey = X_survey.loc[df_emp.index]
    y_job = df_emp['STATUSINEMPL_DETAIL'].astype('int64')
    w_job = df_emp['weight']

    if JOB_MODEL == "gbm":
        job_model = HistGradientBoostingClassifier(
            loss='log_loss',   # handles multiclass as well
            max_depth=3,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        job_model.fit(X_job_survey, y_job, sample_weight=w_job)

    elif JOB_MODEL == "rf":
        job_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        job_model.fit(X_job_survey, y_job, sample_weight=w_job)

    else:
        raise ValueError(f"Unknown JOB_MODEL={JOB_MODEL}, use 'gbm' or 'rf'.")

    print("Fitted global job model using:", JOB_MODEL)

    # =========================================================
    # 5. STOCHASTIC PREDICTION FOR POPULATION
    # =========================================================

    pop_df['CURRACTIVITYSTATUSI_draw'] = np.nan
    pop_df['STATUSINEMPL_DETAIL_draw'] = np.nan

    SEED_ACTIVITY = 123
    SEED_JOB      = 456

    # ----- Activity (1/2/3) -----
    proba_act   = act_model.predict_proba(X_pop)
    classes_act = act_model.classes_.astype('int64')

    act_draw = draw_multinomial_from_proba(
        proba_act,
        classes_act,
        seed=SEED_ACTIVITY
    ).astype('int64')

    pop_df['CURRACTIVITYSTATUSI_draw'] = act_draw

    # ----- Job type -----
    job_draw = np.full(len(pop_df), np.nan)

    # unemployed -> 60, not in workforce -> 70
    job_draw[act_draw == 2] = 60
    job_draw[act_draw == 3] = 70

    # employed -> global job model
    emp_mask = (act_draw == 1)
    if emp_mask.any():
        X_job_pop = X_pop.loc[emp_mask]

        proba_job   = job_model.predict_proba(X_job_pop)
        classes_job = job_model.classes_.astype('int64')

        job_draw_emp = draw_multinomial_from_proba(
            proba_job,
            classes_job,
            seed=SEED_JOB
        ).astype('int64')

        job_draw[emp_mask] = job_draw_emp

    pop_df['STATUSINEMPL_DETAIL_draw'] = job_draw

    # =========================================================
    # 6. FILL ANY REMAINING NANS + CAST
    # =========================================================

    na_job = pop_df['STATUSINEMPL_DETAIL_draw'].isna().sum()
    print(f"NaNs in STATUSINEMPL_DETAIL_draw before fallback: {na_job}")

    if na_job > 0:
        curr = pop_df['CURRACTIVITYSTATUSI_draw'].astype('int64')
        mask_na = pop_df['STATUSINEMPL_DETAIL_draw'].isna()

        pop_df.loc[mask_na & (curr == 2), 'STATUSINEMPL_DETAIL_draw'] = 60
        pop_df.loc[mask_na & (curr == 3), 'STATUSINEMPL_DETAIL_draw'] = 70
        pop_df.loc[mask_na & (curr == 1), 'STATUSINEMPL_DETAIL_draw'] = 43  # generic employee

    pop_df['CURRACTIVITYSTATUSI_draw'] = pop_df['CURRACTIVITYSTATUSI_draw'].astype('int64')
    pop_df['STATUSINEMPL_DETAIL_draw'] = pop_df['STATUSINEMPL_DETAIL_draw'].astype('int64')

    print("Final CURRACTIVITYSTATUSI_draw distribution:")
    print(pop_df['CURRACTIVITYSTATUSI_draw'].value_counts(normalize=True))

    print("\nFinal STATUSINEMPL_DETAIL_draw distribution:")
    print(pop_df['STATUSINEMPL_DETAIL_draw'].value_counts(normalize=True))

    # Overwrite original columns with draws
    pop_df = pop_df.rename(columns={"CURRACTIVITYSTATUSI_draw": "CURRACTIVITYSTATUSI"})
    pop_df = pop_df.rename(columns={"STATUSINEMPL_DETAIL_draw": "STATUSINEMPL_DETAIL"})

    return pop_df
