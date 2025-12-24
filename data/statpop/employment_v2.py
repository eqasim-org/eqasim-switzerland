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
    # CONFIG FOR DIAGNOSTICS
    # -------------------------------------------------------------------
    # Set to None for global comparison, or to e.g. "1" to inspect canton_id==1
    CANTON_FOR_ANALYSIS = None  # e.g. "1"

    # -------------------------------------------------------------------
    # 0. BASIC PREP
    # -------------------------------------------------------------------
    survey_df = context.stage("data.structural_survey.structural_survey")
    survey_df['employed'] = survey_df['employed'].astype('int64')
    survey_df['job_position'] = survey_df['job_position'].astype('int64')

    pop_df = context.stage("data.statpop.statpop")

    sum_by_status = survey_df.groupby("employed")["weight"].sum()
    print("Survey employed weighted totals:")
    print(sum_by_status)

    # =========================================================
    # 1. BASIC CLEANING + AGE BINS
    # =========================================================
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position'
    ])

    # Age bins
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
    # 2. GLOBAL DESIGN MATRICES (AGE_BIN + SEX + GEO IDs)
    # =========================================================
    feature_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id']

    X_survey = pd.get_dummies(survey_df[feature_cols], drop_first=False)
    X_pop    = pd.get_dummies(pop_df[feature_cols], drop_first=False)

    global_feature_cols = X_survey.columns
    X_pop = X_pop.reindex(columns=global_feature_cols, fill_value=0)

    w = survey_df['weight']

    # =========================================================
    # 3. FIT GLOBAL ACTIVITY MODEL (employed)
    # =========================================================
    ACTIVITY_MODEL = "gbm"   # or "rf"
    y_act = survey_df['employed'].astype('int64')

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
    # 4. FIT GLOBAL JOB MODEL (job_position, ONLY EMPLOYED)
    # =========================================================
    JOB_MODEL = "gbm"   # or "rf"

    df_emp = survey_df[survey_df['employed'] == 1]
    X_job_survey = X_survey.loc[df_emp.index]
    y_job = df_emp['job_position'].astype('int64')
    w_job = df_emp['weight']

    if JOB_MODEL == "gbm":
        job_model = HistGradientBoostingClassifier(
            loss='log_loss',
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
    pop_df['employed_draw'] = np.nan
    pop_df['job_position_draw'] = np.nan

    SEED_ACTIVITY = 123
    SEED_JOB      = 456

    proba_act   = act_model.predict_proba(X_pop)
    classes_act = act_model.classes_.astype('int64')

    act_draw = draw_multinomial_from_proba(
        proba_act,
        classes_act,
        seed=SEED_ACTIVITY
    ).astype('int64')

    pop_df['employed_draw'] = act_draw

    job_draw = np.full(len(pop_df), np.nan)
    job_draw[act_draw == 2] = 60
    job_draw[act_draw == 3] = 70

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

    pop_df['job_position_draw'] = job_draw

    # =========================================================
    # 6. FILL ANY REMAINING NANS + CAST
    # =========================================================
    na_job = pop_df['job_position_draw'].isna().sum()
    print(f"NaNs in job_position_draw before fallback: {na_job}")

    if na_job > 0:
        curr = pop_df['employed_draw'].astype('int64')
        mask_na = pop_df['job_position_draw'].isna()

        pop_df.loc[mask_na & (curr == 2), 'job_position_draw'] = 60
        pop_df.loc[mask_na & (curr == 3), 'job_position_draw'] = 70
        pop_df.loc[mask_na & (curr == 1), 'job_position_draw'] = 43  # generic employee

    pop_df['employed_draw'] = pop_df['employed_draw'].astype('int64')
    pop_df['job_position_draw'] = pop_df['job_position_draw'].astype('int64')

    print("Final employed_draw distribution:")
    print(pop_df['employed_draw'].value_counts(normalize=True))

    # Overwrite original columns with draws
    pop_df = pop_df.rename(columns={"employed_draw": "employed"})
    pop_df = pop_df.rename(columns={"job_position_draw": "job_position"})
    pop_df.loc[pop_df['age'] < 15, 'employed'] = 3
    pop_df.loc[pop_df['age'] < 15, 'job_position'] = 70

    # =========================================================
    # 7. DIAGNOSTIC: EMPLOYED DISTRIBUTION BY AGE_BIN (SURVEY vs POP)
    # =========================================================

    if CANTON_FOR_ANALYSIS is not None:
        canton_key = str(CANTON_FOR_ANALYSIS)
        survey_diag = survey_df[survey_df['canton_id'] == canton_key].copy()
        pop_diag    = pop_df[pop_df['canton_id'].astype(str) == canton_key].copy()
        print(f"\n[DIAGNOSTIC] Employed-by-age analysis restricted to canton_id={canton_key}")
    else:
        survey_diag = survey_df.copy()
        pop_diag    = pop_df.copy()
        print("\n[DIAGNOSTIC] Employed-by-age analysis for ALL cantons (global)")

    # --- (A) share of employed==1 by age_bin (survey weighted vs pop unweighted) ---
    survey_employed_rate = (
        survey_diag
        .groupby('age_bin')
        .apply(lambda g: np.average((g['employed'] == 1).astype(float), weights=g['weight']))
        .reset_index(name='share_employed1_survey')
    )

    pop_employed_rate = (
        pop_diag
        .groupby('age_bin')
        .apply(lambda g: (g['employed'] == 1).mean())
        .reset_index(name='share_employed1_pop')
    )

    employed_rate_compare = pd.merge(
        survey_employed_rate,
        pop_employed_rate,
        on='age_bin',
        how='outer'
    ).fillna(0.0)

    employed_rate_compare['diff_pop_minus_survey'] = (
        employed_rate_compare['share_employed1_pop'] - employed_rate_compare['share_employed1_survey']
    )

    print("\nShare employed==1 by age_bin (survey vs population):")
    print(
        employed_rate_compare
        .sort_values('age_bin')
        [['age_bin', 'share_employed1_survey', 'share_employed1_pop', 'diff_pop_minus_survey']]
        .to_string(index=False)
    )

    # --- (B) full distribution of employed (1/2/3) by age_bin ---
    # Survey: weighted shares
    survey_mass = (
        survey_diag
        .groupby(['age_bin', 'employed'], as_index=False)['weight']
        .sum()
        .rename(columns={'weight': 'w'})
    )
    survey_tot = (
        survey_mass
        .groupby('age_bin', as_index=False)['w']
        .sum()
        .rename(columns={'w': 'w_tot'})
    )
    survey_mass = survey_mass.merge(survey_tot, on='age_bin', how='left')
    survey_mass['share_survey'] = survey_mass['w'] / survey_mass['w_tot']

    # Pop: unweighted shares (statpop is complete, so count shares are fine)
    pop_mass = (
        pop_diag
        .groupby(['age_bin', 'employed'], as_index=False)
        .size()
        .rename(columns={'size': 'n'})
    )
    pop_tot = (
        pop_mass
        .groupby('age_bin', as_index=False)['n']
        .sum()
        .rename(columns={'n': 'n_tot'})
    )
    pop_mass = pop_mass.merge(pop_tot, on='age_bin', how='left')
    pop_mass['share_pop'] = pop_mass['n'] / pop_mass['n_tot']

    dist_compare = pd.merge(
        survey_mass[['age_bin', 'employed', 'share_survey']],
        pop_mass[['age_bin', 'employed', 'share_pop']],
        on=['age_bin', 'employed'],
        how='outer'
    ).fillna(0.0)

    dist_compare['share_diff'] = dist_compare['share_pop'] - dist_compare['share_survey']

    print("\nFull employed distribution by age_bin (survey vs population):")
    print(
        dist_compare
        .sort_values(['age_bin', 'employed'])
        [['age_bin', 'employed', 'share_survey', 'share_pop', 'share_diff']]
        .to_string(index=False)
    )

    return pop_df
