import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

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
    pop_df = pop_df[pop_df["AGE"]>=15]
    pop_df = pop_df.rename(columns={"sex": "SEX"})
    #pop_df = pop_df.rename(columns={"canton_id": "RES_CANTON"})

    sum_by_status = survey_df.groupby("CURRACTIVITYSTATUSI")["weight"].sum()

    print(sum_by_status)
  
    # =========================================================
    # 0. BASIC CLEANING + AGE BINS
    # =========================================================

    # Ensure targets are ints
    survey_df['CURRACTIVITYSTATUSI'] = survey_df['CURRACTIVITYSTATUSI'].astype('int64')
    survey_df['STATUSINEMPL_DETAIL'] = survey_df['STATUSINEMPL_DETAIL'].astype('int64')

    # Drop rows with missing key vars in survey
    survey_df = survey_df.dropna(subset=[
        'AGE', 'SEX', 'home_municipality_id', 'district_id', 'canton_id',
        'CURRACTIVITYSTATUSI', 'STATUSINEMPL_DETAIL'
    ])

    # Age bins: 15–19, 20–24, 25–30, 31–40, 41–50, 51–65, 66–70, 71+
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
    # 1. DESIGN MATRICES (AGE_BIN + SEX dummies)
    # =========================================================

    age_sex_cols = ['AGE_BIN', 'SEX']

    X_age_sex_survey = pd.get_dummies(survey_df[age_sex_cols], drop_first=False)
    X_age_sex_pop    = pd.get_dummies(pop_df[age_sex_cols], drop_first=False)

    # Ensure same columns in pop & survey
    global_age_sex_cols = X_age_sex_survey.columns
    X_age_sex_pop = X_age_sex_pop.reindex(columns=global_age_sex_cols, fill_value=0)

    # Short alias
    w = survey_df['weight']

    # =========================================================
    # 2. THRESHOLDS + MODEL STORAGE
    # =========================================================

    MIN_MUNI_ACTIVITY     = 500 #use 200
    MIN_DISTRICT_ACTIVITY = 500 #use 200
    MIN_CANTON_ACTIVITY   = 500 #use 200

    MIN_MUNI_JOB          = 30
    MIN_DISTRICT_JOB      = 50
    MIN_CANTON_JOB        = 100

    muni_activity_models     = {}
    district_activity_models = {}
    canton_activity_models   = {}

    muni_job_models          = {}
    district_job_models      = {}
    canton_job_models        = {}

    # How many cores to use (-1 = all)
    N_JOBS = -1

    # =========================================================
    # 3. HELPERS TO FIT SINGLE MODELS
    # =========================================================

    def fit_activity_gbm(df_sub):
        """Fit GBM for CURRACTIVITYSTATUSI on AGE_BIN + SEX."""
        X = pd.get_dummies(df_sub[age_sex_cols], drop_first=False)
        X = X.reindex(columns=global_age_sex_cols, fill_value=0)
        y = df_sub['CURRACTIVITYSTATUSI'].astype('int64')
        w_sub = df_sub['weight']

        clf = HistGradientBoostingClassifier(
            loss='log_loss',
            max_depth=3,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        clf.fit(X, y, sample_weight=w_sub)
        return clf

    def fit_job_multinom(df_sub):
        """Fit multinomial LR for STATUSINEMPL_DETAIL on AGE_BIN + SEX (employed only)."""
        X = pd.get_dummies(df_sub[age_sex_cols], drop_first=False)
        X = X.reindex(columns=global_age_sex_cols, fill_value=0)
        y = df_sub['STATUSINEMPL_DETAIL'].astype('int64')
        w_sub = df_sub['weight']

        clf = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000
        )
        clf.fit(X, y, sample_weight=w_sub)
        return clf

    # =========================================================
    # 4. PARALLEL MUNICIPALITY-LEVEL FITTING
    # =========================================================

    def fit_muni_models(muni_id, df_m):
        """
        Fit activity GBM + job LR for a single municipality.
        Returns (muni_id, act_model_or_None, job_model_or_None).
        """
        act_model = None
        job_model = None

        if len(df_m) >= MIN_MUNI_ACTIVITY:
            act_model = fit_activity_gbm(df_m)

        df_m_emp = df_m[df_m['CURRACTIVITYSTATUSI'] == 1]
        if len(df_m_emp) >= MIN_MUNI_JOB:
            job_model = fit_job_multinom(df_m_emp)

        return muni_id, act_model, job_model

    muni_groups = list(survey_df.groupby('home_municipality_id'))
    n_muni = len(muni_groups)
    print(f"Fitting municipality models in parallel for {n_muni} municipalities...")

    muni_results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=10)(
        delayed(fit_muni_models)(muni_id, df_m)
        for muni_id, df_m in muni_groups
    )

    print("Finished fitting municipality models, collecting results...")

    for muni_id, act_model, job_model in muni_results:
        if act_model is not None:
            muni_activity_models[muni_id] = act_model
        if job_model is not None:
            muni_job_models[muni_id] = job_model

    print(f"Trained {len(muni_activity_models)} municipality activity models.")
    print(f"Trained {len(muni_job_models)} municipality job models.")

    # =========================================================
    # 5. PARALLEL DISTRICT-LEVEL FITTING
    # =========================================================

    def fit_district_models(dist_id, df_d):
        """
        Fit activity GBM + job LR for a single district.
        Returns (dist_id, act_model_or_None, job_model_or_None).
        """
        act_model = None
        job_model = None

        if len(df_d) >= MIN_DISTRICT_ACTIVITY:
            act_model = fit_activity_gbm(df_d)

        df_d_emp = df_d[df_d['CURRACTIVITYSTATUSI'] == 1]
        if len(df_d_emp) >= MIN_DISTRICT_JOB:
            job_model = fit_job_multinom(df_d_emp)

        return dist_id, act_model, job_model

    district_groups = list(survey_df.groupby('district_id'))
    n_districts = len(district_groups)
    print(f"Fitting district models in parallel for {n_districts} districts...")

    district_results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=10)(
        delayed(fit_district_models)(dist_id, df_d)
        for dist_id, df_d in district_groups
    )

    print("Finished fitting district models, collecting results...")

    for dist_id, act_model, job_model in district_results:
        if act_model is not None:
            district_activity_models[dist_id] = act_model
        if job_model is not None:
            district_job_models[dist_id] = job_model

    print(f"Trained {len(district_activity_models)} district activity models.")
    print(f"Trained {len(district_job_models)} district job models.")

    # =========================================================
    # 6. CANTON-LEVEL FITTING (SEQUENTIAL)
    # =========================================================

    canton_groups = list(survey_df.groupby('canton_id'))
    n_cantons = len(canton_groups)
    print(f"Fitting canton models sequentially for {n_cantons} cantons...")

    for i, (canton_id, df_c) in enumerate(canton_groups, start=1):
        print(f"[Canton {i}/{n_cantons}] id={canton_id}, n={len(df_c)}")

        # Activity
        if len(df_c) < MIN_CANTON_ACTIVITY:
            raise ValueError(f"Not enough activity data in canton {canton_id} "
                            f"({len(df_c)} obs). Lower MIN_CANTON_ACTIVITY or check data.")
        clf_c_act = fit_activity_gbm(df_c)
        canton_activity_models[canton_id] = clf_c_act

        # Job (employed only)
        df_c_emp = df_c[df_c['CURRACTIVITYSTATUSI'] == 1]
        if len(df_c_emp) < MIN_CANTON_JOB:
            raise ValueError(f"Not enough job data in canton {canton_id} "
                            f"({len(df_c_emp)} employed). Lower MIN_CANTON_JOB or check data.")
        clf_c_job = fit_job_multinom(df_c_emp)
        canton_job_models[canton_id] = clf_c_job

    print(f"Trained {len(canton_activity_models)} canton activity models.")
    print(f"Trained {len(canton_job_models)} canton job models.")

    # =========================================================
    # 8. STOCHASTIC PREDICTION (MUNI -> DISTRICT -> CANTON)
    # =========================================================

    pop_df['CURRACTIVITYSTATUSI_draw'] = np.nan
    pop_df['STATUSINEMPL_DETAIL_draw'] = np.nan

    SEED_ACTIVITY_BASE = 123
    SEED_JOB_BASE      = 456

    muni_groups_pop = list(pop_df.groupby('home_municipality_id'))
    n_muni_pop = len(muni_groups_pop)
    print(f"Assigning statuses for population across {n_muni_pop} municipalities...")

    for i, (muni_id, df_p) in enumerate(muni_groups_pop, start=1):
        idx = df_p.index
        dist_id   = df_p['district_id'].iloc[0]
        canton_id = df_p['canton_id'].iloc[0]

        if i % 100 == 0 or i == 1 or i == n_muni_pop:
            print(f"[Assign] Municipality {i}/{n_muni_pop} id={muni_id}, n={len(df_p)}")

        X_p_age_sex = X_age_sex_pop.loc[idx]

        # ----- Activity model hierarchy -----
        if muni_id in muni_activity_models:
            clf_act = muni_activity_models[muni_id]
        elif dist_id in district_activity_models:
            clf_act = district_activity_models[dist_id]
        else:
            clf_act = canton_activity_models[canton_id]

        proba_act   = clf_act.predict_proba(X_p_age_sex)
        classes_act = clf_act.classes_.astype('int64')

        seed_act = SEED_ACTIVITY_BASE + (hash(muni_id) % 10000)
        act_draw = draw_multinomial_from_proba(
            proba_act, classes_act, seed=seed_act
        ).astype('int64')

        pop_df.loc[idx, 'CURRACTIVITYSTATUSI_draw'] = act_draw

        # ----- Job type -----
        job_draw = np.full(len(idx), np.nan)

        # unemployed -> 60, not in workforce -> 70
        job_draw[act_draw == 2] = 60
        job_draw[act_draw == 3] = 70

        # employed -> job model hierarchy
        emp_mask_local = (act_draw == 1)
        if emp_mask_local.any():
            idx_emp = idx[emp_mask_local]
            X_p_emp = X_age_sex_pop.loc[idx_emp]

            if muni_id in muni_job_models:
                clf_job = muni_job_models[muni_id]
            elif dist_id in district_job_models:
                clf_job = district_job_models[dist_id]
            else:
                clf_job = canton_job_models[canton_id]

            proba_job   = clf_job.predict_proba(X_p_emp)
            classes_job = clf_job.classes_.astype('int64')

            seed_job = SEED_JOB_BASE + (hash(muni_id) % 10000)
            job_draw_emp = draw_multinomial_from_proba(
                proba_job, classes_job, seed=seed_job
            ).astype('int64')

            job_draw[emp_mask_local] = job_draw_emp

        pop_df.loc[idx, 'STATUSINEMPL_DETAIL_draw'] = job_draw

    # =========================================================
    # 9. FILL ANY REMAINING NANS + CAST
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
    pop_df = pop_df.rename(columns={"CURRACTIVITYSTATUSI_draw": "CURRACTIVITYSTATUSI"})
    pop_df = pop_df.rename(columns={"STATUSINEMPL_DETAIL_draw": "STATUSINEMPL_DETAIL"})
    return pop_df

# =========================================================
# 7. STOCHASTIC DRAW HELPER
# =========================================================

def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
    rng = np.random.default_rng(seed)
    cum_proba = np.cumsum(proba_matrix, axis=1)
    r = rng.random(proba_matrix.shape[0])[:, None]
    chosen_idx = (r < cum_proba).argmax(axis=1)
    return classes[chosen_idx]

