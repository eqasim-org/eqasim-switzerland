import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import Parallel, delayed

def configure(context):
    context.config("data_path")
    context.stage("data.statpop.employment")
    context.stage("data.structural_survey.structural_survey")


def execute(context):
    

    # =========================================================
    # 0. PREP: CLEAN + ALIGN EMPLOYMENT INFO
    # =========================================================
    survey_df = context.stage("data.structural_survey.structural_survey")
    survey_df = survey_df.rename(columns={"RES_DISTRICT": "district_id"})
    survey_df = survey_df.rename(columns={"RES_CANTON": "canton_id"})
    pop_df = context.stage("data.statpop.employment")
    # Ensure survey targets are ints
    survey_df['CURRACTIVITYSTATUSI'] = survey_df['CURRACTIVITYSTATUSI'].astype('int64')
    survey_df['STATUSINEMPL_DETAIL'] = survey_df['STATUSINEMPL_DETAIL'].astype('int64')
    survey_df['IS_STUDENT'] = survey_df['IS_STUDENT'].astype('int64')

    # Drop rows with missing key vars in survey (safety)
    survey_df = survey_df.dropna(subset=[
        'AGE', 'SEX', 'home_municipality_id', 'district_id', 'canton_id',
        'CURRACTIVITYSTATUSI', 'STATUSINEMPL_DETAIL', 'IS_STUDENT'
    ])

    # In the population, use the *drawn* employment as features
    pop_df['CURRACTIVITYSTATUSI'] = pop_df['CURRACTIVITYSTATUSI'].astype('int64')
    pop_df['STATUSINEMPL_DETAIL'] = pop_df['STATUSINEMPL_DETAIL'].astype('int64')

    # =========================================================
    # 1. AGE BINS + CATEGORICAL CLEANING
    # =========================================================

    # Recreate age bins (safe even if already done)
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
    # 2. DESIGN MATRICES FOR STUDENT MODEL
    # =========================================================

    # Features: age bin, sex, activity status, job type
    student_feat_cols = ['AGE_BIN', 'SEX', 'CURRACTIVITYSTATUSI', 'STATUSINEMPL_DETAIL']

    X_student_survey = pd.get_dummies(survey_df[student_feat_cols], drop_first=False)
    X_student_pop    = pd.get_dummies(pop_df[student_feat_cols], drop_first=False)

    student_feature_cols = X_student_survey.columns
    X_student_pop = X_student_pop.reindex(columns=student_feature_cols, fill_value=0)

    y_student = survey_df['IS_STUDENT']
    w_student = survey_df['weight']

    # =========================================================
    # 3. MODEL STORAGE + THRESHOLDS
    # =========================================================

    MIN_MUNI_STUDENT     = 200
    MIN_DISTRICT_STUDENT = 200
    MIN_CANTON_STUDENT   = 200

    muni_student_models     = {}
    district_student_models = {}
    canton_student_models   = {}

    # number of cores to use (-1 = all)
    N_JOBS = -1

    # =========================================================
    # 4. HELPER TO FIT ONE STUDENT MODEL
    # =========================================================

    def fit_student_gbm(df_sub):
        """
        Fit GBM for IS_STUDENT on AGE_BIN + SEX + CURRACTIVITYSTATUSI + STATUSINEMPL_DETAIL.
        """
        X = pd.get_dummies(df_sub[student_feat_cols], drop_first=False)
        X = X.reindex(columns=student_feature_cols, fill_value=0)
        y = df_sub['IS_STUDENT'].astype('int64')
        w = df_sub['weight']

        clf = HistGradientBoostingClassifier(
            loss='log_loss',   # binary logistic loss
            max_depth=4,
            learning_rate=0.1,
            max_iter=100,
            random_state=42
        )
        clf.fit(X, y, sample_weight=w)
        return clf

    # =========================================================
    # 5. PARALLEL MUNICIPALITY-LEVEL STUDENT MODELS
    # =========================================================

    from joblib import Parallel, delayed

    def fit_muni_student(muni_id, df_m):
        """
        Fit student GBM for a single municipality.
        Returns (muni_id, model_or_None).
        """
        model = None
        if len(df_m) >= MIN_MUNI_STUDENT:
            model = fit_student_gbm(df_m)
        return muni_id, model

    muni_groups = list(survey_df.groupby('home_municipality_id'))
    n_muni = len(muni_groups)
    print(f"Fitting municipality student models in parallel for {n_muni} municipalities...")

    muni_student_results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=10)(
        delayed(fit_muni_student)(muni_id, df_m)
        for muni_id, df_m in muni_groups
    )

    print("Finished fitting municipality student models, collecting results...")

    for muni_id, model in muni_student_results:
        if model is not None:
            muni_student_models[muni_id] = model

    print(f"Trained {len(muni_student_models)} municipality student models.")

    # =========================================================
    # 6. PARALLEL DISTRICT-LEVEL STUDENT MODELS
    # =========================================================

    def fit_district_student(dist_id, df_d):
        """
        Fit student GBM for a single district.
        Returns (dist_id, model_or_None).
        """
        model = None
        if len(df_d) >= MIN_DISTRICT_STUDENT:
            model = fit_student_gbm(df_d)
        return dist_id, model

    district_groups = list(survey_df.groupby('district_id'))
    n_districts = len(district_groups)
    print(f"Fitting district student models in parallel for {n_districts} districts...")

    district_student_results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=10)(
        delayed(fit_district_student)(dist_id, df_d)
        for dist_id, df_d in district_groups
    )

    print("Finished fitting district student models, collecting results...")

    for dist_id, model in district_student_results:
        if model is not None:
            district_student_models[dist_id] = model

    print(f"Trained {len(district_student_models)} district student models.")

    # =========================================================
    # 7. CANTON-LEVEL STUDENT MODELS (SEQUENTIAL)
    # =========================================================

    canton_groups = list(survey_df.groupby('canton_id'))
    n_cantons = len(canton_groups)
    print(f"Fitting canton student models sequentially for {n_cantons} cantons...")

    for i, (canton_id, df_c) in enumerate(canton_groups, start=1):
        print(f"[Student Canton {i}/{n_cantons}] id={canton_id}, n={len(df_c)}")

        if len(df_c) < MIN_CANTON_STUDENT:
            raise ValueError(f"Not enough student data in canton {canton_id} "
                            f"({len(df_c)} obs). Lower MIN_CANTON_STUDENT or check data.")

        clf_c_student = fit_student_gbm(df_c)
        canton_student_models[canton_id] = clf_c_student

    print(f"Trained {len(canton_student_models)} canton student models.")

    # =========================================================
    # 8. STOCHASTIC DRAW HELPER
    # =========================================================

    def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
        rng = np.random.default_rng(seed)
        cum_proba = np.cumsum(proba_matrix, axis=1)
        r = rng.random(proba_matrix.shape[0])[:, None]
        chosen_idx = (r < cum_proba).argmax(axis=1)
        return classes[chosen_idx]

    # =========================================================
    # 9. STOCHASTIC ASSIGNMENT OF STUDENT_draw TO POPULATION
    # =========================================================

    pop_df['STUDENT_draw'] = np.nan

    SEED_STUDENT_BASE = 789

    muni_groups_pop = list(pop_df.groupby('home_municipality_id'))
    n_muni_pop = len(muni_groups_pop)
    print(f"Assigning STUDENT_draw across {n_muni_pop} municipalities...")

    for i, (muni_id, df_p) in enumerate(muni_groups_pop, start=1):
        idx = df_p.index
        dist_id   = df_p['district_id'].iloc[0]
        canton_id = df_p['canton_id'].iloc[0]

        if i % 100 == 0 or i == 1 or i == n_muni_pop:
            print(f"[Assign student] Municipality {i}/{n_muni_pop} id={muni_id}, n={len(df_p)}")

        X_p = X_student_pop.loc[idx]

        # Hierarchy: municipality -> district -> canton
        if muni_id in muni_student_models:
            clf_stu = muni_student_models[muni_id]
        elif dist_id in district_student_models:
            clf_stu = district_student_models[dist_id]
        else:
            clf_stu = canton_student_models[canton_id]

        proba_stu   = clf_stu.predict_proba(X_p)
        classes_stu = clf_stu.classes_.astype('int64')  # should be [0,1] or [1,0]

        seed_stu = SEED_STUDENT_BASE + (hash(muni_id) % 10000)
        stu_draw = draw_multinomial_from_proba(
            proba_stu, classes_stu, seed=seed_stu
        ).astype('int64')

        pop_df.loc[idx, 'STUDENT_draw'] = stu_draw

    # Final cast
    pop_df['STUDENT_draw'] = pop_df['STUDENT_draw'].astype('int64')

    print("Final STUDENT_draw distribution:")
    print(pop_df['STUDENT_draw'].value_counts(normalize=True))

    return pop_df
