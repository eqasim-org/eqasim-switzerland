import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split  # <-- ADD
import logging

logger = logging.getLogger("synpp")
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
    context.stage("data.constants")
    context.stage("data.structural_survey.structural_survey")
    context.stage("data.statpop.statpop")

def execute(context):
    c = context.stage("data.constants")

    # -------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------
    ACTIVITY_MODEL = "catboost"   # "gbm" or "rf" or "catboost"
    JOB_MODEL      = "gbm"        # "gbm" or "rf"

    CHUNK_SIZE = 25_000
    SEED_ACTIVITY = 123
    SEED_JOB      = 456

    CANTON_FOR_ANALYSIS = 22  # e.g. "1"

    # Early stopping config (CatBoost only)  <-- ADD
    VALID_FRAC = 0.10
    EARLY_STOP_ROUNDS = 200

    # -------------------------------------------------------------------
    # 0. LOAD
    # -------------------------------------------------------------------
    survey_df = context.stage("data.structural_survey.structural_survey")
    pop_df = context.stage("data.statpop.statpop")

    survey_df['employed']     = survey_df['employed'].astype('int32')
    survey_df['job_position'] = survey_df['job_position'].astype('int32')

    logger.info("Survey employed weighted totals:")
    logger.info(survey_df.groupby("employed")["weight"].sum())

    # -------------------------------------------------------------------
    # 1. CLEAN + AGE BINS
    # -------------------------------------------------------------------
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'weight', 'municipality_type',
    ]).copy()

    age_bins = [0, 15, 17, 20, 24, 31, 41, 51, 66, 71, 200]
    age_labels = [
        '0-14', '15-16', '17-19', '20-24', '25-30',
        '31-40', '41-50', '51-65', '66-70', '71+'
    ]

    for df in (survey_df, pop_df):
        df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    id_cols = ['home_municipality_id', 'district_id', 'canton_id']

    for df in (survey_df, pop_df):
        for col in id_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    cat_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id', 'municipality_type']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # -------------------------------------------------------------------
    # 2. ONE-HOT DESIGN MATRIX FOR SURVEY
    # -------------------------------------------------------------------
    feature_cols = ['age', 'age_bin', 'sex', 'nationality', 'municipality_type', 'district_id', 'canton_id']

    X_survey = pd.get_dummies(survey_df[feature_cols], drop_first=False)
    X_survey['age'] = survey_df['age'].astype(float).to_numpy()
    X_survey = X_survey.astype(float)

    global_feature_cols = X_survey.columns
    w = survey_df['weight'].astype(float).to_numpy()

    # -------------------------------------------------------------------
    # 3. FIT GLOBAL ACTIVITY MODEL
    # -------------------------------------------------------------------
    def build_model(model_type: str):
        if model_type == "gbm":
            return HistGradientBoostingClassifier(
                loss='log_loss',
                max_depth=3,
                learning_rate=0.1,
                max_iter=100,
                random_state=42
            )
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1
            )
        elif model_type in ("catboost", "cat"):
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=4000,
                learning_rate=0.01,
                depth=10,
                l2_leaf_reg=6.0,
                random_seed=42,
                verbose=200,
                bootstrap_type="Bernoulli",
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}, use 'gbm' or 'rf'.")

    y_act = survey_df['employed'].astype('int64').to_numpy()
    # masks for survey split
    youth_mask_s = (survey_df["age"].to_numpy() >= 15) & (survey_df["age"].to_numpy() <= 23)
    adult_mask_s = (survey_df["age"].to_numpy() >= 24)

    def fit_activity_model(X, y, w, model_name: str):
        model = build_model(ACTIVITY_MODEL)

        if ACTIVITY_MODEL in ("catboost", "cat"):
            idx = np.arange(len(y))
            train_idx, val_idx = train_test_split(
                idx,
                test_size=VALID_FRAC,
                random_state=42,
                stratify=y
            )

            X_tr, y_tr, w_tr = X.iloc[train_idx], y[train_idx], w[train_idx]
            X_va, y_va, w_va = X.iloc[val_idx], y[val_idx], w[val_idx]

            train_pool = Pool(X_tr, y_tr, weight=w_tr)
            val_pool   = Pool(X_va, y_va, weight=w_va)

            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                early_stopping_rounds=EARLY_STOP_ROUNDS
            )
            logger.info(f"Fitted {model_name} activity model | best_iter={model.get_best_iteration()} | trees={model.tree_count_}")
        else:
            model.fit(X, y, sample_weight=w)
            logger.info(f"Fitted {model_name} activity model using: {ACTIVITY_MODEL}")

        return model
    # --- fit youth model (15-23)

    y_features = ['age', 'sex', 'canton_id']
    X_survey_y = pd.get_dummies(survey_df[y_features], drop_first=False)
    X_survey_y['age'] = X_survey_y['age'].astype(float).to_numpy()
    X_survey_y = X_survey_y.astype(float)
    X_y = X_survey_y.loc[youth_mask_s]
    y_y = y_act[youth_mask_s]
    w_y = w[youth_mask_s]
    act_model_y = fit_activity_model(X_y, y_y, w_y, model_name="YOUTH(15-23)")

    # --- fit adult model (24+)
    X_a = X_survey.loc[adult_mask_s]
    y_a = y_act[adult_mask_s]
    w_a = w[adult_mask_s]
    act_model_a = fit_activity_model(X_a, y_a, w_a, model_name="ADULT(24+)")

    # -------------------------------------------------------------------
    # 4. FIT GLOBAL JOB MODEL
    # -------------------------------------------------------------------
    df_emp = survey_df[survey_df['employed'] == c.EMPLOYED]
    X_job_survey = X_survey.loc[df_emp.index]
    y_job = df_emp['job_position'].astype('int64').to_numpy()
    w_job = df_emp['weight'].astype(float).to_numpy()

    job_model = build_model(JOB_MODEL)
    job_model.fit(X_job_survey, y_job, sample_weight=w_job)
    logger.info(f"Fitted global job model using: {JOB_MODEL}")

    # -------------------------------------------------------------------
    # 5. POPULATION PREDICTION IN CHUNKS
    # -------------------------------------------------------------------
    n = len(pop_df)
    employed_out = np.empty(n, dtype=int)
    job_out      = np.empty(n, dtype=int)
    # activity model classes (assume both models saw same label set; we still handle safely)
    classes_act_y = act_model_y.classes_.astype('int64')
    classes_act_a = act_model_a.classes_.astype('int64')
    classes_job = job_model.classes_.astype('int64')

    logger.info(f"Predicting population in chunks: n={n:,}, CHUNK_SIZE={CHUNK_SIZE:,}")
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = pop_df.iloc[start:end]

        X_chunk = pd.get_dummies(chunk[feature_cols], drop_first=False)
        X_chunk['age'] = chunk['age'].astype(float).to_numpy()
        X_chunk = X_chunk.reindex(columns=global_feature_cols, fill_value=0).astype(float)

        # route by age
        age_arr = chunk["age"].to_numpy()
        youth_mask = (age_arr >= 15) & (age_arr <= 23)
        adult_mask = (age_arr >= 24)

        act_draw = np.empty(end - start, dtype=int)

        # youth predictions
        if youth_mask.any():
            X_yc = X_chunk.loc[youth_mask]
            proba_y = act_model_y.predict_proba(X_yc)
            draw_y = draw_multinomial_from_proba(
                proba_y, classes_act_y, seed=SEED_ACTIVITY + start + 1
            ).astype(int)
            act_draw[youth_mask] = draw_y
            del X_yc, proba_y

        # adult predictions
        if adult_mask.any():
            X_ac = X_chunk.loc[adult_mask]
            proba_a = act_model_a.predict_proba(X_ac)
            draw_a = draw_multinomial_from_proba(
                proba_a, classes_act_a, seed=SEED_ACTIVITY + start + 2
            ).astype(int)
            act_draw[adult_mask] = draw_a
            del X_ac, proba_a

        employed_out[start:end] = act_draw

        # job draw (unchanged)
        job_chunk = np.empty(end - start, dtype=int)
        job_chunk[act_draw == c.UNEMPLOYED] = 60
        job_chunk[act_draw == c.INACTIVE] = 70

        emp_mask_local = (act_draw == c.EMPLOYED)
        if emp_mask_local.any():
            X_emp = X_chunk.loc[emp_mask_local]
            proba_job = job_model.predict_proba(X_emp)
            job_draw_emp = draw_multinomial_from_proba(
                proba_job, classes_job, seed=SEED_JOB + start
            ).astype(int)
            job_chunk[emp_mask_local] = job_draw_emp
            del X_emp, proba_job

        job_out[start:end] = job_chunk

        del X_chunk
        gc.collect()

        if (start // CHUNK_SIZE) % 20 == 0:
            logger.info(f"  ... processed {end:,}/{n:,}")

    pop_df['employed']     = employed_out
    pop_df['job_position'] = job_out

    pop_df.loc[pop_df['age'] < 15, 'employed'] = c.INACTIVE
    pop_df.loc[pop_df['age'] < 15, 'job_position'] = 70

    logger.info("Final employed distribution:")
    logger.info(pop_df[pop_df['age'] > 14]['employed'].value_counts(normalize=True))

    # -------------------------------------------------------------------
    # 6. DIAGNOSTICS
    # -------------------------------------------------------------------
    if CANTON_FOR_ANALYSIS is None:
        survey_diag = survey_df
        pop_diag = pop_df
        logger.info("\n[DIAGNOSTIC] Employed-by-age analysis for ALL cantons (global)")
    else:
        canton_key = str(CANTON_FOR_ANALYSIS)
        survey_diag = survey_df[survey_df['canton_id'] == canton_key]
        pop_diag = pop_df[pop_df['canton_id'].astype(str) == canton_key]
        logger.info(f"\n[DIAGNOSTIC] Employed-by-age analysis restricted to canton_id={canton_key}")

    # (A) share employed==EMPLOYED by age_bin (survey weighted vs pop unweighted)
    survey_rate = (
        survey_diag.groupby('age_bin')
        .apply(lambda g: np.average((g['employed'] == c.EMPLOYED).astype(float), weights=g['weight']))
        .reset_index(name='share_employed1_survey')
    )

    pop_rate = (
        pop_diag.groupby('age_bin')
        .apply(lambda g: (g['employed'] == c.EMPLOYED).mean())
        .reset_index(name='share_employed1_pop')
    )

    rate_compare = pd.merge(survey_rate, pop_rate, on='age_bin', how='outer').fillna(0.0)
    rate_compare['diff_pop_minus_survey'] = rate_compare['share_employed1_pop'] - rate_compare['share_employed1_survey']

    logger.info("\nShare employed==EMPLOYED by age_bin (survey vs population):")
    logger.info(rate_compare.sort_values('age_bin').to_string(index=False))

    # (B) full distribution employed(1/2/3) by age_bin
    survey_mass = (
        survey_diag.groupby(['age_bin', 'employed'], as_index=False)['weight']
        .sum().rename(columns={'weight': 'w'})
    )
    survey_tot = survey_mass.groupby('age_bin', as_index=False)['w'].sum().rename(columns={'w': 'w_tot'})
    survey_mass = survey_mass.merge(survey_tot, on='age_bin', how='left')
    survey_mass['share_survey'] = survey_mass['w'] / survey_mass['w_tot']

    pop_mass = (
        pop_diag.groupby(['age_bin', 'employed'], as_index=False)
        .size().rename(columns={'size': 'n'})
    )
    pop_tot = pop_mass.groupby('age_bin', as_index=False)['n'].sum().rename(columns={'n': 'n_tot'})
    pop_mass = pop_mass.merge(pop_tot, on='age_bin', how='left')
    pop_mass['share_pop'] = pop_mass['n'] / pop_mass['n_tot']

    dist_compare = pd.merge(
        survey_mass[['age_bin', 'employed', 'share_survey']],
        pop_mass[['age_bin', 'employed', 'share_pop']],
        on=['age_bin', 'employed'],
        how='outer'
    ).fillna(0.0)
    dist_compare['share_diff'] = dist_compare['share_pop'] - dist_compare['share_survey']

    logger.info("\nFull employed distribution by age_bin (survey vs population):")
    logger.info(dist_compare.sort_values(['age_bin', 'employed']).to_string(index=False))

    # Drop helper features created for model fitting/diagnostics.
    pop_df = pop_df.drop(columns=["age_bin"], errors="ignore")

    return pop_df
