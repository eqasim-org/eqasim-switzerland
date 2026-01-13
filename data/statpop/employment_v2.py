import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import gc

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
    # CONFIG
    # -------------------------------------------------------------------
    ACTIVITY_MODEL = "gbm"   # "gbm" or "rf"
    JOB_MODEL      = "gbm"   # "gbm" or "rf"

    # IMPORTANT: chunk size controls memory. Start conservative on HPC.
    CHUNK_SIZE = 25_000

    SEED_ACTIVITY = 123
    SEED_JOB      = 456

    # Diagnostics: None = all cantons; or e.g. "1"
    CANTON_FOR_ANALYSIS = None  # e.g. "1"

    # -------------------------------------------------------------------
    # 0. LOAD
    # -------------------------------------------------------------------
    survey_df = context.stage("data.structural_survey.structural_survey")

    pop_df = context.stage("data.statpop.statpop")

    survey_df['employed']     = survey_df['employed'].astype('int32')
    survey_df['job_position'] = survey_df['job_position'].astype('int32')

    # Survey sanity
    print("Survey employed weighted totals:")
    print(survey_df.groupby("employed")["weight"].sum())

    # -------------------------------------------------------------------
    # 1. CLEAN + AGE BINS
    # -------------------------------------------------------------------
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'weight'
    ]).copy()

    age_bins = [0, 15, 20, 25, 31, 41, 51, 66, 71, 200]
    age_labels = [
        '0-14', '15-19', '20-24', '25-30',
        '31-40', '41-50', '51-65', '66-70', '71+'
    ]

    for df in (survey_df, pop_df):
        df['age_bin'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    cat_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # -------------------------------------------------------------------
    # 2. ONE-HOT DESIGN MATRIX FOR SURVEY
    # -------------------------------------------------------------------
    feature_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id']

    X_survey = pd.get_dummies(survey_df[feature_cols], drop_first=False)
    # reduce memory without changing values
    X_survey = X_survey.astype(np.float32)

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
                n_jobs=1  # HPC-safe
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}, use 'gbm' or 'rf'.")

    y_act = survey_df['employed'].astype('int64').to_numpy()

    act_model = build_model(ACTIVITY_MODEL)
    act_model.fit(X_survey, y_act, sample_weight=w)
    print("Fitted global activity model using:", ACTIVITY_MODEL)

    # -------------------------------------------------------------------
    # 4. FIT GLOBAL JOB MODEL
    # -------------------------------------------------------------------
    df_emp = survey_df[survey_df['employed'] == 1]
    X_job_survey = X_survey.loc[df_emp.index]
    y_job = df_emp['job_position'].astype('int64').to_numpy()
    w_job = df_emp['weight'].astype(float).to_numpy()

    job_model = build_model(JOB_MODEL)
    job_model.fit(X_job_survey, y_job, sample_weight=w_job)
    print("Fitted global job model using:", JOB_MODEL)

    # -------------------------------------------------------------------
    # 5. POPULATION PREDICTION IN CHUNKS (this is the cluster-safe change)
    #     - DOES NOT change outcomes vs building full X_pop
    # -------------------------------------------------------------------
    n = len(pop_df)
    employed_out = np.empty(n, dtype=np.int16)
    job_out      = np.empty(n, dtype=np.int16)
    classes_act = act_model.classes_.astype('int64')
    classes_job = job_model.classes_.astype('int64')

    print(f"Predicting population in chunks: n={n:,}, CHUNK_SIZE={CHUNK_SIZE:,}")
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = pop_df.iloc[start:end]

        # one-hot exactly like old code, but only for this chunk
        X_chunk = pd.get_dummies(chunk[feature_cols], drop_first=False)
        X_chunk = X_chunk.reindex(columns=global_feature_cols, fill_value=0).astype(np.float32)

        # activity draw
        proba_act = act_model.predict_proba(X_chunk)
        act_draw = draw_multinomial_from_proba(
            proba_act, classes_act, seed=SEED_ACTIVITY + start
        ).astype(np.int16)
        employed_out[start:end] = act_draw

        # job draw
        job_chunk = np.empty(end - start, dtype=np.int16)
        job_chunk[act_draw == 2] = 60
        job_chunk[act_draw == 3] = 70

        emp_mask_local = (act_draw == 1)
        if emp_mask_local.any():
            X_emp = X_chunk.loc[emp_mask_local]
            proba_job = job_model.predict_proba(X_emp)
            job_draw_emp = draw_multinomial_from_proba(
                proba_job, classes_job, seed=SEED_JOB + start
            ).astype(np.int16)
            job_chunk[emp_mask_local] = job_draw_emp

        job_out[start:end] = job_chunk

        # cleanup chunk memory
        del X_chunk, proba_act
        if emp_mask_local.any():
            del X_emp, proba_job
        gc.collect()

        if (start // CHUNK_SIZE) % 20 == 0:
            print(f"  ... processed {end:,}/{n:,}")

    pop_df['employed']     = employed_out
    pop_df['job_position'] = job_out

    # enforce <15 as before
    pop_df.loc[pop_df['age'] < 15, 'employed'] = 3
    pop_df.loc[pop_df['age'] < 15, 'job_position'] = 70

    print("Final employed distribution:")
    print(pop_df['employed'].value_counts(normalize=True))

    # -------------------------------------------------------------------
    # 6. DIAGNOSTICS
    # -------------------------------------------------------------------
    if CANTON_FOR_ANALYSIS is None:
        survey_diag = survey_df
        pop_diag = pop_df
        print("\n[DIAGNOSTIC] Employed-by-age analysis for ALL cantons (global)")
    else:
        canton_key = str(CANTON_FOR_ANALYSIS)
        survey_diag = survey_df[survey_df['canton_id'] == canton_key]
        pop_diag = pop_df[pop_df['canton_id'].astype(str) == canton_key]
        print(f"\n[DIAGNOSTIC] Employed-by-age analysis restricted to canton_id={canton_key}")

    # (A) share employed==1 by age_bin (survey weighted vs pop unweighted)
    survey_rate = (
        survey_diag.groupby('age_bin')
        .apply(lambda g: np.average((g['employed'] == 1).astype(float), weights=g['weight']))
        .reset_index(name='share_employed1_survey')
    )

    pop_rate = (
        pop_diag.groupby('age_bin')
        .apply(lambda g: (g['employed'] == 1).mean())
        .reset_index(name='share_employed1_pop')
    )

    rate_compare = pd.merge(survey_rate, pop_rate, on='age_bin', how='outer').fillna(0.0)
    rate_compare['diff_pop_minus_survey'] = rate_compare['share_employed1_pop'] - rate_compare['share_employed1_survey']

    print("\nShare employed==1 by age_bin (survey vs population):")
    print(rate_compare.sort_values('age_bin').to_string(index=False))

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

    print("\nFull employed distribution by age_bin (survey vs population):")
    print(dist_compare.sort_values(['age_bin', 'employed']).to_string(index=False))

    return pop_df
