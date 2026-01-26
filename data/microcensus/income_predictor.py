import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
# ---------------------------------------------------------
# helper: stochastic draw from class probabilities
# ---------------------------------------------------------
def draw_multinomial_from_proba(proba_matrix, classes, seed=None):
    rng = np.random.default_rng(seed)
    cum_proba = np.cumsum(proba_matrix, axis=1)
    r = rng.random(proba_matrix.shape[0])[:, None]
    chosen_idx = (r < cum_proba).argmax(axis=1)
    return classes[chosen_idx]


def configure(context):
    context.stage("data.microcensus.21.persons")
    context.stage("data.statpop.employment_v2")


def execute(context):
    # -------------------------------------------------------------------
    # CONFIG: model type + calibration switch
    # -------------------------------------------------------------------
    INCOME_MODEL = "catboost"          # "rf" or "gbm" r "catboost"
    USE_CALIBRATION = False       # <--- set to False to turn calibration OFF

    # -------------------------------------------------------------------
    # 0. LOAD DATA
    # -------------------------------------------------------------------
    survey_df = context.stage("data.microcensus.21.persons")
    pop_df = context.stage("data.statpop.employment_v2")
    survey_df = survey_df[survey_df["income_imputed"]== False] #keep only those that do not have imputed income

    # Map population job_position to survey coding
    mapping_pop_to_survey = {
        11: 11,
        12: 12,
        20: 20,
        31: 11,
        32: 12,
        41: 31,
        42: 32,
        43: 33,
        50: 40,
        60: 50,
        70: 60,
    }
    pop_df['job_position'] = pop_df['job_position'].map(mapping_pop_to_survey)

    # Boolean mask: who is an adult (>= 18) – used for N_adults
    adult_mask = pop_df['age'] >= 18
    pop_df['N_adults'] = (
        adult_mask.astype(int)
        .groupby(pop_df['household_id'])
        .transform('sum')
    )

    # -------------------------------------------------------------------
    # 1. PREP SURVEY: PERSONS + INCOME CLASSES
    # -------------------------------------------------------------------
    survey_df = survey_df.dropna(subset=[
        'age', 'sex',
        'household_size',
        'job_position',
        'N_adults',
        'canton_id',
        'income_class',
        'household_weight',
    ])

    # clean job_position: -99 -> 60, drop other negatives
    survey_df['job_position'] = survey_df['job_position'].astype(int)
    survey_df.loc[survey_df['job_position'] == -99, 'job_position'] = 60
    survey_df = survey_df[survey_df['job_position'] >= 0]

    survey_df['income_class'] = survey_df['income_class'].astype('int64')
    survey_df = survey_df[survey_df['income_class'] >= 0]

    # -------------------------------------------------------------------
    # 2. PREP POP: PICK ONE RANDOM REPRESENTATIVE PER HOUSEHOLD (age >= 6)
    # -------------------------------------------------------------------
    REP_MIN_AGE = 18  # you can try 15 or 18 as well

    pop_eligible = pop_df[pop_df['age'] >= REP_MIN_AGE].copy()
    rng = np.random.default_rng(12345)
    pop_eligible['rand'] = rng.random(len(pop_eligible))

    reps = (
        pop_eligible
        .sort_values(['household_id', 'rand'])
        .groupby('household_id')
        .head(1)
        .copy()
    )

    # -------------------------------------------------------------------
    # 3. AGE BINS + CATEGORICAL CLEANING
    # -------------------------------------------------------------------
    age_bins = [0, 15, 20, 25, 31, 41, 51, 66, 71, 200]
    age_labels = [
        '0-14', '15-19', '20-24', '25-30',
        '31-40', '41-50', '51-65', '66-70', '71+'
    ]

    for df in (survey_df, reps):
        df['age_bin'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )

    cat_cols = ['age_bin', 'sex', 'job_position', 'canton_id']
    for df in (survey_df, reps):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    for df in (survey_df, reps):
        df['household_size'] = df['household_size'].astype(float)
        df['N_adults'] = df['N_adults'].astype(float)

    # -------------------------------------------------------------------
    # 4. DESIGN MATRICES (GLOBAL)
    # -------------------------------------------------------------------
    feature_cols = [
        'age_bin',
        'sex',
        'job_position',
        'household_size',
        'N_adults',
        'canton_id'
    ]

    X_survey = pd.get_dummies(survey_df[feature_cols], drop_first=False)
    X_reps   = pd.get_dummies(reps[feature_cols], drop_first=False)

    global_feature_cols = X_survey.columns
    X_reps = X_reps.reindex(columns=global_feature_cols, fill_value=0)

    y = survey_df['income_class'].astype('int64')
    sample_weight = survey_df['household_weight'].astype(float)

    # -------------------------------------------------------------------
    # 5. FIT GLOBAL INCOME-CLASS MODEL (GBM OR RF)
    # -------------------------------------------------------------------
    def build_income_model(model_type: str):
        if model_type == "gbm":
            return HistGradientBoostingClassifier(
                loss='log_loss',
                max_depth=8,
                learning_rate=0.05,
                max_iter=400,
                min_samples_leaf=50,
                random_state=42
            )
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type in ("catboost", "cat"):
            
            return CatBoostClassifier(
                loss_function="MultiClass",
                iterations=1200,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=6.0,
                random_seed=42,
                verbose=False,
                bootstrap_type="Bernoulli",
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model_type={model_type}, use 'gbm', 'rf' or 'catboost'.")

    income_model = build_income_model(INCOME_MODEL)
        
    # CatBoost / sklearn models all handle numpy arrays; standardize dtype
    Xs = X_survey.to_numpy(dtype=float, copy=False)
    Xp = X_reps.to_numpy(dtype=float, copy=False)
    income_model.fit(Xs, y, sample_weight=sample_weight)
    print("Fitted global household income-class model on full survey using:", INCOME_MODEL)

    classes_cls = income_model.classes_.astype('int64')

    # -------------------------------------------------------------------
    # 5b. CALIBRATION SETUP (OPTIONAL)
    # -------------------------------------------------------------------
    # Base probabilities for reps (uncalibrated)
    proba_reps_raw = income_model.predict_proba(Xp)
    reps_job = reps['job_position'].values

    if USE_CALIBRATION:
        # 1) Target: survey shares by job_position x income_class (weighted, household_weight)
        survey_tmp = survey_df[['job_position', 'income_class', 'household_weight']].copy()

        true_mass = (
            survey_tmp
            .groupby(['job_position', 'income_class'], as_index=False)['household_weight']
            .sum()
            .rename(columns={'household_weight': 'w_true'})
        )

        true_tot = (
            true_mass
            .groupby('job_position', as_index=False)['w_true']
            .sum()
            .rename(columns={'w_true': 'w_tot_true'})
        )

        true_shares = true_mass.merge(true_tot, on='job_position', how='left')
        true_shares['share_true'] = true_shares['w_true'] / true_shares['w_tot_true']

        # 2) Model prediction: probabilities on reps (unweighted, one per household)
        proba_reps_df = pd.DataFrame(proba_reps_raw, columns=classes_cls)
        proba_reps_df['job_position'] = reps_job
        proba_reps_df['weight'] = 1.0

        proba_reps_long = proba_reps_df.melt(
            id_vars=['job_position', 'weight'],
            var_name='income_class',
            value_name='prob'
        )
        proba_reps_long['income_class'] = proba_reps_long['income_class'].astype('int64')
        proba_reps_long['w_prob'] = proba_reps_long['prob'] * proba_reps_long['weight']

        pred_mass = (
            proba_reps_long
            .groupby(['job_position', 'income_class'], as_index=False)['w_prob']
            .sum()
            .rename(columns={'w_prob': 'w_pred'})
        )

        pred_tot = (
            proba_reps_df
            .groupby('job_position', as_index=False)['weight']
            .sum()
            .rename(columns={'weight': 'w_tot_pred'})
        )

        pred_shares = pred_mass.merge(pred_tot, on='job_position', how='left')
        pred_shares['share_pred'] = pred_shares['w_pred'] / pred_shares['w_tot_pred']

        # 3) Compute alpha factors per (job_position, income_class)
        calib_df = pd.merge(
            true_shares[['job_position', 'income_class', 'share_true']],
            pred_shares[['job_position', 'income_class', 'share_pred']],
            on=['job_position', 'income_class'],
            how='outer'
        ).fillna(0.0)

        eps = 1e-6
        denom = np.where(calib_df['share_pred'] <= 0, eps, calib_df['share_pred'])
        calib_df['alpha'] = calib_df['share_true'] / denom
        calib_df['alpha'] = calib_df['alpha'].clip(0.25, 4.0)

        print("\n[CALIBRATION] Sample alpha factors (job_position x income_class):")
        print(
            calib_df.sort_values(['job_position', 'income_class'])
            .head(20)
            .to_string(index=False)
        )

        calib_lookup = calib_df.set_index(['job_position', 'income_class'])['alpha']

        # -------------------------------------------------------------------
        # 6. STOCHASTIC PREDICTION FOR REPS (CALIBRATED)
        # -------------------------------------------------------------------
        proba_reps_adj = proba_reps_raw.copy()

        unique_jobs = np.unique(reps_job)
        for job in unique_jobs:
            idx = np.where(reps_job == job)[0]
            if len(idx) == 0:
                continue

            alpha_vec = np.ones_like(classes_cls, dtype=float)
            for j, cls in enumerate(classes_cls):
                key = (job, int(cls))
                if key in calib_lookup.index:
                    alpha_vec[j] = calib_lookup.loc[key]
                else:
                    alpha_vec[j] = 1.0

            adj = proba_reps_raw[idx, :] * alpha_vec[None, :]
            row_sums = adj.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            proba_reps_adj[idx, :] = adj / row_sums

        proba_reps_final = proba_reps_adj
        print("\n[INFO] Using CALIBRATED probabilities for draws.")
    else:
        # no calibration
        proba_reps_final = proba_reps_raw
        print("\n[INFO] Using UNCALIBRATED model probabilities for draws.")

    SEED_INCOME = 2025
    cls_draw = draw_multinomial_from_proba(
        proba_reps_final,
        classes_cls,
        seed=SEED_INCOME
    ).astype('int64')

    reps['HH_INCOME_CLASS_hat'] = classes_cls[proba_reps_final.argmax(axis=1)]
    reps['HH_INCOME_CLASS_draw'] = cls_draw

    # -------------------------------------------------------------------
    # 7. MERGE BACK TO ALL PERSONS
    # -------------------------------------------------------------------
    hh_cols = ['household_id', 'HH_INCOME_CLASS_hat', 'HH_INCOME_CLASS_draw']
    hh_income_df = reps[hh_cols].copy()

    pop_df = pop_df.merge(hh_income_df, on='household_id', how='left')

    #there is a tiny group of hosuehold that do not have a single adult
    #assign income_class to 0 to these households.
    pop_df.loc[pop_df["N_adults"].eq(0), "HH_INCOME_CLASS_hat"] = 0
    pop_df.loc[pop_df["N_adults"].eq(0), "HH_INCOME_CLASS_draw"] = 0

    print("Household income class distribution (drawn), overall (persons):")
    print(pop_df['HH_INCOME_CLASS_draw'].value_counts(normalize=True))

    # -------------------------------------------------------------------
    # 8. CANTON-LEVEL COMPARISON (SURVEY persons vs POP reps)
    # -------------------------------------------------------------------
    survey_cant = survey_df[['canton_id', 'income_class', 'household_weight']].copy()
    survey_cant['canton_id'] = survey_cant['canton_id'].astype(str)

    survey_tot = (
        survey_cant.groupby('canton_id', as_index=False)['household_weight']
        .sum()
        .rename(columns={'household_weight': 'w_tot_survey'})
    )

    survey_mass = (
        survey_cant.groupby(['canton_id', 'income_class'], as_index=False)['household_weight']
        .sum()
        .rename(columns={'household_weight': 'w_true'})
    )

    survey_shares = survey_mass.merge(survey_tot, on='canton_id', how='left')
    survey_shares['share_true'] = survey_shares['w_true'] / survey_shares['w_tot_survey']

    pop_cant = reps[['canton_id', 'HH_INCOME_CLASS_draw']].copy()
    pop_cant['canton_id'] = pop_cant['canton_id'].astype(str)

    pop_tot = (
        pop_cant.groupby('canton_id', as_index=False).size()
        .rename(columns={'size': 'n_pop'})
    )

    pop_mass = (
        pop_cant.groupby(['canton_id', 'HH_INCOME_CLASS_draw'], as_index=False)
        .size()
        .rename(columns={'size': 'n_pred', 'HH_INCOME_CLASS_draw': 'income_class'})
    )

    pop_shares = pop_mass.merge(pop_tot, on='canton_id', how='left')
    pop_shares['share_pred'] = pop_shares['n_pred'] / pop_shares['n_pop']

    diff_df = pd.merge(
        survey_shares[['canton_id', 'income_class', 'share_true']],
        pop_shares[['canton_id', 'income_class', 'share_pred']],
        on=['canton_id', 'income_class'],
        how='outer'
    ).fillna(0.0)

    diff_df['share_diff'] = diff_df['share_pred'] - diff_df['share_true']
    diff_df['abs_diff'] = diff_df['share_diff'].abs()

    print("\nMean abs share diff across canton x income_class:",
          diff_df['abs_diff'].mean())

    print("\nCanton x income_class share comparison (survey vs population reps):")
    print(diff_df.sort_values(['canton_id', 'income_class'])
              [['canton_id', 'income_class', 'share_true', 'share_pred', 'share_diff']])

    CANTON_TO_INSPECT = '1'
    mask = diff_df['canton_id'] == CANTON_TO_INSPECT
    if mask.any():
        print(f"\nDetailed comparison for canton {CANTON_TO_INSPECT}:")
        print(diff_df.loc[mask, ['income_class', 'share_true', 'share_pred', 'share_diff']]
                    .sort_values('income_class')
                    .to_string(index=False))

    # -------------------------------------------------------------------
    # 9. JOB_POSITION-LEVEL COMPARISON: AVERAGE INCOME_CLASS
    # -------------------------------------------------------------------
    survey_job = (
        survey_df
        .groupby('job_position')
        .apply(lambda g: np.average(g['income_class'], weights=g['household_weight']))
        .reset_index(name='avg_income_class_true')
    )

    reps_job = (
        reps
        .groupby('job_position', as_index=False)['HH_INCOME_CLASS_draw']
        .mean()
        .rename(columns={'HH_INCOME_CLASS_draw': 'avg_income_class_pred'})
    )

    job_compare = pd.merge(
        survey_job,
        reps_job,
        on='job_position',
        how='inner'
    )

    suffix = "calibrated" if USE_CALIBRATION else "uncalibrated"
    job_compare['avg_diff'] = job_compare['avg_income_class_pred'] - job_compare['avg_income_class_true']

    print(f"\nAverage income_class by job_position (survey vs population reps, {suffix}):")
    print(job_compare.sort_values('job_position')
                 [['job_position', 'avg_income_class_true', 'avg_income_class_pred', 'avg_diff']]
                 .to_string(index=False))

    # -------------------------------------------------------------------
    # 10. FULL income_class DISTRIBUTION FOR ONE job_position
    # -------------------------------------------------------------------
    JOB_TO_INSPECT = "31"  # job_position is stored as string after preprocessing

    survey_job_mask = survey_df['job_position'] == JOB_TO_INSPECT
    survey_job_sub = survey_df[survey_job_mask].copy()

    if len(survey_job_sub) > 0:
        w_tot = survey_job_sub['household_weight'].sum()
        survey_dist = (
            survey_job_sub
            .groupby('income_class', as_index=False)['household_weight']
            .sum()
            .rename(columns={'household_weight': 'w_true'})
        )
        survey_dist['share_true'] = survey_dist['w_true'] / w_tot
    else:
        survey_dist = pd.DataFrame(columns=['income_class', 'w_true', 'share_true'])

    reps_job_mask = reps['job_position'] == JOB_TO_INSPECT
    reps_job_sub = reps[reps_job_mask].copy()

    if len(reps_job_sub) > 0:
        n_rep = len(reps_job_sub)
        reps_dist = (
            reps_job_sub
            .groupby('HH_INCOME_CLASS_draw', as_index=False)
            .size()
            .rename(columns={'size': 'n_pred', 'HH_INCOME_CLASS_draw': 'income_class'})
        )
        reps_dist['share_pred'] = reps_dist['n_pred'] / n_rep
    else:
        reps_dist = pd.DataFrame(columns=['income_class', 'n_pred', 'share_pred'])

    dist_compare = pd.merge(
        survey_dist[['income_class', 'share_true']],
        reps_dist[['income_class', 'share_pred']],
        on='income_class',
        how='outer'
    ).fillna(0.0)

    dist_compare['share_diff'] = dist_compare['share_pred'] - dist_compare['share_true']

    print(f"\nFull income_class distribution for job_position {JOB_TO_INSPECT} ({suffix}):")
    print(
        dist_compare.sort_values('income_class')
        [['income_class', 'share_true', 'share_pred', 'share_diff']]
        .to_string(index=False)
    )
    pop_df = pop_df.rename(columns={"HH_INCOME_CLASS_draw": "income_class"})
    return pop_df
