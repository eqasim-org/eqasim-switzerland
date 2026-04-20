import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from joblib import Parallel, delayed
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
def configure(context):
    context.config("data_path")
    context.stage("synthesis.population.models.income")
    context.stage("data.structural_survey.structural_survey")
def plot_weighted_student_share_by_age(survey_df, survey_weight_col="weight"):
        def weighted_mean(x, w):
            return np.average(x, weights=w) if len(x) > 0 else np.nan

        age_profile = (
            survey_df
            .groupby("age")
            .apply(lambda g: weighted_mean(g["is_student"], g[survey_weight_col]))
            .reset_index(name="share_student")
            .sort_values("age")
        )

        plt.figure(figsize=(10, 6))
        plt.plot(age_profile["age"], age_profile["share_student"], marker="o")
        plt.xlabel("Age")
        plt.ylabel("Weighted share of students")
        plt.title("Weighted share of students by age in survey data")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
def execute(context):

    # =========================================================
    # CONFIG FOR ANALYSIS
    # =========================================================
    # Examples:
    # None            -> diagnostics for all cantons
    # 25              -> diagnostics for one canton
    # [1, 19, 25]     -> diagnostics for multiple cantons
    CANTONS_FOR_ANALYSIS = [1, 22, 25]

    # choose model type: "gbm" or "rf" or "catboost"
    STUDENT_MODEL = "catboost"   # <-- allow "catboost" too

    # survey weight column
    SURVEY_WEIGHT_COL = "weight"

    CHUNK_SIZE = 50_000

    # =========================================================
    # 0. PREP: CLEAN + ALIGN EMPLOYMENT INFO
    # =========================================================
    survey_df = context.stage("data.structural_survey.structural_survey")
    pop_df    = context.stage("synthesis.population.models.income")

    survey_df['employed']     = survey_df['employed'].astype('int64')
    survey_df['job_position'] = survey_df['job_position'].astype('int64')
    survey_df['is_student']   = survey_df['is_student'].astype('int64')
    survey_df[SURVEY_WEIGHT_COL] = survey_df[SURVEY_WEIGHT_COL].astype(float)
    #plot_weighted_student_share_by_age(survey_df, survey_weight_col=SURVEY_WEIGHT_COL)
    survey_df = survey_df.dropna(subset=[
        'age', 'sex', 'home_municipality_id', 'district_id', 'canton_id',
        'employed', 'job_position', 'is_student', 'municipality_type',
        SURVEY_WEIGHT_COL
    ])

    pop_df['employed']     = pop_df['employed'].astype('int64')
    pop_df['job_position'] = pop_df['job_position'].astype('int64')

    survey_df['age_sq'] = survey_df['age'] ** 2
    pop_df['age_sq'] = pop_df['age'] ** 2

    survey_df['is_school_age'] = ((survey_df['age'] >= 15) & (survey_df['age'] <= 24)).astype(int)
    pop_df['is_school_age'] = ((pop_df['age'] >= 15) & (pop_df['age'] <= 24)).astype(int)
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
    id_cols = ['home_municipality_id', 'district_id', 'canton_id']

    for df in (survey_df, pop_df):
        for col in id_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    cat_cols = ['age_bin', 'sex', 'home_municipality_id', 'district_id', 'canton_id', 'municipality_type']
    for df in (survey_df, pop_df):
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna('Missing')

    # =========================================================
    # 2. DESIGN MATRICES FOR STUDENT MODEL
    # =========================================================

    student_feat_cols = [
        'age',
        'age_sq',
        #'age_bin',
        'sex',
        #'nationality',
        'municipality_type',
        'employed',
        'district_id',
        'canton_id'
    ]

    X_student_survey = pd.get_dummies(survey_df[student_feat_cols], drop_first=False)

    # ensure 'age' is numeric after get_dummies (robust)
    X_student_survey['age'] = survey_df['age'].astype(float).to_numpy()

    # reduce memory
    X_student_survey = X_student_survey.astype(float)

    student_feature_cols = X_student_survey.columns

    y_student = survey_df['is_student'].astype('int64')
    w_student = survey_df[SURVEY_WEIGHT_COL].astype(float)

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
            iterations=3200,
            learning_rate=0.04,
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

    print("Fitted global student model using:", STUDENT_MODEL)
    print(f"Used survey weight column for training: {SURVEY_WEIGHT_COL}")

    # =========================================================
    # 4. STOCHASTIC DRAW HELPER
    # =========================================================
    def plot_student_share_by_agebin_and_district(
        survey_df,
        pop_df,
        survey_weight_col,
        district_col='district_id',
        agebin_col='age_bin',
        target_col='is_student',
        canton_keys=None,
        annotate=False
    ):
        def weighted_mean(x, w):
            return np.average(x, weights=w) if len(x) > 0 else np.nan

        def get_pop_weight_col(df):
            for cand in ['weight', 'person_weight', 'household_weight']:
                if cand in df.columns:
                    return cand
            return None

        survey_plot = survey_df.copy()
        pop_plot = pop_df.copy()

        # optional canton filter
        if canton_keys is not None:
            canton_keys = [str(x) for x in canton_keys]
            survey_plot = survey_plot[survey_plot['canton_id'].astype(str).isin(canton_keys)]
            pop_plot = pop_plot[pop_plot['canton_id'].astype(str).isin(canton_keys)]

        # population side: keep same restriction as your existing diagnostics
        pop_plot = pop_plot[pop_plot['age'] > 14]

        pop_weight_col = get_pop_weight_col(pop_plot)

        survey_grp = (
            survey_plot
            .groupby([district_col, agebin_col])
            .apply(lambda g: weighted_mean(g[target_col], g[survey_weight_col]))
            .reset_index(name='share_student_survey')
        )

        if pop_weight_col is not None:
            pop_grp = (
                pop_plot
                .groupby([district_col, agebin_col])
                .apply(lambda g: weighted_mean(g[target_col], g[pop_weight_col]))
                .reset_index(name='share_student_pop')
            )
        else:
            pop_grp = (
                pop_plot
                .groupby([district_col, agebin_col])[target_col]
                .mean()
                .reset_index(name='share_student_pop')
            )

        compare = pd.merge(
            survey_grp,
            pop_grp,
            on=[district_col, agebin_col],
            how='inner'
        ).dropna(subset=['share_student_survey', 'share_student_pop'])

        agebin_order = [x for x in survey_df[agebin_col].astype(str).unique()]
        agebin_order = sorted(agebin_order)

        for age_bin in agebin_order:
            sub = compare[compare[agebin_col].astype(str) == str(age_bin)].copy()
            r2 = r2_score(sub['share_student_survey'], sub['share_student_pop'])
            if sub.empty:
                continue

            plt.figure(figsize=(7, 7))
            plt.scatter(
                sub['share_student_survey'],
                sub['share_student_pop'],
                alpha=0.75
            )

            # 45-degree reference line
            plt.plot([0, 1], [0, 1], linestyle='--')

            if annotate:
                for _, row in sub.iterrows():
                    plt.text(
                        row['share_student_survey'],
                        row['share_student_pop'],
                        str(row[district_col]),
                        fontsize=8,
                        alpha=0.8
                    )

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Survey share of students')
            plt.ylabel('Population share of students')
            plt.title(f'Student share by district_id | age_bin = {age_bin}')
            plt.grid(alpha=0.3)
            plt.text(
                0.05, 0.95,
                f"R² = {r2:.3f}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.2)
            )
            plt.tight_layout()
            plt.show()
    
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
    stu_out = np.empty(n, dtype=int)

    print(f"Predicting students in chunks: n={n:,}, CHUNK_SIZE={CHUNK_SIZE:,}")

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = pop_df.iloc[start:end]

        X_chunk = pd.get_dummies(chunk[student_feat_cols], drop_first=False)

        # ✅ ensure numeric 'age' exists in chunk too
        X_chunk['age'] = chunk['age'].astype(float).to_numpy()

        X_chunk = X_chunk.reindex(columns=student_feature_cols, fill_value=0).astype(float)

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
            print(f"  ... processed {end:,}/{n:,}")

    pop_df['STUDENT_draw'] = stu_out

    pop_df['STUDENT_draw'] = pop_df['STUDENT_draw'].astype('int64')
    pop_df = pop_df.rename(columns={"STUDENT_draw": "is_student"})
    pop_df.loc[pop_df['age'] < 15, 'is_student'] = 1
    # plot_student_share_by_agebin_and_district(
    #     survey_df=survey_df,
    #     pop_df=pop_df,
    #     survey_weight_col=SURVEY_WEIGHT_COL,
    #     canton_keys=None,   # or None for all cantons
    #     annotate=True                      # set True if you want district_id labels on points
    # )
    print("Final STUDENT distribution in population:")
    print(pop_df['is_student'].value_counts(normalize=True))

    # =========================================================
    # 6. DIAGNOSTICS: COMPARE STUDENT RATES
    #    - by age_bin
    #    - by municipality_type
    #    - for all selected cantons combined
    #    - and for each selected canton separately
    # =========================================================

    def weighted_mean(x, w):
        return np.average(x, weights=w) if len(x) > 0 else np.nan

    def get_pop_weight_col(df):
        for cand in ['weight', 'person_weight', 'household_weight']:
            if cand in df.columns:
                return cand
        return None

    def build_compare_table(survey_sub, pop_sub, group_col, label):
        survey_grp = (
            survey_sub
            .groupby(group_col)
            .apply(lambda g: weighted_mean(g['is_student'], g[SURVEY_WEIGHT_COL]))
            .reset_index(name='share_student_survey')
        )

        pop_weight_col = get_pop_weight_col(pop_sub)

        if pop_weight_col is not None:
            pop_grp = (
                pop_sub
                .groupby(group_col)
                .apply(lambda g: weighted_mean(g['is_student'], g[pop_weight_col]))
                .reset_index(name='share_student_pop')
            )
        else:
            pop_grp = (
                pop_sub
                .groupby(group_col)['is_student']
                .mean()
                .reset_index(name='share_student_pop')
            )

        compare = pd.merge(
            survey_grp,
            pop_grp,
            on=group_col,
            how='outer'
        )

        compare['share_student_survey'] = compare['share_student_survey'].fillna(0.0)
        compare['share_student_pop']    = compare['share_student_pop'].fillna(0.0)
        compare['diff_pop_minus_survey'] = (
            compare['share_student_pop'] - compare['share_student_survey']
        )

        print(f"\nShare of students by {label} (survey vs population):")
        print(compare.sort_values(group_col).to_string(index=False))

    # normalize canton selection
    if CANTONS_FOR_ANALYSIS is None:
        canton_keys = None
    elif isinstance(CANTONS_FOR_ANALYSIS, (list, tuple, set, np.ndarray, pd.Series)):
        canton_keys = [str(x) for x in CANTONS_FOR_ANALYSIS]
    else:
        canton_keys = [str(CANTONS_FOR_ANALYSIS)]

    if canton_keys is None:
        survey_diag = survey_df
        pop_diag    = pop_df
        pop_diag = pop_diag[pop_diag['age']>14]
        print("\n[DIAGNOSTIC] Analysis for ALL cantons (global)")

        build_compare_table(survey_diag, pop_diag, 'age_bin', 'age_bin')
        build_compare_table(survey_diag, pop_diag, 'municipality_type', 'municipality_type')

    else:
        survey_diag = survey_df[survey_df['canton_id'].isin(canton_keys)]
        pop_diag    = pop_df[pop_df['canton_id'].astype(str).isin(canton_keys)]
        pop_diag = pop_diag[pop_diag['age']>14]
        print(f"\n[DIAGNOSTIC] Analysis restricted to canton_id in {canton_keys}")

        # combined across all selected cantons
        print("\n[DIAGNOSTIC] Combined across selected cantons")
        build_compare_table(survey_diag, pop_diag, 'age_bin', 'age_bin')
        build_compare_table(survey_diag, pop_diag, 'municipality_type', 'municipality_type')

        # separately for each selected canton
        for canton_key in canton_keys:
            survey_one = survey_df[survey_df['canton_id'] == canton_key]
            pop_one    = pop_df[pop_df['canton_id'].astype(str) == canton_key]
            pop_one = pop_one[pop_one['age']>14]
            print(f"\n[DIAGNOSTIC] canton_id = {canton_key}")
            build_compare_table(survey_one, pop_one, 'age_bin', 'age_bin')
            build_compare_table(survey_one, pop_one, 'municipality_type', 'municipality_type')

    pop_df['canton_id'] = pop_df['canton_id'].astype("int64")
    return pop_df